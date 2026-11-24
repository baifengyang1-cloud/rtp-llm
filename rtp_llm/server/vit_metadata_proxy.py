"""HTTP metadata forwarding and request affinity shared with the ViT RPC proxy."""

import asyncio
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class MMCacheMetadataRequest(BaseModel):
    keys: List[str] = Field(max_length=256)
    inputs: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=256)
    request_id: int = 0
    timeout_ms: int = Field(default=120000, gt=0)


@dataclass
class _Binding:
    worker: str
    expires: float
    instance: Optional[str] = None


class VitRequestRouter:
    def __init__(self, load_balancer, ttl_seconds=180.0, max_requests=65536):
        self.load_balancer = load_balancer
        self.ttl_seconds = ttl_seconds
        self.max_requests = max_requests
        self._bindings = OrderedDict()
        self._lock = threading.Lock()

    def reserve(self, request_id):
        with self._lock:
            now = time.monotonic()
            binding = self._bindings.get(request_id) if request_id else None
            if binding is not None:
                if binding.expires <= now:
                    raise RuntimeError("ViT metadata request binding expired")
                self._check_alive(binding.worker)
                return binding
            # Fixed expiration order bounds cleanup work and retained request state.
            while self._bindings:
                key, old = next(iter(self._bindings.items()))
                if old.expires > now:
                    break
                self._bindings.pop(key)
            if request_id and len(self._bindings) >= self.max_requests:
                raise RuntimeError("ViT metadata request capacity reached")
            binding = _Binding(self.load_balancer.get_worker(), now + self.ttl_seconds)
            if request_id:
                self._bindings[request_id] = binding
            return binding

    def _check_alive(self, worker):
        if worker not in self.load_balancer.get_alive_worker_addresses():
            raise RuntimeError("Bound ViT worker is unavailable; reroute metadata")

    def lookup(self, request_id):
        with self._lock:
            binding = self._bindings.get(request_id) if request_id else None
            if binding is None:
                return None
            if binding.expires <= time.monotonic():
                raise RuntimeError("ViT metadata request binding expired")
            self._check_alive(binding.worker)
            return binding.worker

    def record_instance(self, binding, instance):
        if not isinstance(instance, str) or not instance:
            raise ValueError("ViT metadata response has no worker instance")
        with self._lock:
            if binding.instance is not None and binding.instance != instance:
                raise RuntimeError("ViT worker restarted during metadata acquisition")
            binding.instance = instance

    def release(self, request_id, worker):
        with self._lock:
            binding = self._bindings.get(request_id)
            if binding is not None and binding.worker == worker:
                self._bindings.pop(request_id)


def register_proxy_metadata_routes(app, request_router, worker_http_addresses):
    workers = request_router.load_balancer.worker_addresses
    if len(workers) != len(worker_http_addresses):
        raise ValueError("ViT HTTP and gRPC worker addresses must match")
    addresses = dict(zip(workers, worker_http_addresses))
    proxy_instance = uuid.uuid4().hex
    session = None

    async def get_session():
        nonlocal session
        if session is None or session.closed:
            session = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar())
        return session

    @app.on_event("shutdown")
    async def close_metadata_session():
        if session is not None:
            await session.close()

    async def forward(worker, method, path, payload, deadline, incoming):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPException(504, "ViT metadata forwarding deadline exceeded")
        if payload is not None:
            payload = dict(payload)
            payload["timeout_ms"] = max(1, int(remaining * 1000))
        client = await get_session()

        async def exchange():
            async with client.request(
                method,
                "http://" + addresses[worker] + path,
                json=payload,
                headers={
                    "x-rtp-vit-trace-id": incoming.headers.get("x-rtp-vit-trace-id", "")
                },
                timeout=aiohttp.ClientTimeout(total=remaining),
                allow_redirects=False,
            ) as response:
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    if len(body) + len(chunk) > 16 * 1024 * 1024:
                        raise HTTPException(502, "ViT metadata response too large")
                    body.extend(chunk)
                try:
                    data = json.loads(body)
                except (ValueError, UnicodeError):
                    raise HTTPException(502, "Invalid ViT metadata response") from None
                return response.status, data

        task = asyncio.create_task(exchange())
        request_router.load_balancer.increment_connections(worker)
        try:
            while not task.done():
                if await incoming.is_disconnected():
                    raise HTTPException(499, "ViT metadata caller disconnected")
                await asyncio.wait({task}, timeout=0.025)
            return task.result()
        except asyncio.TimeoutError:
            raise HTTPException(504, "ViT metadata forwarding timed out") from None
        except aiohttp.ClientError:
            raise HTTPException(503, "ViT metadata worker unavailable") from None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            request_router.load_balancer.decrement_connections(worker)

    @app.post("/mm_cache/metadata")
    async def metadata(payload: MMCacheMetadataRequest, incoming: Request):
        if any(not key or len(key) > 4096 for key in payload.keys):
            raise HTTPException(400, "Invalid multimodal cache key")
        if payload.inputs is not None and not payload.request_id:
            raise HTTPException(400, "ViT metadata fill requires a request_id")
        if not payload.request_id:
            # Older frontends send anonymous probes, then a request-id-bearing fill.
            # Returning a miss prevents mixing one worker's hits with another's fill.
            # The worker still performs its normal cache lookup during the fill.
            return {
                "worker_instance": proxy_instance,
                "feature_hash_version": 1,
                "entries": [
                    {
                        "key": key,
                        "hit": False,
                        "hash_hit": False,
                        "embedding_hit": False,
                    }
                    for key in payload.keys
                ],
            }
        deadline = time.monotonic() + min(payload.timeout_ms / 1000.0, 120.0)
        try:
            binding = request_router.reserve(payload.request_id)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from None
        status, data = await forward(
            binding.worker,
            "POST",
            "/mm_cache/metadata",
            payload.model_dump(exclude_none=True),
            deadline,
            incoming,
        )
        if status == 200:
            try:
                request_router.record_instance(binding, data.get("worker_instance"))
            except RuntimeError as error:
                raise HTTPException(503, str(error)) from None
            except (ValueError, AttributeError):
                raise HTTPException(502, "Invalid ViT metadata response") from None
            # The public endpoint is this proxy. Child identities are checked in
            # the binding so a child restart cannot silently change an active fill.
            data = dict(data, worker_instance=proxy_instance)
        return JSONResponse(data, status_code=status)

    @app.get("/mm_cache/keys")
    @app.post("/mm_cache/keys")
    async def keys(incoming: Request):
        alive = request_router.load_balancer.get_alive_worker_addresses()
        if not alive:
            raise HTTPException(503, "No healthy ViT metadata workers")
        deadline = time.monotonic() + 2.0
        tasks = [
            asyncio.create_task(
                forward(worker, "GET", "/mm_cache/keys", None, deadline, incoming)
            )
            for worker in alive
        ]
        try:
            responses = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        result = {
            name: [] for name in ("keys", "gpu_embedding_keys", "cpu_embedding_keys")
        }
        for status, data in responses:
            if status != 200:
                return JSONResponse(data, status_code=status)
            if not isinstance(data, dict) or data.get("feature_hash_version") != 1:
                raise HTTPException(502, "Invalid ViT cache snapshot")
            for name in result:
                result[name].extend(data.get(name, []))
        for name in result:
            result[name] = list(dict.fromkeys(result[name]))[:100000]
        result["worker_instance"] = proxy_instance
        result["feature_hash_version"] = 1
        return result
