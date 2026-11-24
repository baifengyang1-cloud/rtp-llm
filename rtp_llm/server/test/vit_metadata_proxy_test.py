import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from aiohttp import web
from fastapi import FastAPI

from rtp_llm.server.vit_metadata_proxy import VitRequestRouter


class _Balancer:
    def __init__(self):
        self.worker_addresses = ["worker-0", "worker-1"]
        self.alive = set(self.worker_addresses)
        self.index = 0
        self.connections = {worker: 0 for worker in self.worker_addresses}

    def get_alive_worker_addresses(self):
        return [w for w in self.worker_addresses if w in self.alive]

    def get_worker(self):
        workers = self.get_alive_worker_addresses()
        if not workers:
            raise RuntimeError("No healthy workers")
        worker = workers[self.index % len(workers)]
        self.index += 1
        return worker

    def increment_connections(self, worker):
        self.connections[worker] += 1

    def decrement_connections(self, worker):
        self.connections[worker] -= 1


class VitMetadataProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.balancer = _Balancer()
        self.router = VitRequestRouter(self.balancer)
        self.states = []
        self.runners = []
        addresses = []
        for i in range(2):
            state = {"instance": "epoch-" + str(i), "calls": [], "delay": 0}
            self.states.append(state)

            async def metadata(request, state=state):
                payload = await request.json()
                state["calls"].append(payload)
                await asyncio.sleep(state["delay"])
                if "overload" in payload["keys"]:
                    return web.json_response(
                        {"detail": {"error_code": 409, "message": "full"}}, status=503
                    )
                return web.json_response(
                    {
                        "worker_instance": state["instance"],
                        "feature_hash_version": 1,
                        "entries": [
                            {"key": key, "hash_hit": bool(payload.get("inputs"))}
                            for key in payload["keys"]
                        ],
                    }
                )

            async def keys(request, state=state):
                return web.json_response(
                    {
                        "worker_instance": state["instance"],
                        "feature_hash_version": 1,
                        "keys": [state["instance"]],
                        "gpu_embedding_keys": [state["instance"]],
                        "cpu_embedding_keys": [],
                    }
                )

            worker = web.Application()
            worker.router.add_post("/mm_cache/metadata", metadata)
            worker.router.add_get("/mm_cache/keys", keys)
            runner = web.AppRunner(worker)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self.runners.append(runner)
            addresses.append(
                "127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
            )
        from rtp_llm.multimodal.vit_proxy_start_server import create_proxy_app

        self.app = create_proxy_app(
            SimpleNamespace(request_router=self.router),
            self.balancer.worker_addresses,
            addresses,
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://proxy"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        for shutdown in self.app.router.on_shutdown:
            await shutdown()
        for runner in self.runners:
            await runner.cleanup()
        self.assertEqual(list(self.balancer.connections.values()), [0, 0])

    async def test_probe_and_partial_fill_pin_same_rpc_worker(self):
        probe = await self.client.post(
            "/mm_cache/metadata", json={"keys": ["a", "b"], "request_id": 100}
        )
        fill = await self.client.post(
            "/mm_cache/metadata",
            json={"keys": ["b"], "inputs": [{}], "request_id": 100},
        )
        self.assertEqual(probe.status_code, 200, probe.text)
        self.assertEqual(fill.status_code, 200, fill.text)
        self.assertEqual(
            probe.json()["worker_instance"], fill.json()["worker_instance"]
        )
        self.assertEqual(len(self.states[0]["calls"]), 2)
        self.assertEqual(self.states[1]["calls"], [])
        self.assertEqual(self.router.lookup(100), "worker-0")
        self.assertEqual(fill.json()["entries"], [{"key": "b", "hash_hit": True}])
        self.router.release(100, "worker-0")
        self.assertIsNone(self.router.lookup(100))

    async def test_old_frontend_anonymous_probe_then_fill(self):
        probe = await self.client.post("/mm_cache/metadata", json={"keys": ["a"]})
        self.assertEqual(probe.status_code, 200)
        self.assertFalse(probe.json()["entries"][0]["hash_hit"])
        self.assertTrue(all(not s["calls"] for s in self.states))
        fill = await self.client.post(
            "/mm_cache/metadata",
            json={"keys": ["a"], "inputs": [{}], "request_id": 101},
        )
        self.assertEqual(fill.status_code, 200)
        self.assertEqual(
            probe.json()["worker_instance"], fill.json()["worker_instance"]
        )
        self.assertEqual(self.router.lookup(101), "worker-0")

    async def test_worker_error_status_and_detail_are_preserved(self):
        result = await self.client.post(
            "/mm_cache/metadata", json={"keys": ["overload"], "request_id": 2}
        )
        self.assertEqual(result.status_code, 503)
        self.assertEqual(
            result.json()["detail"], {"error_code": 409, "message": "full"}
        )

    async def test_worker_restart_does_not_merge_generations(self):
        payload = {"keys": ["a"], "request_id": 3}
        self.assertEqual(
            (await self.client.post("/mm_cache/metadata", json=payload)).status_code,
            200,
        )
        self.states[0]["instance"] = "restarted"
        response = await self.client.post(
            "/mm_cache/metadata", json={**payload, "inputs": [{}]}
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("restarted", response.text)
        self.assertEqual(self.states[1]["calls"], [])

    async def test_dead_bound_worker_is_not_reassigned(self):
        self.router.reserve(4)
        self.balancer.alive.remove("worker-0")
        response = await self.client.post(
            "/mm_cache/metadata", json={"keys": ["a"], "request_id": 4}
        )
        self.assertEqual(response.status_code, 503)
        self.assertTrue(all(not s["calls"] for s in self.states))

    async def test_deadline_cancels_outgoing_request(self):
        self.states[0]["delay"] = 0.2
        response = await self.client.post(
            "/mm_cache/metadata",
            json={"keys": ["a"], "request_id": 5, "timeout_ms": 20},
        )
        self.assertEqual(response.status_code, 504)
        self.assertLessEqual(self.states[0]["calls"][0]["timeout_ms"], 20)

    async def test_caller_cancellation_releases_connection(self):
        self.states[0]["delay"] = 0.2
        task = asyncio.create_task(
            self.client.post(
                "/mm_cache/metadata", json={"keys": ["a"], "request_id": 6}
            )
        )
        for _ in range(100):
            if self.states[0]["calls"]:
                break
            await asyncio.sleep(0.002)
        self.assertTrue(self.states[0]["calls"])
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.balancer.connections["worker-0"], 0)

    async def test_concurrent_requests_keep_independent_bindings(self):
        responses = await asyncio.gather(
            *(
                self.client.post(
                    "/mm_cache/metadata", json={"keys": ["a"], "request_id": i}
                )
                for i in (10, 11)
            )
        )
        self.assertTrue(all(r.status_code == 200 for r in responses))
        self.assertEqual(self.router.lookup(10), "worker-0")
        self.assertEqual(self.router.lookup(11), "worker-1")

    async def test_keys_snapshot_get_and_post_are_aggregated(self):
        for method in [self.client.get, self.client.post]:
            response = await method("/mm_cache/keys")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["keys"], ["epoch-0", "epoch-1"])
            self.assertEqual(
                response.json()["gpu_embedding_keys"], ["epoch-0", "epoch-1"]
            )

    async def test_invalid_request_is_rejected_before_forwarding(self):
        for payload in [{"keys": [""]}, {"keys": ["a"], "inputs": [{}]}]:
            response = await self.client.post("/mm_cache/metadata", json=payload)
            self.assertEqual(response.status_code, 400)
        self.assertTrue(all(not s["calls"] for s in self.states))


class VitRequestRouterTest(unittest.TestCase):
    def test_capacity_does_not_evict_live_bindings(self):
        router = VitRequestRouter(_Balancer(), max_requests=1)
        first = router.reserve(1)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            router.reserve(2)
        self.assertIs(router.reserve(1), first)
        router.release(1, first.worker)
        self.assertIsNotNone(router.reserve(2))

    def test_expired_request_cannot_silently_move_to_another_worker(self):
        router = VitRequestRouter(_Balancer(), ttl_seconds=1)
        router.reserve(1)
        with patch(
            "rtp_llm.server.vit_metadata_proxy.time.monotonic",
            return_value=float("inf"),
        ):
            with self.assertRaisesRegex(RuntimeError, "expired"):
                router.lookup(1)


class VitChildMetadataRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_mode_child_serves_metadata_and_keys(self):
        from rtp_llm.server.vit_app import register_mm_cache_routes

        engine = SimpleNamespace(
            is_proxy_mode=True,
            _embedding_cache=SimpleNamespace(
                instance_id="child-epoch",
                metadata=lambda keys: {"worker_instance": "child-epoch", "entries": []},
                metadata_keys=lambda: ["a"],
                resident_tiers=lambda **kwargs: {"a": "cpu"},
            ),
        )
        app = FastAPI()
        register_mm_cache_routes(app, engine)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://child"
        ) as client:
            metadata = await client.post("/mm_cache/metadata", json={"keys": ["a"]})
            keys = await client.get("/mm_cache/keys")
        self.assertEqual(metadata.status_code, 200, metadata.text)
        self.assertEqual(metadata.json()["worker_instance"], "child-epoch")
        self.assertEqual(keys.status_code, 200, keys.text)
        self.assertEqual(keys.json()["cpu_embedding_keys"], ["a"])


if __name__ == "__main__":
    unittest.main()
