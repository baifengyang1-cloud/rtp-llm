"""Sample whole request IDs consistently across frontend and ViT processes."""

import json
import logging
import os
import queue
import threading
import time
from contextlib import contextmanager, nullcontext


try:
    SAMPLE_EVERY = max(0, int(os.getenv("VIT_TRACE_SAMPLE_EVERY", "0")))
except ValueError:
    SAMPLE_EVERY = 0


def selected(request_id):
    if not SAMPLE_EVERY or not isinstance(request_id, int) or not request_id:
        return False
    # Mix timestamp, worker and sequence bits; avoid favoring particular workers.
    value = (request_id ^ (request_id >> 32)) * 11400714819323198485
    return ((value & 0xFFFFFFFFFFFFFFFF) >> 16) % SAMPLE_EVERY == 0


def _reset_writer():
    global _events, _start_lock, _dropped
    _events = None
    _start_lock = threading.Lock()
    _dropped = 0


_reset_writer()
os.register_at_fork(after_in_child=_reset_writer)


def _write(events):
    while True:
        record = events.get()
        try:
            logging.info(
                "vit_request_trace %s", json.dumps(record, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            logging.warning(
                "vit_request_trace discarded invalid fields at %s", record["stage"]
            )


def event(request_id, stage, **fields):
    global _events, _dropped
    if not selected(request_id):
        return
    # Timestamp at the producer. File timestamps may reflect a delayed writer.
    record = dict(
        fields,
        request_id=request_id,
        stage=stage,
        pid=os.getpid(),
        wall_ns=time.time_ns(),
        mono_ns=time.monotonic_ns(),
    )
    try:
        if _events is None:
            with _start_lock:
                if _events is None:
                    events = queue.Queue(maxsize=4096)
                    threading.Thread(
                        target=_write,
                        args=(events,),
                        daemon=True,
                        name="vit-request-log",
                    ).start()
                    _events = events
        record["dropped_total"] = _dropped
        _events.put_nowait(record)
    except (queue.Full, RuntimeError):
        _dropped += 1


@contextmanager
def _stage(request_id, name, fields):
    started = time.monotonic_ns()
    event(request_id, name, phase="begin", **fields)
    try:
        yield
    except BaseException as error:
        event(
            request_id,
            name,
            phase="end",
            elapsed_ms=(time.monotonic_ns() - started) / 1e6,
            error=type(error).__name__,
            status=getattr(error, "status_code", None),
        )
        raise
    else:
        event(
            request_id,
            name,
            phase="end",
            elapsed_ms=(time.monotonic_ns() - started) / 1e6,
        )


def span(request_id, name, **fields):
    return _stage(request_id, name, fields) if selected(request_id) else nullcontext()


class ChatTraceMiddleware:
    """Observe only ASGI messages that the chat application itself consumes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in (
            "/chat/completions",
            "/v1/chat/completions",
        ):
            return await self.app(scope, receive, send)
        attempt = -time.time_ns()
        state = scope.setdefault("state", {})
        state["vit_trace_id"] = attempt
        state["vit_arrival_ns"] = -attempt
        state["vit_arrival_mono_ns"] = time.monotonic_ns()
        event(attempt, "frontend.http_enter")
        status = None

        async def observe_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                event(state["vit_trace_id"], "frontend.disconnect_observed")
            return message

        async def observe_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                event(state["vit_trace_id"], "frontend.response_headers", status=status)
            await send(message)

        try:
            return await self.app(scope, observe_receive, observe_send)
        finally:
            event(
                state["vit_trace_id"],
                "frontend.http_exit",
                status=status,
                arrival_wall_ns=-attempt,
                elapsed_ms=(time.monotonic_ns() - state["vit_arrival_mono_ns"]) / 1e6,
            )


def metadata_http_trace():
    import aiohttp

    trace = aiohttp.TraceConfig()
    for signal, stage in (
        (trace.on_connection_queued_start, "pool_wait_begin"),
        (trace.on_connection_queued_end, "pool_wait_end"),
        (trace.on_connection_create_start, "connect_begin"),
        (trace.on_connection_create_end, "connect_end"),
        (trace.on_connection_reuseconn, "connection_reused"),
        (trace.on_request_headers_sent, "headers_sent"),
    ):

        async def observed(session, context, params, stage=stage):
            if context.trace_request_ctx is not None:
                request_id, kind = context.trace_request_ctx
                event(
                    request_id, "frontend.metadata_transport", kind=kind, action=stage
                )

        signal.append(observed)
    return trace


class MetadataTraceMiddleware:
    """Includes request parsing and response serialization in the HTTP boundary."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/mm_cache/metadata":
            return await self.app(scope, receive, send)
        wire_id = dict(scope["headers"]).get(b"x-rtp-vit-trace-id", b"")
        if len(wire_id) > 20 or not wire_id.isdigit():
            return await self.app(scope, receive, send)
        request_id = int(wire_id)
        scope.setdefault("state", {})["vit_metadata_trace_id"] = request_id

        async def observe_send(message):
            if message["type"] == "http.response.start":
                event(request_id, "vit.metadata_response", status=message["status"])
            await send(message)

        with span(request_id, "vit.metadata_http"):
            return await self.app(scope, receive, observe_send)
