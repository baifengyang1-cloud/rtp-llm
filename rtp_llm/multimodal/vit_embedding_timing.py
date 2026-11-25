"""Structured ViT stage timings, independent of native engine/metrics imports."""

import contextlib
import contextvars
import json
import logging
import os
import time
from typing import Iterator, Optional


_STAGE_TIMING_SYNC = os.getenv("RTP_LLM_VIT_STAGE_TIMING_SYNC", "0") == "1"
_embedding_batch: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "vit_embedding_batch", default=None
)


@contextlib.contextmanager
def embedding_batch_context(batch_id: str, **fields) -> Iterator[None]:
    token = _embedding_batch.set(dict(fields, batch_id=batch_id))
    try:
        yield
    finally:
        _embedding_batch.reset(token)


@contextlib.contextmanager
def embedding_stage_timing(
    stage: str, *, device_sync: bool = False, **fields
) -> Iterator[dict]:
    """Log wall time without adding device synchronization by default.

    The optional diagnostic mode drains the current device before/after the
    vision call. Report the initial drain separately: it belongs to preceding
    work, not this forward. It changes overlap and must not be used as a normal
    performance benchmark. Hash/cache timings include their existing waits.
    """
    record = dict(batch_id=None)
    record.update(_embedding_batch.get() or {})
    record.update(fields)
    record.update(
        stage=stage,
        pid=os.getpid(),
        timing_mode=(
            "device_sync_requested" if device_sync and _STAGE_TIMING_SYNC else "host"
        ),
        outcome="ok",
        pre_sync_ms=0.0,
        device_sync_ms=0.0,
    )
    synchronize = None
    started = time.monotonic_ns()
    logging.info(
        "vit_embedding_stage_timing %s",
        json.dumps(dict(record, phase="begin", wall_ns=time.time_ns())),
    )
    host_finished = None
    try:
        if device_sync and _STAGE_TIMING_SYNC:
            import torch

            record["timing_mode"] = "host"
            if torch.cuda.is_available():
                synchronize = torch.cuda.synchronize
                record["timing_mode"] = "device_sync"
                before_sync = time.monotonic_ns()
                try:
                    synchronize()
                finally:
                    record["pre_sync_ms"] = (time.monotonic_ns() - before_sync) / 1e6
                started = time.monotonic_ns()
        yield record
        host_finished = time.monotonic_ns()
        if synchronize is not None:
            try:
                synchronize()
            finally:
                record["device_sync_ms"] = (time.monotonic_ns() - host_finished) / 1e6
    except BaseException as error:
        record.update(outcome="error", error=type(error).__name__)
        raise
    finally:
        finished = time.monotonic_ns()
        record.update(
            phase="end",
            wall_ns=time.time_ns(),
            host_ms=(
                (host_finished if host_finished is not None else finished) - started
            ) / 1e6,
            elapsed_ms=(finished - started) / 1e6,
        )
        logging.info(
            "vit_embedding_stage_timing %s",
            json.dumps(record, separators=(",", ":")),
        )
