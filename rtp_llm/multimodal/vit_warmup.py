"""Warm the serving executor before exposing a ViT worker's endpoints."""

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager

import torch


@contextmanager
def _device_warmup_lock(device, timeout_s):
    # All local ViT processes share /tmp. Serialize cold work on the same
    # accelerator without holding the lock during normal inference.
    index = torch.device(device).index
    path = f"/tmp/rtp_llm_vit_warmup_{os.getuid()}_{index}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    started = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError("ViT startup warmup lock timed out") from None
                time.sleep(0.05)
        logging.info(
            "vit_warmup_lock device=%s wait_ms=%.3f",
            device,
            (time.monotonic() - started) * 1000,
        )
        yield
    finally:
        os.close(fd)


def warmup_vit(mm_part, device, max_batch_images, timeout_s):
    from rtp_llm.ops import ensure_engine_ops_loaded, get_multimodal_feature_hash
    from rtp_llm.utils.base_model_datatypes import MMUrlType

    started = time.monotonic()
    with _device_warmup_lock(device, timeout_s), torch.inference_mode():
        step = time.monotonic()
        ensure_engine_ops_loaded()
        logging.info(
            "vit_warmup_native pid=%d elapsed_ms=%.3f",
            os.getpid(),
            (time.monotonic() - step) * 1000,
        )
        batches = getattr(mm_part, "get_warmup_batches", None)
        if batches is None:
            logging.info(
                "vit_warmup_skipped reason=model_has_no_warmup_inputs model=%s",
                type(mm_part).__name__,
            )
            return
        repeats = int(os.getenv("RTP_LLM_VIT_WARMUP_REPEATS", "2"))
        if not 1 <= repeats <= 10:
            raise ValueError("RTP_LLM_VIT_WARMUP_REPEATS must be in [1, 10]")
        count = 0
        for description, data in batches(max_batch_images):
            for repeat in range(repeats):
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError("ViT startup warmup exceeded its budget")
                step = time.monotonic()
                outputs = mm_part.batched_embedding(data, [MMUrlType.IMAGE] * len(data))
                if len(outputs) != len(data):
                    raise RuntimeError("ViT warmup output count mismatch")
                for output in outputs:
                    get_multimodal_feature_hash(output[0])
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                logging.info(
                    "vit_warmup_batch %s",
                    json.dumps(
                        dict(
                            pid=os.getpid(),
                            shapes=description,
                            repeat=repeat,
                            elapsed_ms=(time.monotonic() - step) * 1000,
                        )
                    ),
                )
                del outputs
            count += 1
        if count == 0:
            raise RuntimeError("ViT warmup produced no supported batches")
    logging.info(
        "vit_warmup_ready pid=%d batches=%d elapsed_ms=%.3f",
        os.getpid(),
        count,
        (time.monotonic() - started) * 1000,
    )
