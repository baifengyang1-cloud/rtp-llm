import contextlib
import contextvars
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Union

from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics


@dataclass
class VitMetricSample:
    metric: GaugeMetrics
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class VitPreprocessMetrics:
    samples: List[VitMetricSample] = field(default_factory=list)
    request_id: int = 0

    def report(
        self,
        metric: GaugeMetrics,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.samples.append(VitMetricSample(metric, value, tags or {}))


_current_metrics: contextvars.ContextVar[Optional[VitPreprocessMetrics]] = (
    contextvars.ContextVar("vit_preprocess_metrics", default=None)
)


@contextlib.contextmanager
def collect_vit_preprocess_metrics(
    request_id: int = 0,
) -> Iterator[VitPreprocessMetrics]:
    metrics = VitPreprocessMetrics(request_id=request_id)
    token = _current_metrics.set(metrics)
    try:
        yield metrics
    finally:
        _current_metrics.reset(token)


def image_log_id(url: str) -> str:
    # Match the async downloader without logging media URLs or query strings.
    return hashlib.blake2s(
        url.encode("utf-8", "surrogatepass"), digest_size=8
    ).hexdigest()


@contextlib.contextmanager
def image_preprocess_timing(url: str) -> Iterator[dict]:
    """Log per-image CPU stages, including failures, before ViT execution."""
    with contextlib.ExitStack() as stack:
        metrics = _current_metrics.get()
        if metrics is None:
            metrics = stack.enter_context(collect_vit_preprocess_metrics())
        first_sample = len(metrics.samples)
        started = time.monotonic_ns()
        record = {
            "request_id": metrics.request_id,
            "image_id": image_log_id(url),
            "device": "cpu",
            "outcome": "ok",
        }
        try:
            yield record
        except BaseException as error:
            record.update(outcome="error", error=type(error).__name__)
            raise
        finally:
            stages = {
                GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC: "fetch_ms",
                GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC: "decode_ms",
                GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC: "resize_ms",
                GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC: "processor_ms",
            }
            for name in stages.values():
                record[name] = 0.0
            for sample in metrics.samples[first_sample:]:
                name = stages.get(sample.metric)
                if name is not None:
                    record[name] += sample.value / 1000.0
            record["worker_total_ms"] = (time.monotonic_ns() - started) / 1e6
            # Async downloads may finish before this worker starts. Their full
            # duration is logged separately by _run_preprocess_task/downloader.
            record["preprocess_ms"] = max(
                0.0, record["worker_total_ms"] - record["fetch_ms"]
            )
            logging.info(
                "vit_image_preprocess_timing %s",
                json.dumps(record, separators=(",", ":")),
            )


@contextlib.contextmanager
def vit_preprocess_timer(
    metric: GaugeMetrics,
    tags: Optional[Dict[str, str]] = None,
) -> Iterator[None]:
    start_ns = time.monotonic_ns()
    try:
        yield
    finally:
        metrics = _current_metrics.get()
        if metrics is not None:
            metrics.report(metric, (time.monotonic_ns() - start_ns) / 1000.0, tags)


def record_vit_preprocess_value(
    metric: GaugeMetrics,
    value: Union[int, float],
    tags: Optional[Dict[str, str]] = None,
) -> None:
    metrics = _current_metrics.get()
    if metrics is not None:
        metrics.report(metric, float(value), tags)


def video_resized_pixel_count(
    frame_count: int, resized_height: int, resized_width: int
) -> int:
    return int(frame_count) * int(resized_height) * int(resized_width)
