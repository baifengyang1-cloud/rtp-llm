import asyncio
import concurrent.futures
import io
import json
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import List
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import PIL
import pillow_avif
import pillow_heif
import torch
from PIL import Image, ImageFile

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    ProfilingDebugLoggingConfig,
    PyEnvConfigs,
    VitConfig,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputPB,
    MultimodalInputsPB,
)
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import (
    GreenNetHandle,
    GreenNetProvider,
    GreenNetVerdict,
)
from rtp_llm.multimodal.mm_error_messages import MMErr
from rtp_llm.multimodal.mm_process_engine import (
    MMEmbeddingAsyncCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
    MMProcessEngine,
    MMWorkItem,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin import (
    Qwen2_VLImageEmbedding,
)
from rtp_llm.multimodal.multimodal_util import vit_emb_cache_
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


class FakeMultiModalEmbeddingInterface(Qwen2_VLImageEmbedding):
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.data_type = config.compute_dtype
        self.image_processor: Qwen2VLImageProcessor = (
            Qwen2VLImageProcessor.from_pretrained(
                "./rtp_llm/multimodal/test/testdata/qwen2_vl/"
            )
        )
        self.spatial_merge_size = 2

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        return torch.tensor([[0]]), None

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class PreprcoesException(Exception):
    pass


class FakeMultiModalEmbeddingInterfacePreprocessException(
    FakeMultiModalEmbeddingInterface
):
    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        raise PreprcoesException(kwargs)

    def get_preprocess_params(self):
        return {"test": "hello"}


class FakeMultiModalEmbeddingInterfaceSlow(FakeMultiModalEmbeddingInterface):
    """Preprocess function that sleeps to guarantee timeout."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        time.sleep(5)
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}


class FakeMultiModalEmbeddingInterfaceSlowEmbedding(FakeMultiModalEmbeddingInterface):
    """batched_embedding sleeps, to exercise the embedding-level timeout on the
    default (non-gpu-batch) serial path."""

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        time.sleep(5)
        return [(torch.tensor([[0]]), None) for _ in data_list]


class FakeMultiModalEmbeddingInterfaceProcessCrash(FakeMultiModalEmbeddingInterface):
    """Preprocess function that crashes the worker process to trigger BrokenProcessPool."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        os._exit(1)

    def get_preprocess_params(self):
        return {}


class FakeMultiModalEmbeddingInterfaceBadCount(FakeMultiModalEmbeddingInterface):
    """batched_embedding returns the wrong number of outputs."""

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        # One fewer than requested, to trip the count guard.
        return [(torch.tensor([[0]]), None) for _ in range(len(data_list) - 1)]


class FakeEmbeddingLengthInterface(FakeMultiModalEmbeddingInterface):
    """Return distinct token counts so request-level aggregation is testable."""

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput], vit_config: VitConfig, **kwargs
    ):
        return torch.tensor([len(mm_inputs[0].url)])

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        return [
            (torch.zeros((int(data.reshape(-1)[0]), 4)), None) for data in data_list
        ]


class FakeModel:
    def __init__(self, mm_part: MultiModalEmbeddingInterface = None):
        self.model_config = ModelConfig()
        self.model_config.mm_model_config.mm_position_ids_style = 2
        self.mm_part = mm_part


class MMProcessEngineTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = FakeModel(FakeMultiModalEmbeddingInterface())
        self.mm_process_engine = MMProcessEngine(
            self.model.mm_part,
            self.model.model_config,
            VitConfig(),
            ProfilingDebugLoggingConfig(),
        )

    def test_embedding(self):
        res = self.mm_process_engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

        mm_inputs = MultimodalInputsPB()
        mm_input = MultimodalInputPB()
        mm_input.multimodal_url = "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"
        mm_input.multimodal_type = MMUrlType.IMAGE
        mm_input.mm_preprocess_config.mm_timeout_ms = 30000
        mm_inputs.multimodal_inputs.append(mm_input)
        res = self.mm_process_engine.mm_embedding_rpc(mm_inputs)
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        self.assertEqual(res.position_ids, [])

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_embedding_length_metric_sums_a_request(self, report):
        """A multi-image request emits one gauge containing all visual tokens."""
        model = FakeModel(FakeEmbeddingLengthInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.use_gpu_batch = True
        vit_config.gpu_batch_wait_ms = 100
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        inputs = [
            MultimodalInput("a", MMUrlType.IMAGE, torch.empty(0), config),
            MultimodalInput("bb", MMUrlType.IMAGE, torch.empty(0), config),
        ]
        try:
            result = engine.mm_embedding_impl(inputs)
        finally:
            engine.stop()

        self.assertEqual(
            [embedding.shape[0] for embedding in result.embeddings], [1, 2]
        )
        lengths = [
            call.args[1]
            for call in report.call_args_list
            if call.args and call.args[0] == GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC
        ]
        self.assertEqual(lengths, [3])

        image_counts = [
            call.args[1]
            for call in report.call_args_list
            if call.args and call.args[0] == GaugeMetrics.VIT_IMAGE_COUNT_METRIC
        ]
        self.assertEqual(image_counts, [2])

    def test_timeout(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        vit_config = VitConfig()
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        with self.assertRaises(TimeoutError):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 1],
                ],
            )
        engine.stop()

    def test_preprocess(self):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        vit_config = VitConfig()
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        mm_process_engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            mm_process_engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [
                    [-1, -1, -1, -1, -1, -1, -1, [], 30000],
                ],
            )
        except PreprcoesException as e:
            self.assertEqual(str(e), "{'test': 'hello'}")

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_error_qps_is_reported_for_preprocess_failure(self, report):
        model = FakeModel(FakeMultiModalEmbeddingInterfacePreprocessException())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(PreprcoesException):
                engine.mm_embedding_cpp(
                    ["fake://error-qps-preprocess"],
                    [MMUrlType.IMAGE],
                    [torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
                )
        finally:
            engine.stop()

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)

    def test_local_preprocess_mode(self):
        """LocalPreprocessExecutor path: use_local_preprocess=True bypasses the worker pool."""
        model = FakeModel(FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_query_counter(self):
        self.assertEqual(self.mm_process_engine.get_query_num(), 0)
        self.mm_process_engine.inc_query_num()
        self.mm_process_engine.inc_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 2)
        self.mm_process_engine.dec_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 1)
        self.mm_process_engine.dec_query_num()
        self.assertEqual(self.mm_process_engine.get_query_num(), 0)

    def test_work_item_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            MMWorkItem([])

    def test_work_item_uses_global_timeout_when_request_timeout_is_unset(self):
        preprocess_config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], -1)
        mm_input = MultimodalInput(
            "", MMUrlType.IMAGE, torch.empty(0), preprocess_config
        )

        self.assertEqual(preprocess_config.mm_timeout_ms, -1)
        self.assertEqual(
            MMWorkItem([mm_input], mm_timeout_ms=123000).mm_timeout_ms, 123000
        )

    def test_multimodal_cache_key_ignores_timeout(self):
        def make_input(width: int, timeout_ms: int) -> MultimodalInput:
            preprocess_config = MMPreprocessConfig(
                width, 480, 100, 1000, 2, 1, 64, [0.25, 0.75], timeout_ms
            )
            return MultimodalInput(
                "https://example.com/image.jpg",
                MMUrlType.IMAGE,
                torch.empty(0),
                preprocess_config,
            )

        self.assertEqual(
            make_input(640, 30000).cache_key(),
            make_input(640, 120000).cache_key(),
        )
        self.assertNotEqual(
            make_input(640, 30000).cache_key(),
            make_input(800, 30000).cache_key(),
        )

    def test_embedding_timeout_default_path(self):
        """Default (non-gpu-batch) serial path enforces an embedding-level timeout.

        Regression guard for the timeout semantic migrated from the old inline
        path: a slow batched_embedding must surface as TimeoutError, not hang.
        """
        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlowEmbedding())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True  # fast preprocess; isolate embedding
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = (
            0  # no cache hit to short-circuit the forward
        )
        vit_config.mm_cache_gpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(TimeoutError):
                engine.mm_embedding_cpp(
                    ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                    [MMUrlType.IMAGE],
                    [torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 100]],  # mm_timeout_ms=100
                )
        finally:
            engine.stop()

    def test_batched_embedding_count_mismatch(self):
        """Serial-mode scheduler path fails fast when batched_embedding returns wrong count."""
        model = FakeModel(FakeMultiModalEmbeddingInterfaceBadCount())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True  # local preprocess, serial scheduler
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        try:
            with self.assertRaises(RuntimeError):
                engine.mm_embedding_cpp(
                    ["url0", "url1"],
                    [MMUrlType.IMAGE, MMUrlType.IMAGE],
                    [torch.empty(0), torch.empty(0)],
                    [[-1, -1, -1, -1, -1, -1, -1, [], 30000]] * 2,
                )
        finally:
            engine.stop()

    def test_worker_crash_recovery(self):
        """Pool rebuilds after worker process crash and subsequent requests succeed."""
        model = FakeModel(FakeMultiModalEmbeddingInterfaceProcessCrash())
        vit_config = VitConfig()
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

        # First call crashes the worker — should raise but pool rebuilds internally
        with self.assertRaises(Exception):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 5000]],
            )

        # Swap to a working mm_part so the rebuilt pool can serve requests
        working_model = FakeModel(FakeMultiModalEmbeddingInterface())
        engine.preprocess_executor.preprocess_func = (
            working_model.mm_part.preprocess_input
        )
        engine.preprocess_executor._rebuild_pool()

        # Subsequent request should succeed after pool recovery
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_consecutive_timeout_triggers_rebuild(self):
        """Pool rebuilds after consecutive timeouts reach the threshold."""
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        model = FakeModel(FakeMultiModalEmbeddingInterfaceSlow())
        vit_config = VitConfig()
        vit_config.mm_preprocess_max_workers = 2
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

        executor = engine.preprocess_executor
        if not isinstance(executor, MultiprocessPreprocessExecutor):
            self.skipTest("Not using multiprocess executor")

        old_pool = executor.pool

        # Simulate consecutive timeouts reaching the threshold
        executor._consecutive_timeouts = executor._max_consecutive_timeouts - 1

        # This timeout should trigger a rebuild
        with self.assertRaises(TimeoutError):
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 1]],
            )

        # Pool should have been rebuilt
        self.assertIsNot(executor.pool, old_pool)
        self.assertEqual(executor._consecutive_timeouts, 0)
        engine.stop()


class PreprocessMetricTest(TestCase):
    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_preprocess_queue_metric_tracks_pending_tasks(self, report):
        from rtp_llm.multimodal.mm_process_engine import MultiprocessPreprocessExecutor

        class FakePool:
            def __init__(self):
                self.callbacks = []

            def apply_async(self, *args, **kwargs):
                self.callbacks.append((kwargs["callback"], kwargs["error_callback"]))
                return object()

        executor = object.__new__(MultiprocessPreprocessExecutor)
        executor.pool = FakePool()
        executor._pool_lock = threading.Lock()
        executor._preprocess_queue_lock = threading.Lock()
        executor._pending_preprocess_tasks = set()
        executor._next_preprocess_task_id = 0

        config = MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000)
        work_items = [
            MMWorkItem(
                [
                    MultimodalInput(
                        f"fake://queue-{index}",
                        MMUrlType.IMAGE,
                        torch.empty(0),
                        config,
                    )
                ],
                mm_timeout_ms=30000,
            )
            for index in range(2)
        ]

        executor.submit(work_items[0])
        executor.submit(work_items[1])
        depth_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC
        ]
        self.assertEqual(depth_values[-1], 2)

        executor.pool.callbacks[0][0](None)
        executor.pool.callbacks[1][1](RuntimeError("preprocess failed"))
        depth_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC
        ]
        self.assertEqual(depth_values[-1], 0)


class FakeSlowEmbeddingInterface(FakeMultiModalEmbeddingInterface):
    """Embedding that takes a configurable delay, for testing async concurrency."""

    delay = 0.3

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        time.sleep(self.delay)
        return torch.tensor([[1]]), None


class MMEmbeddingCacheEntryTest(TestCase):
    def test_complete_then_wait(self):
        entry = MMEmbeddingCacheEntry()
        self.assertFalse(entry.is_done)
        entry.complete("result_value")
        self.assertTrue(entry.is_done)
        self.assertEqual(entry.wait(), "result_value")

    def test_wait_blocks_until_complete(self):
        entry = MMEmbeddingCacheEntry()
        result_holder = [None]

        def setter():
            time.sleep(0.1)
            entry.complete(42)

        threading.Thread(target=setter, daemon=True).start()
        result_holder[0] = entry.wait(timeout=5.0)
        self.assertEqual(result_holder[0], 42)

    def test_wait_timeout(self):
        entry = MMEmbeddingCacheEntry()
        with self.assertRaises(TimeoutError):
            entry.wait(timeout=0.05)

    def test_fail_then_wait_raises(self):
        entry = MMEmbeddingCacheEntry()
        entry.fail(ValueError("boom"))
        self.assertTrue(entry.is_done)
        with self.assertRaises(ValueError):
            entry.wait()


class MMEmbeddingAsyncCacheTest(TestCase):

    def test_lookup_during_failure_publication_recomputes(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=4096, cpu_max_bytes=4096)
        _, entry = cache.try_acquire("failed")
        published, release = threading.Event(), threading.Event()
        original = entry._on_fail

        def delayed_remove(failed, error):
            published.set()
            if not release.wait(5):
                raise TimeoutError("failure removal barrier was not released")
            original(failed, error)

        entry._on_fail = delayed_remove
        producer = threading.Thread(target=entry.fail, args=(ValueError("failed"),))
        producer.start()
        try:
            self.assertTrue(published.wait(5))
            state, replacement = cache.try_acquire("failed")
            self.assertEqual(state, "miss")
            self.assertIsNot(replacement, entry)
            release.set()
            producer.join(5)
            self.assertIs(cache.peek("failed"), replacement)
            with self.assertRaisesRegex(ValueError, "failed"):
                entry.wait(1)
        finally:
            release.set()
            producer.join(5)

    def test_metadata_is_read_only_and_tracks_eviction(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=48)
        hash_cache = MMHashKeyCache(max_bytes=4096)
        _, first = cache.try_acquire("a")
        _, second = cache.try_acquire("b")
        hashes = [torch.tensor([-1, 2147483647], dtype=torch.int32)]
        first.complete((torch.ones(2, 4), None))
        second_hashes = [torch.tensor([7], dtype=torch.int32)]
        second.complete((torch.ones(1, 4), None))
        hash_cache.put("a", hashes, first.generation)
        hash_cache.put("b", second_hashes, second.generation)
        before = cache.stats()
        metadata = hash_cache.metadata(["a", "a", "absent"], cache)
        self.assertEqual(metadata["entries"][0]["feature_hashes"], [-1, 2147483647])
        self.assertEqual(metadata["entries"][0]["split_size"], [2])
        self.assertEqual(metadata["entries"][0], metadata["entries"][1])
        self.assertFalse(metadata["entries"][2]["hit"])
        self.assertEqual(before, cache.stats())
        # Feature hashes live in the routing-key sidecar and are not charged
        # to the embedding tensor budget.
        self.assertEqual(before["resident_bytes"], 2 * 4 * 4 + 1 * 4 * 4)
        _, third = cache.try_acquire("c")
        third.complete((torch.ones(2, 4), None))
        self.assertIsNone(cache.peek("a"))
        self.assertEqual(hash_cache.keys(), ["a", "b"])
        evicted = hash_cache.metadata(["a"], cache)["entries"][0]
        self.assertTrue(evicted["hash_hit"])
        self.assertFalse(evicted["embedding_hit"])
        cache.clear()
        self.assertEqual(hash_cache.keys(), ["a", "b"])

    def test_hash_key_cache_is_generation_aware(self):
        cache = MMHashKeyCache(max_bytes=4096)
        hashes = [torch.tensor([-1, 2147483647], dtype=torch.int32)]
        cache.put("key", hashes, "generation-1")
        self.assertTrue(cache.contains("key"))
        self.assertTrue(torch.equal(cache.get("key", "generation-1")[0], hashes[0]))
        self.assertIsNone(cache.get("key", "generation-2"))
        cache.put("key", [torch.tensor([7], dtype=torch.int32)], "generation-2")
        self.assertTrue(
            torch.equal(
                cache.get("key", "generation-2")[0],
                torch.tensor([7], dtype=torch.int32),
            )
        )

    def test_embedding_eviction_keeps_hash_key(self):
        hash_keys = MMHashKeyCache(max_bytes=4096)
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=16)
        _, first = cache.try_acquire("first")
        first.complete((torch.ones(1, 4), None), [torch.tensor([1])])
        hash_keys.put("first", [torch.tensor([1], dtype=torch.int32)], first.generation)

        _, second = cache.try_acquire("second")
        self.assertTrue(hash_keys.contains("first"))
        self.assertTrue(
            torch.equal(
                hash_keys.get("first", first.generation)[0],
                torch.tensor([1], dtype=torch.int32),
            )
        )
        second.complete((torch.ones(1, 4), None), [torch.tensor([2])])
        hash_keys.put(
            "second", [torch.tensor([2], dtype=torch.int32)], second.generation
        )
        self.assertEqual(hash_keys.keys(), ["first", "second"])
        metadata = hash_keys.metadata(["first"], cache)
        self.assertTrue(metadata["entries"][0]["hash_hit"])
        self.assertFalse(metadata["entries"][0]["embedding_hit"])

    def test_failed_and_legacy_results_have_no_metadata(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=4096)
        _, error = cache.try_acquire("error")
        error.fail(ValueError("failed"))
        _, legacy = cache.try_acquire("legacy")
        legacy.complete((torch.ones(1, 4), None))
        self.assertEqual(cache.metadata_keys(), [])
        self.assertTrue(
            all(not e["hit"] for e in cache.metadata(["error", "legacy"])["entries"])
        )

    def test_miss_then_complete_then_hit(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=4096)
        state, entry = cache.try_acquire("key1")
        self.assertEqual(state, "miss")
        self.assertFalse(entry.is_done)

        entry.complete("val1")

        state2, entry2 = cache.try_acquire("key1")
        self.assertEqual(state2, "complete")
        self.assertIs(entry2, entry)
        self.assertEqual(entry2.wait(), "val1")

    def test_in_progress_state(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=4096)
        state, entry = cache.try_acquire("key1")
        self.assertEqual(state, "miss")

        state2, entry2 = cache.try_acquire("key1")
        self.assertEqual(state2, "in_progress")
        self.assertIs(entry2, entry)

    def test_remove(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=4096)
        _, entry = cache.try_acquire("key1")
        entry.complete("v")
        cache.remove("key1")

        state, entry2 = cache.try_acquire("key1")
        self.assertEqual(state, "miss")
        self.assertIsNot(entry2, entry)

    def test_eviction(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=8)
        _, e1 = cache.try_acquire("k1")
        e1.complete(torch.tensor([1.0]))
        _, e2 = cache.try_acquire("k2")
        e2.complete(torch.tensor([2.0]))
        _, e3 = cache.try_acquire("k3")

        # Pending entries do not consume tensor bytes or evict completed work.
        self.assertEqual(cache.stats()["resident_bytes"], 8)
        self.assertEqual(cache.try_acquire("k3")[0], "in_progress")
        e3.complete(torch.tensor([3.0]))
        self.assertNotIn("k1", cache._entries)
        self.assertEqual(cache.stats()["resident_entries"], 2)
        self.assertEqual(cache.stats()["resident_bytes"], 8)
        self.assertEqual(e1.wait().item(), 1.0)

    def test_resize(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, first = cache.try_acquire("first")
        first.complete(torch.ones(4))
        _, second = cache.try_acquire("second")
        second.complete(torch.ones(4))
        cache.resize(0, 16)
        self.assertIsNone(cache.peek("first"))
        self.assertIs(cache.peek("second"), second)
        self.assertEqual(cache.stats()["resident_bytes"], 16)

    def test_weighted_lru_evicts_by_actual_tensor_bytes(self):
        cache = MMEmbeddingAsyncCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, e1 = cache.try_acquire("k1")
        e1.complete((torch.zeros((2, 2)), None))  # 16 bytes
        _, e2 = cache.try_acquire("k2")
        e2.complete((torch.zeros((2, 2)), None))  # 16 bytes

        # Both entries fit within the byte budget.
        self.assertEqual(cache.stats()["resident_entries"], 2)
        self.assertEqual(cache.try_acquire("k1")[0], "complete")

        _, e3 = cache.try_acquire("k3")
        e3.complete((torch.zeros((2, 2)), None))

        # k1 was touched, so k2 is the least-recently-used completed entry.
        self.assertNotIn("k2", cache._entries)
        self.assertIn("k1", cache._entries)
        self.assertIn("k3", cache._entries)
        stats = cache.stats()
        self.assertEqual(stats["resident_bytes"], 32)
        self.assertEqual(stats["resident_tokens"], 4)
        self.assertEqual(stats["eviction"], 1)

    @patch("rtp_llm.multimodal.mm_embedding_cache.kmonitor.report")
    def test_cache_token_metric_tracks_eviction_and_one_dimensional_embedding(
        self, report
    ):
        from rtp_llm.multimodal.mm_embedding_cache import _embedding_result_cost

        # The cache stores embedding vectors; a [hidden] tensor represents one
        # token rather than hidden scalar tokens.
        self.assertEqual(
            _embedding_result_cost((torch.zeros(4), None))[0],
            1,
        )

        cache = MMEmbeddingAsyncCache(
            gpu_max_bytes=0, cpu_max_bytes=32, report_metrics=True
        )
        _, first = cache.try_acquire("first")
        first.complete((torch.zeros((2, 4)), None))

        # Disabling the cache drops the resident gauge immediately. A pending
        # entry is still allowed to finish for existing waiters.
        cache.try_acquire("second")
        cache.resize(0, 0)
        token_values = [
            call.args[1]
            for call in report.call_args_list
            if call.args
            and call.args[0] == GaugeMetrics.VIT_EMBEDDING_CACHE_TOKENS_METRIC
        ]
        self.assertEqual(token_values, [2, 0])


class VitErrorReportingTest(TestCase):
    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_proxy_worker_reports_each_error_once(self, report):
        engine = object.__new__(MMProcessEngine)
        engine.is_proxy_mode = True
        error = RuntimeError("worker preprocessing failed")

        engine.report_vit_error(error)
        engine.report_vit_error(error)

        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)


class AsyncSubmitGetEmbeddingTest(TestCase):

    def test_admission_metrics_use_mapping_tags_and_are_best_effort(self):
        engine = self._make_engine()
        media = self._make_input("fake://metrics")
        try:
            with patch.object(
                engine._async_compute_executor,
                "submit",
                side_effect=lambda *args: concurrent.futures.Future(),
            ):
                with patch(
                    "rtp_llm.multimodal.mm_process_engine.kmonitor.report"
                ) as report:
                    engine.async_submit([media], 801, pretrigger=True)
                    engine.cancel_queued_request(801)
                    self.assertTrue(report.call_args_list)
                    for call in report.call_args_list:
                        if len(call.args) > 2:
                            self.assertIsInstance(call.args[2], dict)
                with patch(
                    "rtp_llm.multimodal.mm_process_engine.kmonitor.report",
                    side_effect=RuntimeError("telemetry unavailable"),
                ), patch("logging.exception"):
                    self.assertEqual(
                        engine.async_submit([media], 802, pretrigger=True),
                        [media.cache_key()],
                    )
                    engine.cancel_queued_request(802)
                    self.assertEqual(engine._async_admitted, 0)
                    self.assertEqual(engine._async_tasks, {})
        finally:
            engine.stop()

    def test_partial_overlap_failure_recompute_and_disabled_cache(self):
        engine = self._make_engine()
        a, b = self._make_input("fake://a"), self._make_input("fake://b")
        futures = []

        def enqueue(*args):
            future = concurrent.futures.Future()
            futures.append(future)
            return future

        try:
            with patch.object(
                engine._async_compute_executor, "submit", side_effect=enqueue
            ):
                engine.async_submit([a], 601, pretrigger=True)
                claims = engine._claim_and_submit_async([a, b], request_id=602)
                self.assertEqual(len(futures), 2)
                self.assertIs(claims[0][1], engine._embedding_cache.peek(a.cache_key()))
                self.assertEqual(engine.cancel_queued_request(602), 1)  # only B
                self.assertEqual(engine._async_admitted, 1)
                failure = ValueError("shared failure")
                futures[0].set_exception(failure)
                with self.assertRaisesRegex(ValueError, "shared failure"):
                    claims[0][1].wait(1)
                self.assertIsNone(engine._embedding_cache.peek(a.cache_key()))
                self.assertEqual(engine._async_admitted, 0)
                engine.async_submit([a], 603, pretrigger=True)
                self.assertEqual(len(futures), 3)
                self.assertIsNot(
                    claims[0][1], engine._embedding_cache.peek(a.cache_key())
                )
                engine.cancel_queued_request(603)
                engine._embedding_cache.resize(0, 0)
                engine.async_submit([a], 604, pretrigger=True)
                engine.async_submit([a], 605, pretrigger=True)
                self.assertEqual(len(futures), 5)
                self.assertEqual(engine._async_admitted, 2)
                engine.cancel_queued_request(604)
                engine.cancel_queued_request(605)
                self.assertEqual(engine._async_admitted, 0)
                self.assertEqual(engine._async_tasks, {})
                self.assertEqual(engine._async_request_tasks, {})
        finally:
            engine.stop()

    def test_join_cannot_acknowledge_before_executor_handoff(self):
        engine = self._make_engine()
        media = self._make_input("fake://handoff")
        claimed, release, attempting, entered_early = (
            threading.Event() for _ in range(4)
        )
        original_lock = engine._async_task_lock
        original_submit = engine._submit_async_compute_batch

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "joining-pretrigger":
                    acquired = original_lock.acquire(blocking=False)
                    if acquired:
                        entered_early.set()
                    attempting.set()
                    if acquired:
                        return
                original_lock.acquire()

            def __exit__(self, *args):
                original_lock.release()

        def gated_submit(*args, **kwargs):
            if kwargs.get("request_id") == 301:
                claimed.set()
                self.assertTrue(release.wait(5))
            return original_submit(*args, **kwargs)

        engine._async_task_lock = ObservedLock()
        engine._submit_async_compute_batch = gated_submit
        acknowledged = threading.Event()
        failures = []

        def submit(request_id):
            try:
                engine.async_submit([media], request_id=request_id, pretrigger=True)
                if request_id == 302:
                    acknowledged.set()
            except Exception as error:
                failures.append(error)

        owner = threading.Thread(target=submit, args=(301,))
        joiner = threading.Thread(target=submit, args=(302,), name="joining-pretrigger")
        try:
            owner.start()
            self.assertTrue(claimed.wait(5))
            joiner.start()
            self.assertTrue(attempting.wait(5))
            self.assertFalse(entered_early.is_set())
            self.assertFalse(acknowledged.is_set())
            release.set()
            owner.join(5)
            joiner.join(5)
            self.assertFalse(owner.is_alive())
            self.assertFalse(joiner.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(acknowledged.is_set())
        finally:
            release.set()
            owner.join(5)
            if joiner.ident is not None:
                joiner.join(5)
            engine.stop()

    def test_cancellation_during_claim_never_orphans_joining_pretrigger(self):
        engine = self._make_engine()
        media = self._make_input("fake://cancel-handoff")
        claimed, release, cancelled = (threading.Event() for _ in range(3))
        submit = engine._submit_async_compute_batch
        futures = []

        def enqueue(*args):
            future = concurrent.futures.Future()
            futures.append(future)
            return future

        def gated_submit(*args, **kwargs):
            if kwargs.get("request_id") == 401:
                claimed.set()
                self.assertTrue(release.wait(5))
            return submit(*args, **kwargs)

        engine._submit_async_compute_batch = gated_submit
        with patch.object(
            engine._async_compute_executor, "submit", side_effect=enqueue
        ):
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                owner = pool.submit(
                    engine._claim_and_submit_async, [media], 401, None, cancelled
                )
                try:
                    self.assertTrue(claimed.wait(5))
                    joining = pool.submit(
                        engine.async_submit, [media], 402, pretrigger=True
                    )
                    cancelled.set()
                    release.set()
                    with self.assertRaises(FtRuntimeException):
                        owner.result(5)
                    self.assertEqual(joining.result(5), [media.cache_key()])
                    entry = engine._embedding_cache.peek(media.cache_key())
                    self.assertIsNotNone(engine._async_tasks[entry].future)
                    self.assertIn(402, engine._async_tasks[entry].request_ids)
                    self.assertEqual(engine._async_admitted, 1)
                    engine.cancel_queued_request(402)
                    self.assertEqual(engine._async_admitted, 0)
                    self.assertEqual(engine._async_tasks, {})
                finally:
                    release.set()
                    engine.stop()

    def test_only_pretrigger_initiated_computation_is_credited(self):
        engine = self._make_engine()
        first = self._make_input("fake://pretrigger-first")
        second = self._make_input("fake://inference-first")
        try:
            engine.async_submit([first], 501, pretrigger=True)
            engine.get_embedding_result([first], request_id=502)
            entry = engine._embedding_cache.peek(first.cache_key())
            self.assertTrue(entry.pretrigger_initiated)
            self.assertFalse(entry.claim_pretrigger_reuse())  # already counted
            engine.get_embedding_result([second], request_id=503)
            engine.async_submit([second], 504, pretrigger=True)
            entry = engine._embedding_cache.peek(second.cache_key())
            self.assertFalse(entry.pretrigger_initiated)
            self.assertFalse(entry.claim_pretrigger_reuse())
        finally:
            engine.stop()

    def test_hashes_only_waits_on_shared_submit_and_does_not_read_cached_embeddings(
        self,
    ):
        engine = self._make_engine()
        inp = self._make_input("fake://hash-only")
        started, finish = threading.Event(), threading.Event()
        computations = []

        def compute(mm_inputs, cache_key, entry, request_id=0):
            computations.append(cache_key)
            started.set()
            finish.wait(timeout=5)
            engine._hash_key_cache.put(
                cache_key, [torch.tensor([-1, 2], dtype=torch.int32)], entry.generation
            )
            entry.complete((torch.ones(2, 4), None))

        engine._async_compute = compute
        try:
            engine.async_submit([inp], request_id=100)
            self.assertTrue(started.wait(timeout=2))
            entry = engine._embedding_cache.peek(inp.cache_key())
            with patch.object(
                entry,
                "wait",
                side_effect=AssertionError("must not read/promote embedding"),
            ):
                finish.set()
                results = engine.get_embedding_result(
                    [inp], request_id=101, hashes_only=True
                )
                self.assertEqual(results[0].embeddings, [])
                self.assertEqual(results[0].feature_hashes[0].tolist(), [-1, 2])
            self.assertEqual(computations, [inp.cache_key()])
        finally:
            finish.set()
            engine.stop()

    def test_hashes_only_returns_computed_hashes_when_both_caches_are_disabled(self):
        engine = self._make_engine()
        engine._embedding_cache.resize(0, 0)
        engine._hash_key_cache.resize(0)
        inp = self._make_input("fake://no-cache")

        def compute(mm_inputs, cache_key, entry, request_id=0):
            entry.complete((torch.ones(2, 4), None))

        engine._async_compute = compute
        try:
            with patch(
                "rtp_llm.multimodal.mm_process_engine._feature_hashes_from_result",
                return_value=[torch.tensor([-3, 4], dtype=torch.int32)],
            ):
                result = engine.get_embedding_result([inp], hashes_only=True)[0]
            self.assertEqual(result.embeddings, [])
            self.assertEqual(result.feature_hashes[0].tolist(), [-3, 4])
            self.assertIsNone(engine._embedding_cache.peek(inp.cache_key()))
            self.assertEqual(engine._hash_key_cache.keys(), [])
        finally:
            engine.stop()

    def _make_engine(self, mm_part=None, vit_concurrency=64, vit_max_queue_size=64):
        model = FakeModel(mm_part or FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.vit_concurrency = vit_concurrency
        vit_config.vit_max_queue_size = vit_max_queue_size
        return MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

    def _make_input(self, url):
        return MultimodalInput(
            url,
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def test_async_submit_returns_keys(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        keys = engine.async_submit([inp])
        self.assertEqual(len(keys), 1)
        self.assertIsInstance(keys[0], str)
        self.assertTrue(len(keys[0]) > 0)
        engine.stop()

    def test_submit_then_get(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        engine.async_submit([inp])
        results = engine.get_embedding_result([inp])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].embeddings, [torch.tensor(0)])
        engine.stop()

    def test_get_without_submit_computes_synchronously(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        results = engine.get_embedding_result([inp])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].embeddings, [torch.tensor(0)])
        engine.stop()

    def test_cache_hit_is_fast(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        engine.get_embedding_result([inp])

        t0 = time.time()
        results = engine.get_embedding_result([inp])
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.05)
        self.assertEqual(results[0].embeddings, [torch.tensor(0)])
        engine.stop()

    def test_duplicate_submit_no_recompute(self):
        engine = self._make_engine()
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        keys1 = engine.async_submit([inp])
        keys2 = engine.async_submit([inp])
        self.assertEqual(keys1, keys2)
        engine.stop()

    def test_async_compute_concurrency_is_bounded(self):
        engine = self._make_engine(vit_concurrency=2)
        release = threading.Event()
        saturated = threading.Event()
        completed = threading.Event()
        lock = threading.Lock()
        active = 0
        max_active = 0
        completed_count = 0

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            nonlocal active, max_active, completed_count
            with lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    saturated.set()
            try:
                release.wait(timeout=5)
                entry.complete((torch.tensor(0), None))
            finally:
                with lock:
                    active -= 1
                    completed_count += 1
                    if completed_count == 8:
                        completed.set()

        engine._async_compute = blocked_compute
        try:
            for index in range(8):
                engine.async_submit([self._make_input(f"fake://bounded-{index}")])

            self.assertTrue(saturated.wait(timeout=2))
            time.sleep(0.1)
            self.assertEqual(max_active, 2)

            release.set()
            self.assertTrue(completed.wait(timeout=5))
            self.assertEqual(max_active, 2)
        finally:
            release.set()
            engine.stop()

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_async_compute_queue_rejects_over_capacity(self, report):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        release = threading.Event()
        started = threading.Event()
        completed = threading.Event()
        lock = threading.Lock()
        completed_count = 0

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            nonlocal completed_count
            started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))
            with lock:
                completed_count += 1
                if completed_count == 2:
                    completed.set()

        engine._async_compute = blocked_compute
        first = self._make_input("fake://queue-running")
        queued = self._make_input("fake://queue-waiting")
        rejected = self._make_input("fake://queue-rejected")
        try:
            engine.async_submit([first])
            self.assertTrue(started.wait(timeout=2))
            engine.async_submit([queued])

            with self.assertRaises(FtRuntimeException) as raised:
                engine.async_submit([rejected])
            self.assertEqual(
                raised.exception.exception_type,
                ExceptionType.CONCURRENCY_LIMIT_ERROR,
            )
            self.assertIsNone(engine._embedding_cache.peek(rejected.cache_key()))
            error_reports = [
                call
                for call in report.call_args_list
                if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
            ]
            self.assertEqual(len(error_reports), 1)

            release.set()
            self.assertTrue(completed.wait(timeout=5))
        finally:
            release.set()
            engine.stop()

    def test_cancel_request_removes_only_queued_work(self):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        release = threading.Event()
        first_started = threading.Event()
        first_done = threading.Event()
        started_urls = []

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            started_urls.append(mm_inputs[0].url)
            first_started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))
            first_done.set()

        engine._async_compute = blocked_compute
        running = self._make_input("fake://cancel-running")
        queued = self._make_input("fake://cancel-queued")
        try:
            engine.async_submit([running], request_id=101)
            self.assertTrue(first_started.wait(timeout=2))
            engine.async_submit([queued], request_id=102)

            self.assertEqual(engine.cancel_queued_request(102), 1)
            self.assertIsNone(engine._embedding_cache.peek(queued.cache_key()))
            self.assertEqual(engine._async_admitted, 1)
            self.assertEqual(engine.cancel_queued_request(101), 0)

            release.set()
            self.assertTrue(first_done.wait(timeout=5))
            self.assertEqual(started_urls, [running.url])
            running_entry = engine._embedding_cache.peek(running.cache_key())
            self.assertIsNotNone(running_entry)
            self.assertTrue(running_entry.is_done)
        finally:
            release.set()
            engine.stop()

    def test_cancel_keeps_queued_work_owned_by_another_request(self):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        release = threading.Event()
        first_started = threading.Event()
        first_done = threading.Event()
        started_urls = []

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            started_urls.append(mm_inputs[0].url)
            first_started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))
            first_done.set()

        engine._async_compute = blocked_compute
        running = self._make_input("fake://shared-running")
        shared = self._make_input("fake://shared-queued")
        try:
            engine.async_submit([running], request_id=201)
            self.assertTrue(first_started.wait(timeout=2))
            engine.async_submit([shared], request_id=202)
            engine.async_submit([shared], request_id=203)

            self.assertEqual(engine.cancel_queued_request(202), 0)
            self.assertIsNotNone(engine._embedding_cache.peek(shared.cache_key()))
            self.assertEqual(engine.cancel_queued_request(203), 1)
            self.assertIsNone(engine._embedding_cache.peek(shared.cache_key()))

            release.set()
            self.assertTrue(first_done.wait(timeout=5))
            self.assertEqual(started_urls, [running.url])
        finally:
            release.set()
            engine.stop()

    def test_already_cancelled_request_is_not_submitted(self):
        engine = self._make_engine(vit_concurrency=1, vit_max_queue_size=1)
        cancellation_event = threading.Event()
        cancellation_event.set()
        inp = self._make_input("fake://cancel-before-submit")
        try:
            with self.assertRaises(FtRuntimeException) as raised:
                engine.get_embedding_result(
                    [inp],
                    request_id=301,
                    cancellation_event=cancellation_event,
                )
            self.assertEqual(
                raised.exception.exception_type, ExceptionType.CANCELLED_ERROR
            )
            self.assertIsNone(engine._embedding_cache.peek(inp.cache_key()))
            self.assertEqual(engine._async_admitted, 0)
        finally:
            engine.stop()

    def test_get_submits_all_inputs_before_waiting(self):
        engine = self._make_engine(vit_concurrency=2, vit_max_queue_size=0)
        release = threading.Event()
        both_started = threading.Event()
        lock = threading.Lock()
        started_count = 0
        result = None
        error = None

        def blocked_compute(mm_inputs, cache_key, entry, request_id=0):
            nonlocal started_count
            with lock:
                started_count += 1
                if started_count == 2:
                    both_started.set()
            release.wait(timeout=5)
            entry.complete((torch.tensor(0), None))

        def get_results():
            nonlocal result, error
            try:
                result = engine.get_embedding_result(
                    [
                        self._make_input("fake://parallel-get-0"),
                        self._make_input("fake://parallel-get-1"),
                    ]
                )
            except Exception as caught:
                error = caught

        engine._async_compute = blocked_compute
        thread = threading.Thread(target=get_results)
        try:
            thread.start()
            self.assertTrue(both_started.wait(timeout=2))
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertIsNone(error)
            self.assertEqual(len(result), 2)
        finally:
            release.set()
            thread.join(timeout=5)
            engine.stop()

    def test_multiple_inputs_independent(self):
        engine = self._make_engine()
        inp1 = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        inp2 = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        # Same URL → same cache key
        keys = engine.async_submit([inp1, inp2])
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])

        results = engine.get_embedding_result([inp1, inp2])
        self.assertEqual(len(results), 2)
        engine.stop()

    def test_concurrent_get_same_key(self):
        engine = self._make_engine(FakeSlowEmbeddingInterface())
        inp = self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")
        results = [None, None]
        errors = [None, None]

        def worker(idx):
            try:
                results[idx] = engine.get_embedding_result([inp])
            except Exception as e:
                errors[idx] = e

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.time() - t0

        for e in errors:
            self.assertIsNone(e)
        for r in results:
            self.assertIsNotNone(r)
            self.assertEqual(len(r), 1)
        # Both should finish in roughly one embedding time, not two
        self.assertLess(elapsed, FakeSlowEmbeddingInterface.delay * 2)
        engine.stop()

    def test_async_and_sync_same_key_share_one_embedding(self):
        class CountingSlowEmbedding(FakeMultiModalEmbeddingInterface):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.lock = threading.Lock()

            @torch.inference_mode()
            def embedding(self, data, **kwargs):
                with self.lock:
                    self.calls += 1
                time.sleep(0.2)
                return torch.tensor([[1.0]]), None

        mm_part = CountingSlowEmbedding()
        engine = self._make_engine(mm_part)
        inp = self._make_input("fake://sync-async-dedup")

        engine.async_submit([inp])
        sync_result = engine.mm_embedding_cpp(
            [inp.url],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        async_result = engine.get_embedding_result([inp])

        self.assertEqual(mm_part.calls, 1)
        self.assertEqual(sync_result.embeddings[0].item(), 1.0)
        self.assertEqual(async_result[0].embeddings[0].item(), 1.0)
        self.assertGreaterEqual(engine._embedding_cache.stats()["inflight_dedup"], 1)
        engine.stop()

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_error_clears_cache(self, report):
        engine = self._make_engine(
            FakeMultiModalEmbeddingInterfacePreprocessException()
        )
        # Use a unique URL so the global vit_emb_cache_ won't have a hit
        # from earlier tests (which would skip preprocessing entirely).
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?error_test"
        )

        with self.assertRaises(PreprcoesException):
            engine.get_embedding_result([inp])

        # After error, cache entry should be removed — next call should re-attempt
        state, _ = engine._async_cache.try_acquire(inp.cache_key())
        self.assertEqual(state, "miss")
        error_reports = [
            call
            for call in report.call_args_list
            if call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
        ]
        self.assertEqual(len(error_reports), 1)
        engine.stop()

    def test_empty_url_raises(self):
        engine = self._make_engine()
        inp = self._make_input("")
        with self.assertRaises(ValueError):
            engine.async_submit([inp])
        with self.assertRaises(ValueError):
            engine.get_embedding_result([inp])
        engine.stop()


# ----------------------------------------------------------------------------
# GreenNet (content safety) integration
# ----------------------------------------------------------------------------


class _StubGreenNetHandle(GreenNetHandle):
    def __init__(self, rewritten_inputs, verdict, delay=0.0):
        self.rewritten_inputs = rewritten_inputs
        self._verdict = verdict
        self._delay = delay
        self.cancelled = False

    async def wait_result(self) -> GreenNetVerdict:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._verdict

    def cancel(self) -> None:
        self.cancelled = True


class _StubGreenNetProvider(GreenNetProvider):
    """Records inputs and returns a programmable verdict. Optionally rewrites
    each input's url so we can assert the rewritten inputs reach ViT."""

    def __init__(self, verdict, rewrite_suffix=None, delay=0.0):
        self._verdict = verdict
        self._rewrite_suffix = rewrite_suffix
        self._delay = delay
        self.calls = 0
        self.last_handle = None
        self.request_ids = []

    def is_enabled(self) -> bool:
        return True

    async def preprocess_and_submit(self, request, mm_inputs):
        self.calls += 1
        self.request_ids.append(str(request.id))
        if self._rewrite_suffix is not None:
            rewritten = [
                MultimodalInput(
                    mi.url + self._rewrite_suffix,
                    mi.mm_type,
                    torch.empty(0),
                    mi.mm_preprocess_config,
                )
                for mi in mm_inputs
            ]
        else:
            rewritten = list(mm_inputs)
        handle = _StubGreenNetHandle(rewritten, self._verdict, self._delay)
        self.last_handle = handle
        return handle


class _UrlRecordingEmbedding(FakeMultiModalEmbeddingInterface):
    """Records the urls preprocess_input actually received (to verify the
    greennet-rewritten inputs are what ViT consumes)."""

    seen_urls: List[str] = []

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        _UrlRecordingEmbedding.seen_urls.extend(mi.url for mi in mm_inputs)
        return mm_inputs, kwargs


class MMProcessEngineGreenNetTest(TestCase):
    def _make_engine(self, mm_part=None):
        model = FakeModel(mm_part or FakeMultiModalEmbeddingInterface())
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        return MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )

    def _make_input(self, url):
        return MultimodalInput(
            url,
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(-1, -1, -1, -1, -1, -1, -1, [], 30000),
        )

    def test_default_provider_is_noop(self):
        # No internal_source in the open-source test env → no-op provider,
        # so greennet is disabled and the engine behaves exactly as before.
        engine = self._make_engine()
        self.assertFalse(engine._greennet_enabled())
        verdict = engine.wait_greennet_verdict(
            [self._make_input("./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg")]
        )
        self.assertTrue(verdict.passed)
        engine.stop()

    def test_request_id_reaches_greennet_and_vit_access_logs(self):
        engine = self._make_engine()
        provider = _StubGreenNetProvider(GreenNetVerdict(passed=True, code=1))
        engine._greennet_provider = provider
        engine._access_logger = MagicMock(spec=MMAccessLogger)
        request_id = 987654321

        engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?request_id"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
            request_id,
        )

        self.assertEqual(provider.request_ids, [str(request_id)])
        self.assertEqual(
            engine._access_logger.log_query_access.call_args.args[1], request_id
        )
        self.assertEqual(
            engine._access_logger.log_success_access.call_args.args[2], request_id
        )
        engine.stop()

    def test_local_path_passes_when_verdict_passes(self):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1)
        )
        res = engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_pass"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertEqual(res.embeddings, [torch.tensor(0)])
        engine.stop()

    def test_local_path_raises_when_verdict_fails(self):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="blocked")
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            engine.mm_embedding_cpp(
                ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_fail"],
                [MMUrlType.IMAGE],
                [torch.empty(0)],
                [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
            )
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSAFE_INPUT_CONTENT
        )
        self.assertIn("blocked", ctx.exception.message)
        engine.stop()

    def test_rewritten_inputs_reach_vit(self):
        _UrlRecordingEmbedding.seen_urls = []
        engine = self._make_engine(_UrlRecordingEmbedding())
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1), rewrite_suffix="#rewritten"
        )
        engine.mm_embedding_cpp(
            ["./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_rw"],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        self.assertTrue(
            any(u.endswith("#rewritten") for u in _UrlRecordingEmbedding.seen_urls),
            f"ViT did not see rewritten url: {_UrlRecordingEmbedding.seen_urls}",
        )
        engine.stop()

    def test_rewritten_sync_and_async_share_original_key(self):
        class SlowRecordingEmbedding(_UrlRecordingEmbedding):
            calls = 0
            lock = threading.Lock()

            @torch.inference_mode()
            def embedding(self, data, **kwargs):
                with self.lock:
                    self.calls += 1
                time.sleep(0.2)
                return torch.tensor([[1.0]]), None

        _UrlRecordingEmbedding.seen_urls = []
        part = SlowRecordingEmbedding()
        engine = self._make_engine(part)
        provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1), rewrite_suffix="#rewritten"
        )
        engine._greennet_provider = provider
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_dedup"
        )

        engine.async_submit([inp])
        sync_result = engine.mm_embedding_cpp(
            [inp.url],
            [MMUrlType.IMAGE],
            [torch.empty(0)],
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000]],
        )
        async_result = engine.get_embedding_result([inp])

        self.assertEqual(part.calls, 1)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(sync_result.embeddings[0].item(), 1.0)
        self.assertEqual(async_result[0].embeddings[0].item(), 1.0)
        self.assertTrue(
            any(u.endswith("#rewritten") for u in _UrlRecordingEmbedding.seen_urls)
        )
        engine.stop()

    def test_wait_verdict_pass_after_async_submit(self):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=True, code=1)
        )
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_wait_pass"
        )
        engine.async_submit([inp])
        verdict = engine.wait_greennet_verdict([inp])
        self.assertTrue(verdict.passed)
        engine.stop()

    @patch("rtp_llm.multimodal.mm_process_engine.kmonitor.report")
    def test_wait_verdict_fail_after_async_submit(self, report):
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="nsfw")
        )
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_wait_fail"
        )
        engine.async_submit([inp])
        verdict = engine.wait_greennet_verdict([inp])
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.code, 2)
        # The embedding entry must also surface the violation.
        with self.assertRaises(FtRuntimeException) as ctx:
            engine.get_embedding_result([inp])
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSAFE_INPUT_CONTENT
        )
        self.assertTrue(
            any(
                call.args and call.args[0] == AccMetrics.VIT_ERROR_QPS_METRIC
                for call in report.call_args_list
            )
        )
        engine.stop()

    def test_wait_verdict_kicks_compute_on_miss(self):
        # wait_greennet_verdict called without a prior async_submit must still
        # produce a verdict (kick compute itself).
        engine = self._make_engine()
        engine._greennet_provider = _StubGreenNetProvider(
            GreenNetVerdict(passed=False, code=2, message="bad")
        )
        inp = self._make_input(
            "./rtp_llm/multimodal/test/testdata/qwen2_vl/1.jpg?gn_miss"
        )
        verdict = engine.wait_greennet_verdict([inp])
        self.assertFalse(verdict.passed)
        engine.stop()


class MMAccessLoggerRequestIdTest(TestCase):
    def test_request_id_is_serialized_at_top_level(self):
        access_logger = MMAccessLogger.__new__(MMAccessLogger)
        access_logger.query_logger = MagicMock()
        mm_input = MagicMock()
        mm_input.to_string.return_value = "image://test"

        access_logger.log_query_access([mm_input], request_id=123456)

        payload = json.loads(access_logger.query_logger.info.call_args.args[0])
        self.assertEqual(payload["id"], 123456)
        self.assertEqual(payload["query"], ["image://test"])


_DEFAULT_CONFIG = [-1, -1, -1, -1, -1, -1, -1, [], 30000]


class FakeBatchMMPart(MultiModalEmbeddingInterface):
    """mm_part returning identity-encoded (emb, pos, extra) tuples.

    Each input carries an index in its url ("fake://<i>"); embedding echoes that
    index into all three output tensors so tests can assert ordering, and counts
    embedding/batched_embedding invocations to observe cache hits and batching.
    """

    def __init__(self):
        self.embedding_calls = 0
        self.batch_sizes: List[int] = []
        self._lock = threading.Lock()

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        # Carry the inputs through; embedding derives identity from the url.
        return mm_inputs, kwargs

    def get_preprocess_params(self):
        return {}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        mm_inputs, _ = data
        idx = float(int(mm_inputs[0].url.split("://")[1]))
        with self._lock:
            self.embedding_calls += 1
        emb = torch.tensor([[idx]])  # (1, 1) -> one embedding per work item
        pos = torch.tensor([[idx]])  # (1, 1)
        extra = torch.tensor([idx])  # (1,) -> one flat extra tensor
        return emb, pos, extra

    def batched_embedding(self, data_list, mm_types, **kwargs):
        with self._lock:
            self.batch_sizes.append(len(data_list))
        return super().batched_embedding(data_list, mm_types, **kwargs)


class MMProcessEngineGpuBatchTest(TestCase):
    def setUp(self):
        # vit_emb_cache_ is a process-global; isolate it so cache state never
        # leaks between these tests (or into other test classes in the process).
        vit_emb_cache_.resize_cache(0)

    def tearDown(self):
        vit_emb_cache_.resize_cache(0)

    def _make_engine(self, **vit_overrides):
        model = FakeModel(FakeBatchMMPart())
        vit_config = VitConfig()
        vit_config.use_gpu_batch = True
        # Local preprocess keeps the test in-process and deterministic.
        vit_config.use_local_preprocess = True
        # Cache off by default; the cache test opts in explicitly.
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        for key, value in vit_overrides.items():
            setattr(vit_config, key, value)
        engine = MMProcessEngine(
            model.mm_part,
            model.model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
        )
        self.addCleanup(engine.stop)
        return engine, model.mm_part

    def _embed(self, engine, urls):
        n = len(urls)
        return engine.mm_embedding_cpp(
            urls,
            [MMUrlType.IMAGE] * n,
            [torch.empty(0)] * n,
            [list(_DEFAULT_CONFIG) for _ in range(n)],
        )

    def test_gpu_batch_order_and_outputs(self):
        """Single multi-image request: emb/pos/extra preserve input order."""
        engine, _ = self._make_engine()
        urls = [f"fake://{i}" for i in range(4)]
        res = self._embed(engine, urls)

        self.assertEqual([e.item() for e in res.embeddings], [0, 1, 2, 3])
        self.assertEqual([p.item() for p in res.position_ids], [0, 1, 2, 3])
        self.assertEqual([x.item() for x in res.extra_input], [0, 1, 2, 3])

    def test_gpu_batch_multi_request(self):
        """Concurrent requests are batched yet each gets its own correct result."""
        engine, part = self._make_engine(gpu_batch_wait_ms=400, gpu_max_batch_size=16)
        n = 5
        results: List[float] = [None] * n

        def run(i: int):
            res = self._embed(engine, [f"fake://{i}"])
            results[i] = res.embeddings[0].item()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results, [0.0, 1.0, 2.0, 3.0, 4.0])
        # The wait window should let at least one forward serve >1 request.
        self.assertGreaterEqual(max(part.batch_sizes), 2)

    def test_embedding_byte_budget_is_independent_of_legacy_item_count(self):
        engine, part = self._make_engine(
            mm_cache_item_num=0,
            mm_cache_cpu_max_bytes=12,
            mm_hash_key_cache_max_bytes=4096,
        )
        # FakeBatchMMPart returns embedding, position and extra: 3 float32s.
        first = self._embed(engine, ["fake://1"])
        cached = self._embed(engine, ["fake://1"])
        self.assertEqual(part.embedding_calls, 1)
        self.assertTrue(torch.equal(first.feature_hashes[0], cached.feature_hashes[0]))
        self._embed(engine, ["fake://2"])
        self.assertEqual(engine._embedding_cache.stats()["resident_bytes"], 12)
        recomputed = self._embed(engine, ["fake://1"])
        self.assertEqual(part.embedding_calls, 3)
        self.assertEqual(recomputed.embeddings[0].item(), 1)
        self.assertTrue(
            torch.equal(first.feature_hashes[0], recomputed.feature_hashes[0])
        )

    def test_sync_and_async_cpu_hits_restore_gpu_outputs_without_recompute(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA for engine cache round trip")
        engine, part = self._make_engine(
            mm_cache_gpu_max_bytes=8,
            mm_cache_cpu_max_bytes=64,
            mm_hash_key_cache_max_bytes=4096,
        )
        embedding = part.embedding

        def gpu_embedding(data, **kwargs):
            emb, pos, extra = embedding(data, **kwargs)
            return emb.cuda(), pos, extra.cuda()

        part.embedding = gpu_embedding
        first = self._embed(engine, ["fake://1"])
        # Use the same preprocess configuration as the sync C++ entry point.
        inp = MultimodalInput(
            "fake://1",
            MMUrlType.IMAGE,
            torch.empty(0),
            MMPreprocessConfig(*_DEFAULT_CONFIG),
        )
        key = inp.cache_key()
        cached_entry = engine._embedding_cache.peek(key)
        generation = cached_entry.generation
        self._embed(engine, ["fake://2"])
        self.assertEqual(cached_entry.tier, "cpu")
        self.assertTrue(
            engine._hash_key_cache.metadata([key], engine._embedding_cache)["entries"][
                0
            ]["hit"]
        )
        sync_hit = self._embed(engine, ["fake://1"])
        self.assertEqual(part.embedding_calls, 2)
        self.assertEqual(cached_entry.tier, "gpu")
        self._embed(engine, ["fake://2"])
        self.assertEqual(cached_entry.tier, "cpu")
        async_hit = engine.get_embedding_result([inp])[0]
        self.assertEqual(part.embedding_calls, 2)
        self.assertEqual(cached_entry.generation, generation)
        for actual in (sync_hit, async_hit):
            self.assertEqual(actual.embeddings[0].device.type, "cuda")
            self.assertEqual(actual.position_ids[0].device.type, "cpu")
            self.assertEqual(actual.extra_input[0].device.type, "cuda")
            self.assertTrue(torch.equal(actual.embeddings[0], first.embeddings[0]))
            self.assertTrue(torch.equal(actual.position_ids[0], first.position_ids[0]))
            self.assertTrue(torch.equal(actual.extra_input[0], first.extra_input[0]))
            self.assertTrue(
                torch.equal(actual.feature_hashes[0], first.feature_hashes[0])
            )

    def test_hash_cache_can_be_disabled_independently(self):
        engine, part = self._make_engine(
            mm_cache_cpu_max_bytes=4096, mm_hash_key_cache_max_bytes=0
        )
        first = self._embed(engine, ["fake://1"])
        cached = self._embed(engine, ["fake://1"])
        self.assertEqual(part.embedding_calls, 1)
        self.assertEqual(engine._hash_key_cache.keys(), [])
        self.assertTrue(torch.equal(first.feature_hashes[0], cached.feature_hashes[0]))

    def test_gpu_batch_cache_hit(self):
        """A repeated url is served from cache without a second embedding call."""
        engine, part = self._make_engine(mm_cache_cpu_max_bytes=4096)
        # tearDown restores the global cache to disabled for other tests.

        url = "fake://7"
        r1 = self._embed(engine, [url])
        r2 = self._embed(engine, [url])

        self.assertEqual(r1.embeddings[0].item(), 7)
        self.assertEqual(r2.embeddings[0].item(), 7)
        self.assertEqual(part.embedding_calls, 1)
        self.assertTrue(torch.equal(r1.feature_hashes[0], r2.feature_hashes[0]))
        from rtp_llm.ops import get_multimodal_feature_hash

        self.assertTrue(
            torch.equal(
                r1.feature_hashes[0], get_multimodal_feature_hash(r1.embeddings[0])
            )
        )
        metadata = engine._hash_key_cache.metadata(
            engine._hash_key_cache.metadata_keys(), engine._embedding_cache
        )
        self.assertEqual(len(metadata["entries"]), 1)
        self.assertEqual(
            metadata["entries"][0]["feature_hashes"], r1.feature_hashes[0].tolist()
        )
        self.assertEqual(len(engine._hash_key_cache.keys()), 1)

    def test_cached_hashes_survive_async_and_rpc_serialization(self):
        from rtp_llm.multimodal.multimodal_util import (
            build_multimodal_output_pb as trans_output,
        )
        from rtp_llm.server.vit_rpc_server import merge_embedding_results
        from rtp_llm.utils.grpc_util import trans_tensor

        engine, part = self._make_engine(mm_cache_cpu_max_bytes=4096)
        urls = ["fake://7", "fake://9"]
        self._embed(engine, urls)
        inputs = [
            MultimodalInput(
                url,
                MMUrlType.IMAGE,
                torch.empty(0),
                MMPreprocessConfig(*_DEFAULT_CONFIG),
            )
            for url in urls
        ]
        results = engine.get_embedding_result(inputs)
        self.assertEqual(part.embedding_calls, 2)
        merged = merge_embedding_results(results)
        response = trans_output(
            merged.embeddings,
            merged.position_ids,
            merged.extra_input,
            merged.feature_hashes,
        )
        self.assertEqual(response.feature_hash_version, 1)
        self.assertEqual(list(response.split_size), [1, 1])
        expected = torch.cat([r.feature_hashes[0] for r in results])
        self.assertTrue(
            torch.equal(trans_tensor(response.multimodal_feature_hash), expected)
        )


if __name__ == "__main__":
    main()
