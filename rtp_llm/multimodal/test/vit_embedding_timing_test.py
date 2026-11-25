import json
import sys
import threading
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

from rtp_llm.multimodal import vit_embedding_timing as timing


class VitEmbeddingTimingTest(TestCase):
    def setUp(self):
        self.clock = 0
        clock = patch.object(
            timing.time, "monotonic_ns", side_effect=lambda: self.clock
        )
        clock.start()
        self.addCleanup(clock.stop)
        sync = patch.object(timing, "_STAGE_TIMING_SYNC", False)
        sync.start()
        self.addCleanup(sync.stop)

    def advance(self, ms):
        self.clock += int(ms * 1e6)

    def record(self, logs):
        return json.loads(logs.records[-1].getMessage().split(" ", 1)[1])

    def test_default_does_not_synchronize(self):
        cuda = Mock()
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            with self.assertLogs(level="INFO") as logs:
                with timing.embedding_batch_context("batch-1", request_ids=[42]):
                    with timing.embedding_stage_timing(
                        "vision_forward", device_sync=True
                    ):
                        self.advance(11)
        cuda.is_available.assert_not_called()
        cuda.synchronize.assert_not_called()
        record = self.record(logs)
        self.assertEqual(record["elapsed_ms"], 11)
        self.assertEqual(record["host_ms"], 11)
        self.assertEqual(record["timing_mode"], "host")
        self.assertEqual(record["batch_id"], "batch-1")
        self.assertEqual(record["request_ids"], [42])

    def test_sync_separates_prior_work_and_forward_completion(self):
        costs = iter([7, 5])
        cuda = SimpleNamespace(
            is_available=lambda: True,
            synchronize=Mock(side_effect=lambda: self.advance(next(costs))),
        )
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            with patch.object(timing, "_STAGE_TIMING_SYNC", True):
                with self.assertLogs(level="INFO") as logs:
                    with timing.embedding_stage_timing(
                        "vision_forward", device_sync=True
                    ):
                        self.advance(11)
        record = self.record(logs)
        self.assertEqual(record["pre_sync_ms"], 7)
        self.assertEqual(record["host_ms"], 11)
        self.assertEqual(record["device_sync_ms"], 5)
        self.assertEqual(record["elapsed_ms"], 16)
        self.assertEqual(record["timing_mode"], "device_sync")

    def test_body_failure_is_logged_and_preserved_without_post_sync(self):
        failure = ValueError("private input must not appear in logs")
        cuda = SimpleNamespace(is_available=lambda: True, synchronize=Mock())
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            with patch.object(timing, "_STAGE_TIMING_SYNC", True):
                with self.assertLogs(level="INFO") as logs:
                    with self.assertRaises(ValueError) as caught:
                        with timing.embedding_stage_timing(
                            "vision_forward", device_sync=True
                        ):
                            self.advance(3)
                            raise failure
        self.assertIs(caught.exception, failure)
        cuda.synchronize.assert_called_once()
        record = self.record(logs)
        self.assertEqual(record["outcome"], "error")
        self.assertEqual(record["error"], "ValueError")
        self.assertEqual(record["host_ms"], 3)
        self.assertNotIn("private input", " ".join(logs.output))

    def test_post_sync_failure_preserves_host_time(self):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            synchronize=Mock(side_effect=[None, RuntimeError("sync failed")]),
        )
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            with patch.object(timing, "_STAGE_TIMING_SYNC", True):
                with self.assertLogs(level="INFO") as logs:
                    with self.assertRaisesRegex(RuntimeError, "sync failed"):
                        with timing.embedding_stage_timing(
                            "vision_forward", device_sync=True
                        ):
                            self.advance(4)
        self.assertEqual(self.record(logs)["host_ms"], 4)
        self.assertEqual(self.record(logs)["outcome"], "error")

    def test_cpu_process_does_not_synchronize(self):
        cuda = SimpleNamespace(is_available=lambda: False, synchronize=Mock())
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            with patch.object(timing, "_STAGE_TIMING_SYNC", True):
                with self.assertLogs(level="INFO") as logs:
                    with timing.embedding_stage_timing(
                        "vision_forward", device_sync=True
                    ):
                        self.advance(2)
        cuda.synchronize.assert_not_called()
        self.assertEqual(self.record(logs)["timing_mode"], "host")

    def test_hash_does_not_add_device_sync_in_diagnostic_mode(self):
        cuda = Mock()
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda)):
            with patch.object(timing, "_STAGE_TIMING_SYNC", True):
                with self.assertLogs(level="INFO") as logs:
                    with timing.embedding_stage_timing("feature_hash"):
                        self.advance(8)
        cuda.is_available.assert_not_called()
        self.assertEqual(self.record(logs)["elapsed_ms"], 8)

    def test_context_restores_after_nested_failure(self):
        with timing.embedding_batch_context("outer", request_ids=[1]):
            with self.assertRaises(ValueError):
                with timing.embedding_batch_context("inner", request_ids=[2]):
                    raise ValueError()
            with self.assertLogs(level="INFO") as logs:
                with timing.embedding_stage_timing("cache_complete"):
                    pass
            self.assertEqual(self.record(logs)["batch_id"], "outer")
        with self.assertLogs(level="INFO") as logs:
            with timing.embedding_stage_timing("cache_complete"):
                pass
        self.assertIsNone(self.record(logs)["batch_id"])
        self.assertNotIn("request_ids", self.record(logs))

    def test_batch_context_isolated_between_threads(self):
        barrier = threading.Barrier(2)
        observed = {}

        def worker(batch_id):
            with timing.embedding_batch_context(batch_id):
                barrier.wait(timeout=5)
                with timing.embedding_stage_timing("cache_complete") as record:
                    observed[batch_id] = record["batch_id"]

        threads = [threading.Thread(target=worker, args=(name,)) for name in ["a", "b"]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=6)
            self.assertFalse(thread.is_alive())
        self.assertEqual(observed, {"a": "a", "b": "b"})

    def test_skipped_cache_records_reason(self):
        with self.assertLogs(level="INFO") as logs:
            with timing.embedding_stage_timing(
                "cache_complete", request_id=42
            ) as record:
                record.update(outcome="skipped", reason="deferred")
        record = self.record(logs)
        self.assertEqual(record["outcome"], "skipped")
        self.assertEqual(record["reason"], "deferred")
        self.assertEqual(record["request_id"], 42)


if __name__ == "__main__":
    main()
