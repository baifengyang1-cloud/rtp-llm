import concurrent.futures
import gc
import threading
import unittest
import weakref
from unittest.mock import Mock, patch

import torch
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
    _copy_result_to_devices,
    _embedding_result_cost,
    _tensor_tier_bytes,
)
from rtp_llm.multimodal.mm_scheduler import MMSchedulerExecutionError


class EmbeddingEntryLifetimeTest(unittest.TestCase):
    def test_terminal_callbacks_released_even_if_publication_raises(self):
        for complete in (True, False):
            for raises in (True, False):
                with self.subTest(complete=complete, raises=raises):
                    on_complete, on_fail, on_read = Mock(), Mock(), Mock()
                    entry = MMEmbeddingCacheEntry(on_complete, on_fail, on_read)
                    callback = on_complete if complete else on_fail
                    transition = entry.complete if complete else entry.fail
                    value = "result" if complete else ValueError("bad image")
                    if raises:
                        callback.side_effect = RuntimeError("callback failed")
                        with self.assertRaisesRegex(RuntimeError, "callback failed"):
                            transition(value)
                    else:
                        self.assertTrue(transition(value))
                    self.assertFalse(transition(value))
                    callback.assert_called_once_with(entry, value)
                    self.assertIsNone(entry._on_complete)
                    self.assertIsNone(entry._on_fail)
                    if complete and not raises:
                        self.assertIs(entry.wait(0), on_read.return_value)
                    else:
                        expected_error = RuntimeError if complete else ValueError
                        with self.assertRaises(expected_error):
                            entry.wait_ready(0)

    def test_early_verdict_never_allocates_an_event(self):
        entry = MMEmbeddingCacheEntry()
        verdict = GreenNetVerdict(passed=False)
        with patch.object(threading, "Event", side_effect=AssertionError):
            self.assertIsNone(entry._greennet_event)
            self.assertFalse(entry.is_greennet_decided)
            entry.set_greennet_verdict(verdict)
            self.assertTrue(entry.is_greennet_decided)
            self.assertIs(entry.wait_greennet(0), verdict)

    def test_pending_waiters_share_event_and_release_it_after_verdict(self):
        entry = MMEmbeddingCacheEntry()
        with self.assertRaises(TimeoutError):
            entry.wait_greennet(0)
        event = entry._greennet_event
        wait = event.wait
        registered = threading.Barrier(3, timeout=5)

        def wait_together(timeout):
            registered.wait()
            return wait(timeout)

        verdict = GreenNetVerdict(passed=True)
        with patch.object(event, "wait", side_effect=wait_together):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(entry.wait_greennet, 5) for _ in range(2)]
                registered.wait()
                entry.set_greennet_verdict(verdict)
                for future in futures:
                    self.assertIs(future.result(timeout=5), verdict)
        self.assertIsNone(entry._greennet_event)
        self.assertIs(entry.wait_greennet(0), verdict)


class EmbeddingFailureLifetimeTest(unittest.TestCase):
    def test_producer_and_consumer_frames_release_without_gc(self):
        def run(consumer):
            entry = MMEmbeddingCacheEntry()
            ref = weakref.ref(entry)
            if consumer:
                entry.fail(ValueError("bad image"))
                try:
                    entry.wait_ready()
                except ValueError:
                    pass
            else:
                try:
                    raise ValueError("bad image")
                except ValueError as error:
                    entry.fail(error)
            return ref

        enabled = gc.isenabled()
        gc.disable()
        try:
            for consumer in (False, True):
                with self.subTest(consumer=consumer):
                    refs = [run(consumer) for _ in range(32)]
                    self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if enabled:
                gc.enable()

    def test_each_waiter_gets_independent_runtime_error(self):
        entry = MMEmbeddingCacheEntry()
        error = FtRuntimeException(ExceptionType.MM_DOWNLOAD_FAILED, "download timeout")
        error._vit_error_qps_reported = True
        entry.fail(error)

        def consume():
            try:
                entry.wait_ready(1)
            except FtRuntimeException as caught:
                return caught

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            errors = list(executor.map(lambda _: consume(), range(8)))
        self.assertEqual(len({id(error) for error in errors}), 8)
        for caught in errors:
            self.assertEqual(caught.exception_type, ExceptionType.MM_DOWNLOAD_FAILED)
            self.assertEqual(caught.message, "download timeout")
            self.assertTrue(caught._vit_error_qps_reported)
            self.assertIsNot(caught, error)
        self.assertIsNone(entry.error.__traceback__)

    def test_custom_scheduler_error_fields_survive(self):
        entry = MMEmbeddingCacheEntry()
        entry.fail(MMSchedulerExecutionError("OutOfMemoryError", "GPU full", True))
        with self.assertRaises(MMSchedulerExecutionError) as caught:
            entry.wait_ready(1)
        self.assertEqual(caught.exception.source_type, "OutOfMemoryError")
        self.assertEqual(caught.exception.source_message, "GPU full")
        self.assertTrue(caught.exception.is_oom)

    def test_native_error_messages_and_fields_survive(self):
        for error in (
            TimeoutError("download deadline"),
            FileNotFoundError(2, "missing image", "/missing/image"),
            torch.OutOfMemoryError("GPU full"),
        ):
            with self.subTest(error=type(error).__name__):
                entry = MMEmbeddingCacheEntry()
                entry.fail(error)
                copied = entry.error
                self.assertIs(type(copied), type(error))
                self.assertEqual(str(copied), str(error))
                self.assertEqual(copied.args, error.args)
                if isinstance(error, OSError):
                    self.assertEqual(copied.errno, error.errno)
                    self.assertEqual(copied.filename, error.filename)

    def test_failure_snapshot_does_not_retain_exception_payload(self):
        class Payload:
            pass

        entry = MMEmbeddingCacheEntry()
        payload = Payload()
        ref = weakref.ref(payload)
        error = RuntimeError("failure")
        error.request = payload
        error.__cause__ = ValueError("cause")
        entry.fail(error)
        del payload, error
        self.assertIsNone(ref())
        self.assertIsNone(entry.error.__cause__)
        self.assertFalse(hasattr(entry.error, "request"))

    def test_completion_callback_failure_does_not_retain_entry(self):
        def run():
            def fail_publish(entry, result):
                raise ValueError("cannot publish")

            entry = MMEmbeddingCacheEntry(on_complete=fail_publish)
            ref = weakref.ref(entry)
            try:
                entry.complete(None)
            except ValueError:
                pass
            try:
                entry.wait_ready(1)
            except ValueError:
                pass
            return ref

        enabled = gc.isenabled()
        gc.disable()
        try:
            self.assertIsNone(run()())
        finally:
            if enabled:
                gc.enable()


class EmptyEmbeddingCacheTest(unittest.TestCase):
    def test_empty_results_complete_waiters_without_residency(self):
        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=64)
        for value in (([], []), (torch.empty(0), None), None):
            with self.subTest(value=value):
                _, entry = cache.try_acquire("empty")
                self.assertIs(cache.try_acquire("empty")[1], entry)
                self.assertTrue(entry.complete(value))
                self.assertIs(entry.wait(1), value)
                self.assertIsNone(cache.peek("empty"))
                self.assertEqual(cache.stats()["resident_entries"], 0)
                self.assertEqual(cache.stats()["resident_bytes"], 0)
                self.assertEqual(cache.try_acquire("empty")[0], "miss")
                cache.clear()

    def test_unique_empty_results_do_not_accumulate_or_evict_valid_value(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        hashes = MMHashKeyCache(4096)
        _, valid = cache.try_acquire("valid")
        valid.complete((torch.ones(2, 4), None))
        for i in range(256):
            key = str(i)
            _, entry = cache.try_acquire(key)
            cache.complete(key, entry, ([], []), [], hash_key_cache=hashes)
        self.assertIs(cache.peek("valid"), valid)
        self.assertEqual(cache.stats()["resident_entries"], 1)
        self.assertEqual(cache.stats()["resident_bytes"], 32)
        self.assertEqual(cache.stats()["pending_entries"], 0)
        self.assertEqual(hashes.stats()["resident_entries"], 0)


class EmbeddingTraversalLifetimeTest(unittest.TestCase):
    def test_accounting_scratch_sets_released_without_gc(self):
        refs = []

        def scratch_set():
            value = set()
            refs.append(weakref.ref(value))
            return value

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with patch(
                "rtp_llm.multimodal.mm_embedding_cache.set", scratch_set, create=True
            ):
                for helper in (_tensor_tier_bytes, _embedding_result_cost):
                    with self.subTest(helper=helper.__name__):
                        refs.clear()
                        helper(([torch.ones(2, 4)], None))
                        self.assertTrue(refs)
                        self.assertTrue(
                            all(ref() is None for ref in refs),
                            "accounting scratch state requires cyclic GC",
                        )
        finally:
            if gc_was_enabled:
                gc.enable()

    def test_copied_tensors_released_without_gc(self):
        value = torch.ones(4)
        target = torch.device("meta")
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            copied = _copy_result_to_devices(
                ([value, value], {"none": None}),
                ([target, target], {"none": None}),
            )
            self.assertIs(copied[0][0], copied[0][1])
            self.assertEqual(copied[0][0].device, target)
            self.assertIsNone(copied[1]["none"])
            copied_ref = weakref.ref(copied[0][0])
            del copied
            self.assertIsNone(copied_ref(), "copy scratch state retains its tensors")
        finally:
            if gc_was_enabled:
                gc.enable()

    def test_nested_accounting_preserves_storage_and_token_semantics(self):
        backing = torch.arange(100, dtype=torch.float32)
        view = backing[:8].reshape(2, 4)
        result = (
            [view, view, torch.empty(0, 8), torch.ones(3), torch.tensor(4.0)],
            torch.ones(2, dtype=torch.int64),
            {"alias": view, "none": None},
        )
        self.assertEqual(_tensor_tier_bytes(result), (0, 432))
        self.assertEqual(_embedding_result_cost(result), (6, 432))


class EmbeddingHashPublicationTest(unittest.TestCase):
    def setUp(self):
        self.cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=4096)
        self.hashes = MMHashKeyCache(max_bytes=4096)
        self.value = (torch.ones(2, 4), None)
        self.tokens = [torch.tensor([-3, 4], dtype=torch.int32)]

    def complete(self, entry):
        return self.cache.complete(
            "image", entry, self.value, self.tokens, hash_key_cache=self.hashes
        )

    def test_waiters_cannot_wake_before_hash_publication(self):
        _, entry = self.cache.try_acquire("image")
        inserting, release = threading.Event(), threading.Event()
        original = self.hashes.put

        def delayed_put(*args, **kwargs):
            inserting.set()
            if not release.wait(5):
                raise TimeoutError("hash publication barrier was not released")
            original(*args, **kwargs)

        def consume():
            entry.wait_ready(5)
            return self.hashes.get("image", entry.generation)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            with patch.object(self.hashes, "put", side_effect=delayed_put):
                consumers = [executor.submit(consume) for _ in range(2)]
                producer = executor.submit(self.complete, entry)
                try:
                    self.assertTrue(inserting.wait(5))
                    self.assertFalse(entry._event.is_set())
                    self.assertTrue(all(not future.done() for future in consumers))
                finally:
                    release.set()
                self.assertTrue(producer.result(5))
                for consumer in consumers:
                    self.assertTrue(torch.equal(consumer.result(5)[0], self.tokens[0]))

    def test_old_producer_cannot_replace_new_generation_hashes(self):
        _, old = self.cache.try_acquire("image")
        self.cache.remove("image")
        _, current = self.cache.try_acquire("image")
        self.assertTrue(self.complete(current))
        expected = self.hashes.get("image", current.generation)
        self.assertTrue(self.complete(old))
        self.assertIs(self.hashes.get("image", current.generation), expected)
        self.assertIsNone(self.hashes.get("image", old.generation))

    def test_cancelled_and_duplicate_completions_do_not_publish(self):
        _, cancelled = self.cache.try_acquire("image")
        cancelled.fail(ValueError("cancelled"))
        self.assertFalse(self.complete(cancelled))
        self.assertIsNone(self.hashes.get("image"))
        _, current = self.cache.try_acquire("image")
        self.assertTrue(self.complete(current))
        self.hashes.clear()
        self.assertFalse(self.complete(current))
        self.assertIsNone(self.hashes.get("image"))

    def test_publication_error_reaches_waiter(self):
        _, entry = self.cache.try_acquire("image")
        with patch.object(self.hashes, "put", side_effect=RuntimeError("publish")):
            with self.assertRaisesRegex(RuntimeError, "publish"):
                self.complete(entry)
        with self.assertRaisesRegex(RuntimeError, "publish"):
            entry.wait_ready(1)
        self.assertEqual(self.cache.try_acquire("image")[0], "miss")

    def test_disabled_or_oversized_sidecar_still_completes(self):
        for budget in (0, 1):
            with self.subTest(budget=budget):
                self.cache.clear()
                self.hashes.resize(budget)
                _, entry = self.cache.try_acquire("image")
                self.assertTrue(self.complete(entry))
                entry.wait_ready(1)
                self.assertIsNone(self.hashes.get("image"))


class EmbeddingCapacityTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_active_tensor_survives_demotion_eviction_and_same_key_replacement(self):
        cache = MMEmbeddingCache(gpu_max_bytes=128, cpu_max_bytes=128)
        _, entry = cache.try_acquire("image")
        expected = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
        entry.complete((expected, None))
        held = entry.wait()[0]
        cache.resize(0, 128)
        self.assertEqual(entry.result[0].device.type, "cpu")
        cache.resize(0, 0)
        cache.resize(128, 128)
        _, replacement = cache.try_acquire("image")
        replacement.complete((torch.full((4, 8), 99.0, device="cuda"), None))
        self.assertNotEqual(held.data_ptr(), replacement.wait()[0].data_ptr())
        self.assertTrue(
            torch.equal(
                held, torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
            )
        )
        cache.clear()
        self.assertTrue(torch.equal(held, expected))

    def test_variable_sizes_and_lru(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=64)
        entries = {}
        for key, rows in (("a", 2), ("b", 4), ("c", 2)):
            _, entry = cache.try_acquire(key)
            entry.complete((torch.full((rows, 2), float(rows)), None))
            entries[key] = entry
        self.assertEqual(cache.stats()["resident_bytes"], 64)
        cache.try_acquire("a")
        _, latest = cache.try_acquire("d")
        latest.complete((torch.zeros(4, 2), None))
        self.assertIsNone(cache.peek("b"))
        self.assertIs(cache.peek("a"), entries["a"])
        self.assertIs(cache.peek("c"), entries["c"])
        self.assertEqual(cache.stats()["resident_bytes"], 64)

    def test_backing_storage_and_extra_outputs_are_charged(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=512)
        batch = torch.ones(100)
        position = torch.ones(2, dtype=torch.int64)
        extra = torch.ones(8)
        _, entry = cache.try_acquire("view")
        # Two small, distinct views share and retain the 400-byte batch.
        result = ([batch[:2].view(1, 2), batch[10:12].view(1, 2)], position, extra)
        entry.complete(result)
        self.assertEqual(cache.stats()["resident_bytes"], 400 + 16 + 32)
        self.assertEqual(cache.stats()["resident_tokens"], 2)
        cache.remove("view")
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        # Evicting the index entry does not invalidate a waiting request.
        self.assertIs(entry.wait(), result)

    def test_oversized_result_bypasses_cache_without_flushing_small_results(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, small = cache.try_acquire("small")
        small.complete(torch.ones(4))
        _, large = cache.try_acquire("large")
        result = torch.arange(100)
        large.complete(result)
        self.assertIs(cache.peek("small"), small)
        self.assertIsNone(cache.peek("large"))
        self.assertEqual(cache.stats()["resident_bytes"], 16)
        self.assertIs(large.wait(), result)

    def test_pending_dedup_survives_resize_and_disabled_completion(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=16)
        _, pending = cache.try_acquire("pending")
        self.assertIs(cache.try_acquire("pending")[1], pending)
        _, ready = cache.try_acquire("ready")
        ready.complete(torch.ones(4))
        cache.resize(0, 0)
        self.assertIsNone(cache.peek("ready"))
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        pending.complete(torch.ones(2))
        self.assertIsNone(cache.peek("pending"))
        self.assertEqual(pending.wait().numel(), 2)
        cache.resize(0, 16)
        self.assertEqual(cache.try_acquire("pending")[0], "miss")

    def test_concurrent_acquire_and_completion(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=128)
        barrier = threading.Barrier(8)

        def acquire(_):
            barrier.wait(timeout=5)
            return cache.try_acquire("shared")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(acquire, range(8)))
        self.assertEqual(sum(state == "miss" for state, _ in claims), 1)
        entry = claims[0][1]
        self.assertTrue(all(claimed is entry for _, claimed in claims))
        result = torch.ones(4)
        entry.complete(result)
        self.assertTrue(all(claimed.wait() is result for _, claimed in claims))

        def complete(i):
            _, current = cache.try_acquire(str(i))
            value = torch.full((i % 8 + 1,), i, dtype=torch.int32)
            current.complete(value)
            self.assertTrue(torch.equal(current.wait(), value))
            self.assertLessEqual(cache.stats()["resident_bytes"], 128)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(complete, range(80)))
        self.assertEqual(cache.stats()["pending_entries"], 0)

    def test_removed_pending_completion_cannot_charge_replacement(self):
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, old = cache.try_acquire("key")
        cache.remove("key")
        _, new = cache.try_acquire("key")
        old.complete(torch.ones(100))
        new.complete(torch.ones(2))
        self.assertIs(cache.peek("key"), new)
        self.assertEqual(cache.stats()["resident_bytes"], 8)


@unittest.skipUnless(
    torch.cuda.is_available(), "requires CUDA for actual tier transfers"
)
class EmbeddingGpuCpuTest(unittest.TestCase):
    def add(self, cache, key, count=8):
        _, entry = cache.try_acquire(str(key))
        value = torch.full((count,), float(key), device="cuda")
        entry.complete(value)
        return entry

    def test_gpu_spill_cpu_promotion_and_cpu_lru_eviction(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        hashes = MMHashKeyCache(max_bytes=8192)
        a = self.add(cache, 1)
        hashes.put("1", [torch.tensor([91])], a.generation)
        b = self.add(cache, 2)
        c = self.add(cache, 3)
        self.assertEqual(a.result.device.type, "cpu")
        self.assertEqual(c.result.device.type, "cuda")
        before = cache.stats()
        self.assertTrue(hashes.metadata(["1"], cache)["entries"][0]["hit"])
        self.assertEqual(cache.stats(), before)
        self.assertEqual(cache.try_acquire("1")[0], "complete")
        restored = a.wait()
        self.assertEqual(restored.device, torch.device("cuda:0"))
        self.assertEqual(restored.tolist(), [1.0] * 8)
        self.assertEqual(c.result.device.type, "cpu")
        self.assertEqual(cache.stats()["promotion"], 1)
        self.assertEqual(hashes.get("1", a.generation)[0].tolist(), [91])
        self.add(cache, 4)
        self.assertIsNone(cache.peek("2"))
        self.assertEqual(cache.stats()["gpu_resident_bytes"], 32)
        self.assertEqual(cache.stats()["cpu_resident_bytes"], 64)
        # Both an already-returned tensor and an evicted entry remain usable.
        self.assertEqual(restored.tolist(), [1.0] * 8)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)
        self.assertIsNone(cache.peek("2"))

    def test_mixed_devices_nested_outputs_views_and_dtypes_round_trip(self):
        cache = MMEmbeddingCache(gpu_max_bytes=512, cpu_max_bytes=2048)
        batch = torch.arange(128, device="cuda", dtype=torch.bfloat16)
        embedding = batch[8:16].view(2, 4)
        position = torch.arange(4, dtype=torch.int64)
        extra = torch.arange(6, device="cuda", dtype=torch.int16).reshape(2, 3).t()
        result = ([embedding, embedding], position, {"extra": extra, "none": None})
        _, entry = cache.try_acquire("mixed")
        entry.complete(result)
        generation = entry.generation
        self.add(cache, 2, count=128)
        self.assertEqual(entry.result[0][0].device.type, "cpu")
        self.assertIs(entry.result[0][0], entry.result[0][1])
        self.assertIs(entry.result[1], position)
        actual = entry.wait()
        self.assertEqual(entry.generation, generation)
        for expected, got in (
            (embedding, actual[0][0]),
            (position, actual[1]),
            (extra, actual[2]["extra"]),
        ):
            self.assertEqual(got.device, expected.device)
            self.assertEqual(got.dtype, expected.dtype)
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(torch.equal(got, expected))
        self.assertIs(actual[0][0], actual[0][1])
        self.assertIsNone(actual[2]["none"])
        self.assertLessEqual(cache.stats()["gpu_resident_bytes"], 512)
        self.assertLessEqual(cache.stats()["cpu_resident_bytes"], 2048)

    def test_disable_either_tier_and_oversized_results(self):
        for gpu_bytes in (0, 16):
            cache = MMEmbeddingCache(gpu_max_bytes=gpu_bytes, cpu_max_bytes=64)
            entry = self.add(cache, 1)
            self.assertEqual(entry.tier, "cpu")
            self.assertEqual(entry.wait().device.type, "cuda")
            self.assertEqual(entry.tier, "cpu")
            self.assertEqual(cache.stats()["gpu_resident_bytes"], 0)
            huge = self.add(cache, 2, count=100)
            self.assertIsNone(cache.peek("2"))
            self.assertIs(cache.peek("1"), entry)
            self.assertEqual(huge.wait().tolist(), [2.0] * 100)
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=0)
        first = self.add(cache, 1)
        self.add(cache, 2)
        self.assertIsNone(cache.peek("1"))
        self.assertEqual(cache.stats()["cpu_resident_bytes"], 0)
        self.assertEqual(first.wait().device.type, "cuda")

    def test_concurrent_cpu_hit_copies_once(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        entry = self.add(cache, 1)
        self.add(cache, 2)
        barrier = threading.Barrier(8)

        def read(_):
            _, claimed = cache.try_acquire("1")
            barrier.wait(timeout=5)
            return claimed.wait(timeout=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(8)))
        self.assertTrue(all(value is results[0] for value in results))
        self.assertEqual(cache.stats()["promotion"], 1)
        self.assertEqual(entry.tier, "gpu")
        self.assertEqual(results[0].tolist(), [1.0] * 8)

    def test_concurrent_insert_hit_evict_and_clear_preserve_each_value(self):
        cache = MMEmbeddingCache(gpu_max_bytes=128, cpu_max_bytes=256)

        def request(i):
            key = str(i % 13)
            state, entry = cache.try_acquire(key)
            if state == "miss":
                entry.complete(torch.full((8,), float(key), device="cuda"))
            value = entry.wait(timeout=10)
            self.assertEqual(value.device.type, "cuda")
            self.assertEqual(value.tolist(), [float(key)] * 8)
            stats = cache.stats()
            self.assertLessEqual(stats["gpu_resident_bytes"], 128)
            self.assertLessEqual(stats["cpu_resident_bytes"], 256)
            if i % 19 == 0:
                cache.clear()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(request, range(160)))
        cache.clear()
        self.assertEqual(cache.stats()["resident_bytes"], 0)

    def test_promotion_failure_keeps_cpu_generation_for_retry(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        entry = self.add(cache, 1)
        self.add(cache, 2)
        generation = entry.generation
        with patch(
            "rtp_llm.multimodal.mm_embedding_cache._copy_result_to_devices",
            side_effect=torch.OutOfMemoryError("test H2D allocation"),
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                entry.wait()
        self.assertIs(cache.peek("1"), entry)
        self.assertEqual(entry.generation, generation)
        self.assertEqual(entry.result.device.type, "cpu")
        self.assertEqual(entry.wait().tolist(), [1.0] * 8)
        self.assertEqual(entry.tier, "gpu")

    def test_spill_failure_does_not_fail_completed_computation(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        a = self.add(cache, 1)
        with patch(
            "rtp_llm.multimodal.mm_embedding_cache._copy_result_to_devices",
            side_effect=MemoryError("test host allocation"),
        ):
            b = self.add(cache, 2)
        self.assertIsNone(cache.peek("1"))
        self.assertIs(cache.peek("2"), b)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)
        self.assertEqual(a.wait().tolist(), [1.0] * 8)
        self.assertEqual(cache.stats()["transfer_error"], 1)

    def test_metadata_and_hot_reads_do_not_wait_for_an_unrelated_spill(self):
        from rtp_llm.multimodal.mm_embedding_cache import _copy_result_to_devices

        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=64)
        a = self.add(cache, 1)
        b = self.add(cache, 2)
        entered, release = threading.Event(), threading.Event()

        def copy(result, devices):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release spill")
            return _copy_result_to_devices(result, devices)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with patch(
                "rtp_llm.multimodal.mm_embedding_cache._copy_result_to_devices", copy
            ):
                inserting = pool.submit(self.add, cache, 3)
                try:
                    self.assertTrue(entered.wait(timeout=5))

                    def probe():
                        self.assertIs(cache.peek("1"), a)
                        self.assertEqual(
                            cache.try_acquire("2")[1].wait().tolist(), [2.0] * 8
                        )
                        cache.stats()

                    pool.submit(probe).result(timeout=2)
                finally:
                    release.set()
                inserting.result(timeout=5)
        self.assertEqual(a.tier, "cpu")
        self.assertEqual(b.tier, "gpu")

    def test_resize_spills_and_old_waiter_cannot_replace_new_generation(self):
        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=64)
        a = self.add(cache, 1)
        b = self.add(cache, 2)
        cache.resize(0, 64)
        self.assertEqual(a.tier, "cpu")
        self.assertEqual(b.tier, "cpu")
        cache.resize(0, 32)
        self.assertIsNone(cache.peek("1"))
        cache.resize(32, 64)
        replacement = self.add(cache, 1)
        self.assertNotEqual(replacement.generation, a.generation)
        a.wait()
        self.assertIs(cache.peek("1"), replacement)
        cache.resize(0, 0)
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        self.assertEqual(cache.stats()["resident_entries"], 0)
        self.assertEqual(b.wait().tolist(), [2.0] * 8)

    def test_wait_timeout_includes_transfer_queue(self):
        cache = MMEmbeddingCache(gpu_max_bytes=32, cpu_max_bytes=64)
        entry = self.add(cache, 1)
        self.add(cache, 2)
        with cache._transfer_lock:
            with self.assertRaises(TimeoutError):
                entry.wait(timeout=0.01)
        self.assertEqual(entry.wait().tolist(), [1.0] * 8)

    def test_cpu_metadata_on_gpu_entries_counts_against_cpu_budget(self):
        cache = MMEmbeddingCache(gpu_max_bytes=64, cpu_max_bytes=8)
        for key in ("1", "2"):
            _, entry = cache.try_acquire(key)
            entry.complete((torch.ones(4, device="cuda"), torch.ones(2)))
        self.assertIsNone(cache.peek("1"))
        self.assertEqual(cache.stats()["gpu_resident_bytes"], 16)
        self.assertEqual(cache.stats()["cpu_resident_bytes"], 8)

    def test_non_default_producer_stream_and_storage_release(self):
        import weakref

        cache = MMEmbeddingCache(gpu_max_bytes=4096, cpu_max_bytes=8192)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            value = torch.arange(1024, device="cuda", dtype=torch.float32) * 3
            ref = weakref.ref(value)
            _, entry = cache.try_acquire("stream")
            entry.complete(value)
        del value
        self.add(cache, 2, count=1024)
        self.assertIsNone(ref())
        with torch.cuda.stream(torch.cuda.Stream()):
            actual = entry.wait()
            expected = torch.arange(1024, device="cuda", dtype=torch.float32) * 3
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(entry.tier, "gpu")


class HashCapacityTest(unittest.TestCase):
    def test_capacity_accounts_for_hash_length_and_python_metadata(self):
        cache = MMHashKeyCache(max_bytes=5120)
        for i in range(3):
            cache.put(str(i), [torch.ones(1)], "g")
        self.assertEqual(cache.stats()["resident_entries"], 3)
        self.assertGreater(cache.stats()["resident_bytes"], 3 * 4)
        cache.put("big", [torch.ones(1024)], "g")
        self.assertEqual(cache.keys(), ["big"])
        self.assertLessEqual(cache.stats()["resident_bytes"], 5120)
        self.assertGreater(cache.stats()["resident_bytes"], 1024 * 4)

    def test_replacement_resize_and_clear_balance_bytes(self):
        cache = MMHashKeyCache(max_bytes=8192)
        cache.put("a", [torch.ones(8)], "g")
        first_size = cache.stats()["resident_bytes"]
        cache.put("a", [torch.ones(32)], "g")
        self.assertEqual(cache.stats()["resident_bytes"], first_size + 24 * 4)
        cache.put("a", [torch.ones(8)], "g")
        self.assertEqual(cache.stats()["resident_bytes"], first_size)
        cache.put("b", [torch.ones(8)], "g")
        cache.resize(first_size)
        self.assertEqual(cache.keys(), ["b"])
        self.assertEqual(cache.stats()["resident_bytes"], first_size)
        cache.clear()
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        self.assertEqual(cache.keys(), [])

    def test_real_hits_touch_lru_but_probes_and_wrong_generation_do_not(self):
        cache = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        for key in ("a", "b"):
            _, entry = embeddings.try_acquire(key)
            entry.complete(torch.ones(1, 2))
            cache.put(key, [torch.ones(1)], entry.generation)
        limit = cache.stats()["resident_bytes"]
        cache.resize(limit)
        self.assertTrue(cache.metadata(["a"], embeddings)["entries"][0]["hit"])
        self.assertIsNone(cache.get("a", "wrong-generation"))
        self.assertEqual(cache.keys(), ["a", "b"])
        self.assertIsNotNone(cache.get("a", embeddings.peek("a").generation))
        self.assertEqual(cache.keys(), ["b", "a"])
        cache.put("c", [torch.ones(1)], embeddings.peek("a").generation)
        self.assertEqual(cache.keys(), ["a", "c"])

    def test_cpu_owned_int32_copies_and_generation(self):
        cache = MMHashKeyCache(max_bytes=8192)
        original = torch.tensor([-1, 2147483647], dtype=torch.int64)
        cache.put("a", [original], "g1")
        original.zero_()
        hashes = cache.get("a", "g1")
        self.assertEqual(hashes[0].tolist(), [-1, 2147483647])
        self.assertEqual(hashes[0].dtype, torch.int32)
        self.assertEqual(hashes[0].device.type, "cpu")
        self.assertIsNone(cache.get("a", "g2"))

    def test_oversized_update_drops_stale_generation_only(self):
        cache = MMHashKeyCache(max_bytes=4096)
        cache.put("old", [torch.ones(1)], "g1")
        cache.put("keep", [torch.ones(1)], "g1")
        cache.put("old", [torch.ones(4096)], "g2")
        self.assertEqual(cache.keys(), ["keep"])
        # Keys and generation strings count even without a hash payload.
        cache.put("x" * 8192, [], "g")
        self.assertEqual(cache.keys(), ["keep"])

    def test_hash_metadata_survives_embedding_eviction(self):
        cache = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=8)
        _, entry = embeddings.try_acquire("a")
        entry.complete(torch.ones(1, 2))
        cache.put("a", [torch.ones(1)], entry.generation)
        self.assertTrue(cache.metadata(["a"], embeddings)["entries"][0]["hit"])
        embeddings.resize(0, 0)
        self.assertEqual(cache.keys(), ["a"])
        metadata = cache.metadata(["a"], embeddings)["entries"][0]
        self.assertTrue(metadata["hit"])
        self.assertTrue(metadata["hash_hit"])
        self.assertFalse(metadata["embedding_hit"])
        self.assertIsNone(metadata["embedding_tier"])
        self.assertEqual(metadata["feature_hashes"], [1])
        self.assertEqual(metadata["entry_generation"], entry.generation)

    def test_hash_and_embedding_availability_are_independent(self):
        hashes = MMHashKeyCache(max_bytes=8192)
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=32)
        _, ready = embeddings.try_acquire("ready")
        ready.complete(torch.ones(2))
        hashes.put("history", [torch.tensor([71, 72])], "old-generation")
        embeddings.try_acquire("history")
        before = embeddings.stats(), hashes.keys(), hashes.stats()
        entries = hashes.metadata(["ready", "history", "absent"], embeddings)["entries"]
        self.assertFalse(entries[0]["hash_hit"])
        self.assertTrue(entries[0]["embedding_hit"])
        self.assertEqual(entries[0]["embedding_tier"], "cpu")
        self.assertTrue(entries[1]["hash_hit"])
        self.assertFalse(entries[1]["embedding_hit"])
        self.assertEqual(entries[1]["feature_hashes"], [71, 72])
        self.assertFalse(entries[2]["hash_hit"])
        self.assertFalse(entries[2]["embedding_hit"])
        self.assertEqual(before, (embeddings.stats(), hashes.keys(), hashes.stats()))

    def test_residency_snapshot_excludes_pending_and_does_not_touch_lru(self):
        embeddings = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=16)
        for key in ("a", "b"):
            _, entry = embeddings.try_acquire(key)
            entry.complete(torch.ones(2))
        embeddings.try_acquire("pending")
        before = embeddings.stats()
        self.assertEqual(embeddings.resident_tiers(limit=1), {"b": "cpu"})
        self.assertEqual(
            embeddings.resident_tiers(["a", "pending", "absent"]), {"a": "cpu"}
        )
        self.assertEqual(embeddings.resident_tiers(limit=0), {})
        self.assertEqual(before, embeddings.stats())
        _, latest = embeddings.try_acquire("c")
        latest.complete(torch.ones(2))
        self.assertIsNone(embeddings.peek("a"))

    def test_concurrent_puts_replacements_and_resize_stay_bounded(self):
        cache = MMHashKeyCache(max_bytes=8192)

        def write(worker):
            for i in range(30):
                cache.put(str((worker + i) % 12), [torch.arange(i * 8)], str(i))
                self.assertLessEqual(cache.stats()["resident_bytes"], 8192)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(8)))
        cache.resize(0)
        cache.put("disabled", [torch.ones(1)])
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        self.assertEqual(cache.keys(), [])

    def test_disabled_and_negative_budgets(self):
        with self.assertRaises(ValueError):
            MMHashKeyCache(max_bytes=-1)
        cache = MMHashKeyCache(max_bytes=0)
        self.assertFalse(cache.enabled)
        with self.assertRaises(ValueError):
            cache.resize(-1)
        for gpu, cpu in ((-1, 0), (0, -1)):
            with self.assertRaises(ValueError):
                MMEmbeddingCache(gpu_max_bytes=gpu, cpu_max_bytes=cpu)
        cache = MMEmbeddingCache(gpu_max_bytes=0, cpu_max_bytes=0)
        self.assertFalse(cache.enabled)
        with self.assertRaises(ValueError):
            cache.resize(0, -1)


if __name__ == "__main__":
    unittest.main()
