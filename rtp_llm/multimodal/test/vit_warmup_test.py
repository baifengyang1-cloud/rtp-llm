import os
import threading
import time
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from rtp_llm.multimodal.qwen_vit_warmup import qwen_warmup_batches, warmup_shapes
from rtp_llm.multimodal.vit_request_deadline import (
    current_vit_deadline,
    use_vit_deadline,
)
from rtp_llm.multimodal.vit_warmup import _device_warmup_lock, warmup_vit


class VitWarmupTest(TestCase):
    def test_tiny_qwen35_real_forward_accepts_synthetic_inputs(self):
        from transformers import Qwen2VLImageProcessor

        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
            Qwen3_5MoeImageEmbedding,
        )
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
            Qwen3_5MoeVisionConfig,
            Qwen3_5MoeVisionModel,
        )

        config = Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=16,
            num_position_embeddings=64,
        )
        config._attn_implementation = "sdpa"
        part = Qwen3_5MoeImageEmbedding.__new__(Qwen3_5MoeImageEmbedding)
        part.visual = Qwen3_5MoeVisionModel(config).eval()
        part.mm_processor = SimpleNamespace(
            image_processor=Qwen2VLImageProcessor(
                patch_size=16, temporal_patch_size=2, merge_size=2
            )
        )
        for decode in ("0", "1"):
            with (
                self.subTest(decode=decode),
                mock.patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_VIT_WARMUP_BATCHES": "[[[64,96],[96,64]]]",
                        "RTP_LLM_VIT_WARMUP_REPEATS": "1",
                        "RTP_LLM_QWEN35_GPU_PREPROCESS": "1",
                        "RTP_LLM_QWEN_IMAGE_GPU_DECODE": decode,
                    },
                ),
                mock.patch("rtp_llm.ops.ensure_engine_ops_loaded"),
                mock.patch("rtp_llm.ops.get_multimodal_feature_hash") as hash_feature,
            ):
                warmup_vit(part, "cpu", 8, 10)
            self.assertEqual(hash_feature.call_count, 2)
            self.assertEqual(hash_feature.call_args.args[0].shape, (6, 16))

    def test_common_shapes_and_compositions_respect_caps(self):
        with mock.patch.dict(os.environ, {"RTP_LLM_VIT_WARMUP_BATCHES": "null"}):
            batches = list(warmup_shapes(32, 2, 2000000))
        self.assertIn([[512, 512]], batches)
        self.assertIn([[1408, 1408]], batches)
        self.assertIn([[800, 800], [800, 800]], batches)
        self.assertNotIn([[512, 512]] * 3, batches)

    def test_invalid_shapes_fail_startup(self):
        for value in ("[]", "[[[511,512]]]", "[[[true,512]]]", "{}"):
            with mock.patch.dict(os.environ, {"RTP_LLM_VIT_WARMUP_BATCHES": value}):
                with self.assertRaises(ValueError):
                    list(warmup_shapes(32, 8, 2000000))

    def test_preload_forward_hash_and_sync_order(self):
        calls = []
        part = SimpleNamespace(
            get_warmup_batches=lambda _: iter([([[512, 512]], ["synthetic"])]),
            batched_embedding=lambda data, types: calls.append("forward")
            or [(torch.zeros(2, 3), None)],
        )
        with (
            mock.patch.dict(os.environ, {"RTP_LLM_VIT_WARMUP_REPEATS": "2"}),
            mock.patch(
                "rtp_llm.ops.ensure_engine_ops_loaded",
                side_effect=lambda: calls.append("preload"),
            ),
            mock.patch(
                "rtp_llm.ops.get_multimodal_feature_hash",
                side_effect=lambda _: calls.append("hash"),
            ),
            mock.patch(
                "torch.cuda.synchronize", side_effect=lambda _: calls.append("sync")
            ),
        ):
            warmup_vit(part, "cuda:0", 8, 2)
        self.assertEqual(
            calls, ["preload", "forward", "hash", "sync", "forward", "hash", "sync"]
        )

    def test_per_device_warmup_is_serialized(self):
        entered = threading.Event()

        def other():
            with _device_warmup_lock("cuda:0", 1):
                entered.set()

        with _device_warmup_lock("cuda:0", 1):
            worker = threading.Thread(target=other)
            worker.start()
            self.assertFalse(entered.wait(0.05))
        worker.join(1)
        self.assertTrue(entered.is_set())

    def test_deadline_context_is_restored(self):
        deadline = lambda: time.monotonic() + 1
        with use_vit_deadline(deadline):
            self.assertIs(current_vit_deadline(), deadline)
        self.assertIsNone(current_vit_deadline())

    def test_synthetic_inputs_follow_processor_contract(self):
        from transformers import Qwen2VLImageProcessor

        processor = Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2
        )
        part = SimpleNamespace(
            mm_processor=SimpleNamespace(image_processor=processor),
            visual=SimpleNamespace(patch_size=16, spatial_merge_size=2),
            get_batch_work_budget=lambda _: None,
            estimate_work=lambda *_: None,
        )
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_VIT_WARMUP_BATCHES": "[[[64,96],[64,96]]]",
                "RTP_LLM_QWEN35_GPU_PREPROCESS": "1",
                "RTP_LLM_QWEN_IMAGE_GPU_DECODE": "0",
            },
        ):
            shapes, data = next(qwen_warmup_batches(part, 8))
        self.assertEqual(shapes, [[64, 96]] * 2)
        self.assertEqual(data[0][0].to_patches("cpu", torch.float32).shape, (24, 1536))
        self.assertEqual(data[0][1].tolist(), [[1, 4, 6]])

    def test_zero_service_cap_preserves_model_budget(self):
        from transformers import Qwen2VLImageProcessor

        processor = Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2
        )
        part = SimpleNamespace(
            mm_processor=SimpleNamespace(image_processor=processor),
            visual=SimpleNamespace(patch_size=16, spatial_merge_size=2),
            get_batch_work_budget=lambda _: None,
            estimate_work=lambda *_: None,
        )
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_VIT_WARMUP_BATCHES": "[[[64,96]]]",
                "RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "0",
                "RTP_LLM_QWEN35_GPU_PREPROCESS": "1",
                "RTP_LLM_QWEN_IMAGE_GPU_DECODE": "0",
            },
        ):
            self.assertEqual(len(list(qwen_warmup_batches(part, 8))), 1)
