import json
import os
import pickle
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from rtp_llm.multimodal.qwen_image_decode import QwenPngPixels, prepare_device_png
from rtp_llm.multimodal.qwen_image_preprocess import (
    resize_qwen_image,
    vision_pixel_values,
)
from transformers import Qwen2VLImageProcessor

MODULE = "rtp_llm.multimodal.qwen_image_decode"


class QwenPngPreprocessTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN_IMAGE_PNG_GPU_RESIZE": "1",
                "RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS": "33554432",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.processor = Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2
        )

    def encoded(self, size=(128, 96), mode="RGB"):
        arr = np.random.default_rng(53).integers(
            256, size=(size[1], size[0], 3), dtype=np.uint8
        )
        image = Image.fromarray(arr).convert(mode)
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def prepare(self, encoded, target=(64, 64), processor_resize=False, timing=None):
        return prepare_device_png(
            Image.open(BytesIO(encoded)),
            encoded,
            self.processor,
            target,
            processor_resize,
            timing if timing is not None else {},
        )

    def test_cpu_only_decode_preserves_all_pixels_and_transport(self):
        encoded = self.encoded()
        with patch(
            "torch.cuda.is_available", side_effect=AssertionError("worker CUDA probe")
        ):
            payload, grid = self.prepare(encoded)
        self.assertIsInstance(payload, QwenPngPixels)
        self.assertEqual(payload.pixels.device.type, "cpu")
        np.testing.assert_array_equal(
            payload.pixels.permute(1, 2, 0).numpy(),
            np.array(Image.open(BytesIO(encoded))),
        )
        copied, copied_grid = pickle.loads(pickle.dumps((payload, grid)))
        torch.testing.assert_close(copied.pixels, payload.pixels)
        self.assertEqual(copied_grid.tolist(), [[1, 4, 4]])

    def test_cpu_fallback_keeps_exact_legacy_patch_values_and_grid(self):
        encoded = self.encoded()
        for target in [(96, 128), (64, 64), (128, 160)]:
            for backend in [False, True]:
                payload, grid = self.prepare(encoded, target, backend)
                ref_image = resize_qwen_image(
                    Image.open(BytesIO(encoded)),
                    self.processor,
                    target,
                    processor_resize=backend,
                )
                ref = self.processor(ref_image, do_resize=False, return_tensors="pt")
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    with self.subTest(target=target, dtype=dtype, backend=backend):
                        actual = vision_pixel_values(payload, "cpu", dtype)
                        torch.testing.assert_close(
                            actual, ref["pixel_values"].to(dtype), rtol=0, atol=0
                        )
                        torch.testing.assert_close(grid, ref["image_grid_thw"])

    def test_missing_or_rejecting_libpng_still_defers_resize(self):
        encoded = self.encoded()
        for decoder in [
            None,
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("libpng unavailable")),
        ]:
            timing = {}
            with patch(MODULE + "._png_decoder", return_value=decoder):
                payload, _ = self.prepare(encoded, timing=timing)
            self.assertIsInstance(payload, QwenPngPixels)
            self.assertEqual(tuple(payload.pixels.shape), (3, 96, 128))
            self.assertEqual(payload.target_hw, (64, 64))
            self.assertEqual(timing["decode_backend"], "pillow_cpu")

    def test_native_decoder_used_and_invalid_output_falls_back(self):
        encoded = self.encoded()
        expected = torch.from_numpy(np.array(Image.open(BytesIO(encoded)))).permute(
            2, 0, 1
        )
        timing = {}
        with patch(MODULE + "._png_decoder", return_value=lambda *a, **k: expected):
            payload, _ = self.prepare(encoded, timing=timing)
        self.assertEqual(timing["decode_backend"], "libpng_cpu")
        torch.testing.assert_close(payload.pixels, expected)
        with patch(
            MODULE + "._png_decoder", return_value=lambda *a, **k: expected.float()
        ):
            payload, _ = self.prepare(encoded, timing=timing)
        self.assertEqual(timing["decode_backend"], "pillow_cpu")
        self.assertEqual(payload.pixels.dtype, torch.uint8)

    def test_non_rgb_animated_16bit_and_disabled_use_legacy_path(self):
        for mode in ["RGBA", "L", "LA", "P"]:
            self.assertIsNone(self.prepare(self.encoded(mode=mode)))
        encoded = self.encoded()
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_PNG_GPU_RESIZE": "0"}):
            self.assertIsNone(self.prepare(encoded))
        with patch.dict(
            os.environ, {"RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS": "1000"}
        ):
            self.assertIsNone(self.prepare(encoded))
        self.assertIsNone(self.prepare(encoded, target=(33, 64)))
        self.assertIsNone(self.prepare(encoded, target=None))
        # RGB16 may be exposed by Pillow as RGB8, but must retain Pillow's conversion.
        altered = bytearray(encoded)
        altered[24] = 16
        self.assertIsNone(
            prepare_device_png(
                Image.open(BytesIO(encoded)),
                bytes(altered),
                self.processor,
                (64, 64),
                False,
                {},
            )
        )
        im = Image.open(BytesIO(encoded))
        with patch.object(im, "is_animated", True, create=True):
            self.assertIsNone(
                prepare_device_png(im, encoded, self.processor, (64, 64), False, {})
            )

    def test_oom_from_decode_is_not_hidden(self):
        with patch(
            MODULE + "._png_decoder", side_effect=torch.OutOfMemoryError("test")
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                self.prepare(self.encoded())

    def test_real_preprocess_entry_keeps_cap_and_never_resizes_on_cpu(self):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
            Qwen3_VLImageEmbedding,
        )
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        encoded = self.encoded(size=(2048, 2048))
        config = SimpleNamespace(height=-1, width=-1, max_pixels=4000000, min_pixels=-1)
        item = SimpleNamespace(
            mm_type=MMUrlType.IMAGE, url="test.png", mm_preprocess_config=config
        )
        vit_config = SimpleNamespace(
            download_headers={}, mm_image_max_file_size_kb=32768
        )
        with (
            patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "2000000"}),
            patch(
                "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.get_bytes_io_from_url",
                return_value=BytesIO(encoded),
            ),
            patch(
                "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.resize_qwen_image",
                side_effect=AssertionError("CPU resize must be deferred"),
            ),
        ):
            payload, grid = Qwen3_VLImageEmbedding.preprocess_input(
                [item],
                vit_config,
                SimpleNamespace(image_processor=self.processor),
                factor=32,
                defer_pixel_processing=True,
            )
        self.assertEqual(payload.target_hw, (1408, 1408))
        self.assertEqual(tuple(payload.pixels.shape), (3, 2048, 2048))
        self.assertEqual(grid.tolist(), [[1, 88, 88]])

    def test_png_and_jpeg_mixed_batch_runs_tiny_qwen35_vision(self):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
            Qwen3_5MoeImageEmbedding,
        )
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
            Qwen3_5MoeVisionConfig,
            Qwen3_5MoeVisionModel,
        )
        from rtp_llm.multimodal.qwen_image_decode import prepare_device_jpeg
        from rtp_llm.utils.base_model_datatypes import MMUrlType

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
        png = self.prepare(self.encoded())
        encoded = BytesIO()
        Image.new("RGB", (96, 64), (20, 40, 80)).save(encoded, format="JPEG")
        raw = encoded.getvalue()
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN_IMAGE_GPU_DECODE": "1",
                "RTP_LLM_QWEN_IMAGE_JPEG_DRAFT": "0",
            },
        ):
            jpeg = prepare_device_jpeg(
                Image.open(BytesIO(raw)), raw, self.processor, (64, 64), False, {}
            )
        outputs = part.batched_embedding(
            [png, jpeg], [MMUrlType.IMAGE, MMUrlType.IMAGE]
        )
        self.assertEqual(len(outputs), 2)
        for features, _ in outputs:
            self.assertEqual(tuple(features.shape), (4, 16))
            self.assertTrue(torch.isfinite(features).all())

    @unittest.skipUnless(torch.cuda.is_available(), "requires an accelerator")
    def test_real_gpu_resize_patch_consistency_and_no_cpu_fallback(self):
        encoded = self.encoded()
        for target, tensor_resize in [
            (target, backend)
            for target in [(64, 64), (128, 160), (96, 128)]
            for backend in [False, True]
        ]:
            payload, grid = self.prepare(encoded, target, True)
            payload.torchvision_resize = tensor_resize
            reference = vision_pixel_values(payload, "cpu", torch.float32)
            with self.assertLogs(level="INFO") as logs:
                actual = vision_pixel_values(payload, "cuda:0", torch.float32)
            torch.cuda.synchronize()
            record = json.loads(logs.output[-1].split("timing ", 1)[1])
            self.assertEqual(record["resize_backend"], "device")
            self.assertEqual(actual.device.type, "cuda")
            # Different CPU/CUDA uint8 rounding is at most two pixel levels.
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0.03)
            self.assertEqual(grid.tolist(), [[1, target[0] // 16, target[1] // 16]])

    @unittest.skipUnless(torch.cuda.is_available(), "requires an accelerator")
    def test_gpu_failure_falls_back_but_oom_propagates(self):
        payload, _ = self.prepare(self.encoded())
        with patch(
            MODULE + "._resize_device_rgb", side_effect=RuntimeError("unsupported")
        ):
            with self.assertLogs(level="INFO") as logs:
                result = vision_pixel_values(payload, "cuda:0", torch.float16)
            record = json.loads(logs.output[-1].split("timing ", 1)[1])
            self.assertEqual(record["resize_backend"], "cpu")
            self.assertTrue(torch.isfinite(result).all())
        with patch(
            MODULE + "._resize_device_rgb", side_effect=torch.OutOfMemoryError("test")
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                vision_pixel_values(payload, "cuda:0", torch.float16)


if __name__ == "__main__":
    unittest.main()
