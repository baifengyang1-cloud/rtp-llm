import os
import pickle
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from rtp_llm.multimodal.qwen_image_preprocess import (
    default_resize_size,
    limit_image_size,
    maybe_draft_jpeg,
    prepare_device_pixels,
    resize_qwen_image,
    service_image_max_pixels,
    vision_pixel_values,
)


class QwenImagePreprocessTest(unittest.TestCase):
    def setUp(self):
        self.processor = Qwen2VLImageProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            min_pixels=1024,
            max_pixels=262144,
        )

    def test_service_pixel_cap_preserves_disabled_policy(self):
        image = Image.new("RGB", (2048, 2048))
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "0"}):
            self.assertIsNone(limit_image_size(image, self.processor, 32, None))
            self.assertEqual(
                limit_image_size(image, self.processor, 32, (4096, 4096)),
                (4096, 4096),
            )

    def test_service_pixel_cap_bounds_request_and_model_sizes(self):
        image = Image.new("RGB", (32, 32))
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "2000000"}):
            # A square is floored to a legal grid, below the decimal 2MP cap.
            self.assertEqual(
                limit_image_size(image, self.processor, 32, (4096, 4096)),
                (1408, 1408),
            )
            self.assertEqual(
                limit_image_size(image, self.processor, 32, (1024, 1024)),
                (1024, 1024),
            )
            for size in [(6144, 8192), (8192, 6144), (1408, 1440)]:
                h, w = limit_image_size(image, self.processor, 32, size)
                self.assertLessEqual(h * w, 2000000)
                self.assertEqual((h % 32, w % 32), (0, 0))

    def test_service_pixel_cap_also_covers_cpu_fallback(self):
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "2000000"}):
            processor = Qwen2VLImageProcessor(
                patch_size=16, merge_size=2, min_pixels=1024, max_pixels=16777216
            )
            for mode in ["RGB", "RGBA", "L"]:
                image = Image.new(mode, (1600, 1600))
                self.assertEqual(
                    limit_image_size(image, processor, 32, None), (1408, 1408)
                )
                self.assertEqual(
                    limit_image_size(image, self.processor, 32, None), (512, 512)
                )

    def test_service_pixel_cap_rejects_invalid_configuration(self):
        for value in ["-1", "1023", "bad", "2.5"]:
            with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": value}):
                with self.assertRaises(ValueError):
                    service_image_max_pixels(32)

    def test_service_pixel_cap_wins_over_model_minimum(self):
        processor = Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, min_pixels=4000000, max_pixels=16777216
        )
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "2000000"}):
            target = limit_image_size(Image.new("RGB", (640, 640)), processor, 32, None)
        self.assertEqual(target, (1408, 1408))

    def test_early_resize_preserves_hf_pixels_and_grid(self):
        rng = np.random.default_rng(17)
        for height, width in [(32, 64), (175, 121), (768, 1024), (121, 175)]:
            image = Image.fromarray(
                rng.integers(256, size=(height, width, 3), dtype=np.uint8)
            )
            reference = self.processor(image, return_tensors="pt")
            rh, rw = default_resize_size(image, self.processor, 32)
            resized = resize_qwen_image(
                image, self.processor, (rh, rw), processor_resize=True
            )
            actual = self.processor(resized, do_resize=False, return_tensors="pt")
            with self.subTest(size=(height, width)):
                torch.testing.assert_close(
                    actual["pixel_values"], reference["pixel_values"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual["image_grid_thw"],
                    reference["image_grid_thw"],
                    rtol=0,
                    atol=0,
                )

    def _check_device_pixels(self, device):
        # Every uint8 value occurs in every channel. Distinct channel/position
        # patterns detect normalization, temporal-repeat and merge-order errors.
        pixels = np.arange(32 * 256 * 3, dtype=np.uint32).reshape(32, 256, 3)
        image = Image.fromarray((pixels % 256).astype(np.uint8))
        for patch_size in (8, 16):
            for temporal in (1, 2):
                proc = Qwen2VLImageProcessor(
                    patch_size=patch_size, merge_size=2, temporal_patch_size=temporal
                )
                reference = proc(image, do_resize=False, return_tensors="pt")
                payload, grid = prepare_device_pixels(image, proc)
                self.assertEqual(payload.pixels.device.type, "cpu")
                self.assertEqual(grid.device.type, "cpu")
                payload = pickle.loads(pickle.dumps(payload))
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    with self.subTest(
                        device=device, dtype=dtype, patch=patch_size, temporal=temporal
                    ):
                        actual = vision_pixel_values(payload, device, dtype)
                        expected = reference["pixel_values"].to(
                            device=device, dtype=dtype
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            grid, reference["image_grid_thw"], rtol=0, atol=0
                        )

    def test_deferred_pixels_match_hf_and_survive_process_transport(self):
        self._check_device_pixels(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires an accelerator")
    def test_accelerator_pixels_match_hf_at_vision_boundary(self):
        self._check_device_pixels(torch.device("cuda"))

    def test_unsupported_images_keep_cpu_processor(self):
        for mode in ("L", "RGBA"):
            image = Image.new(mode, (32, 64))
            self.assertIsNone(default_resize_size(image, self.processor, 32))
            self.assertIsNone(prepare_device_pixels(image, self.processor))
        self.assertIsNone(
            prepare_device_pixels(Image.new("RGB", (33, 64)), self.processor)
        )
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_EARLY_RESIZE": "0"}):
            self.assertIsNone(
                default_resize_size(Image.new("RGB", (1024, 768)), self.processor, 32)
            )
        tensor = torch.randn(4, 32)
        self.assertIs(
            vision_pixel_values(tensor, torch.device("cpu"), tensor.dtype), tensor
        )

    def test_jpeg_draft_is_opt_in_and_preserves_planned_grid(self):
        data = BytesIO()
        Image.new("RGB", (4096, 2304), "red").save(data, format="JPEG")
        original = data.getvalue()
        image = Image.open(BytesIO(original))
        target = default_resize_size(image, self.processor, 32)
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_JPEG_DRAFT": "0"}):
            self.assertFalse(maybe_draft_jpeg(image, target))
            self.assertEqual(image.size, (4096, 2304))
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_JPEG_DRAFT": "1"}):
            self.assertTrue(maybe_draft_jpeg(image, target))
        self.assertLess(image.width * image.height, 4096 * 2304)
        image.load()
        resized = image.resize((target[1], target[0]))
        actual = self.processor(resized, do_resize=False, return_tensors="pt")
        reference = self.processor(Image.open(BytesIO(original)), return_tensors="pt")
        torch.testing.assert_close(
            actual["image_grid_thw"], reference["image_grid_thw"], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
