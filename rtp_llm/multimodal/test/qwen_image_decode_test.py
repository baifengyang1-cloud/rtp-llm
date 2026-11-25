import json
import os
import pickle
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from rtp_llm.multimodal.qwen_image_decode import QwenJpegBytes, prepare_device_jpeg
from rtp_llm.multimodal.qwen_image_preprocess import vision_pixel_values


class QwenImageDecodeTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "RTP_LLM_QWEN_IMAGE_GPU_DECODE": "1",
                "RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS": "33554432",
                "RTP_LLM_QWEN_IMAGE_JPEG_DRAFT": "0",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.processor = Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2
        )

    def encoded_image(self, mode="RGB", fmt="JPEG", size=(96, 64), **kwargs):
        buffer = BytesIO()
        Image.new(mode, size, 127).save(buffer, format=fmt, **kwargs)
        content = buffer.getvalue()
        return Image.open(BytesIO(content)), content

    def prepare(self, image, encoded, target=(32, 64), processor_resize=False):
        return prepare_device_jpeg(
            image,
            encoded,
            self.processor,
            target,
            processor_resize,
            {"request_id": 42, "image_id": "opaque-image-id"},
        )

    def test_disabled_and_unsupported_inputs_stay_on_existing_path(self):
        image, encoded = self.encoded_image()
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_GPU_DECODE": "0"}):
            self.assertIsNone(self.prepare(image, encoded))
        for mode, fmt in [("RGB", "PNG"), ("L", "JPEG"), ("CMYK", "JPEG")]:
            image, encoded = self.encoded_image(mode, fmt)
            self.assertIsNone(self.prepare(image, encoded))
        image, encoded = self.encoded_image()
        for env in [
            {"RTP_LLM_QWEN_IMAGE_JPEG_DRAFT": "1"},
            {"RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS": "1000"},
        ]:
            with patch.dict(os.environ, env):
                self.assertIsNone(self.prepare(image, encoded))
        self.assertIsNone(self.prepare(image, encoded, target=(33, 64)))
        self.assertIsNone(self.prepare(image, encoded, target=None))
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS": "0"}):
            with self.assertRaises(ValueError):
                self.prepare(image, encoded)

    def test_planning_is_lazy_and_survives_worker_transport(self):
        image, encoded = self.encoded_image(size=(1600, 1600))
        with (
            patch.object(Image.Image, "load", side_effect=AssertionError("decoded")),
            patch("torch.cuda.is_available", side_effect=AssertionError("CUDA probe")),
        ):
            payload, grid = self.prepare(image, encoded, target=(1408, 1408))
        self.assertIsInstance(payload, QwenJpegBytes)
        self.assertEqual(payload.target_hw, (1408, 1408))
        self.assertEqual(payload.original_hw, (1600, 1600))
        self.assertEqual(payload.request_id, 42)
        self.assertEqual(grid.tolist(), [[1, 88, 88]])
        self.assertEqual(payload.lookup.device.type, "cpu")
        copied = pickle.loads(pickle.dumps((payload, grid)))
        self.assertEqual(copied[0].encoded, encoded)
        torch.testing.assert_close(copied[1], grid)

    def test_cpu_fallback_preserves_pixels_grid_and_exif_policy(self):
        rng = np.random.default_rng(17)
        image = Image.fromarray(rng.integers(256, size=(64, 96, 3), dtype=np.uint8))
        exif = Image.Exif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        encoded = buffer.getvalue()
        for target in [(64, 96), (32, 64)]:
            for processor_resize in [False, True]:
                opened = Image.open(BytesIO(encoded))
                payload, grid = self.prepare(opened, encoded, target, processor_resize)
                from rtp_llm.multimodal.qwen_image_preprocess import resize_qwen_image

                reference_image = resize_qwen_image(
                    opened, self.processor, target, processor_resize=processor_resize
                )
                reference = self.processor(
                    reference_image, do_resize=False, return_tensors="pt"
                )
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    actual = vision_pixel_values(payload, "cpu", dtype)
                    torch.testing.assert_close(
                        actual, reference["pixel_values"].to(dtype), rtol=0, atol=0
                    )
                torch.testing.assert_close(grid, reference["image_grid_thw"])

    def test_unsupported_device_decoder_falls_back_without_hiding_oom(self):
        image, encoded = self.encoded_image()
        payload, _ = self.prepare(image, encoded)
        with (
            patch(
                "rtp_llm.multimodal.qwen_image_decode._decode_device_jpeg",
                side_effect=RuntimeError("unsupported JPEG"),
            ),
            patch.object(QwenJpegBytes, "_cpu_pixels") as cpu,
            patch("rtp_llm.multimodal.qwen_image_decode.QwenImagePixels") as pack,
        ):
            pack.return_value.to_patches.return_value = "packed"
            # Avoid requiring a device for the fallback control-flow test.
            cpu.return_value = torch.zeros(3, 32, 64, dtype=torch.uint8)
            with self.assertLogs(level="INFO") as logs:
                self.assertEqual(
                    vision_pixel_values(payload, "cuda:0", torch.float16), "packed"
                )
            cpu.assert_called_once()
            record = json.loads(logs.output[-1].split("timing ", 1)[1])
            self.assertEqual(record["decode_backend"], "pillow_cpu")
            self.assertEqual(record["fallback_reason"], "RuntimeError")
        with (
            patch(
                "rtp_llm.multimodal.qwen_image_decode._decode_device_jpeg",
                side_effect=torch.OutOfMemoryError("test device OOM"),
            ),
            patch.object(QwenJpegBytes, "_cpu_pixels") as cpu,
        ):
            with self.assertRaises(torch.OutOfMemoryError):
                vision_pixel_values(payload, "cuda:0", torch.float16)
            cpu.assert_not_called()

    def test_corrupt_jpeg_is_not_silently_accepted_by_cpu_fallback(self):
        image, encoded = self.encoded_image()
        payload, _ = self.prepare(image, encoded)
        payload.encoded = b"not an image"
        with self.assertRaises(OSError):
            vision_pixel_values(payload, "cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires an accelerator")
    def test_real_device_decode_and_resize_match_reference_with_bounded_error(self):
        yy, xx = np.mgrid[:96, :128]
        pixels = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
        for progressive in [False, True]:
            for subsampling in [0, 2]:
                buffer = BytesIO()
                Image.fromarray(pixels).save(
                    buffer,
                    format="JPEG",
                    quality=95,
                    progressive=progressive,
                    subsampling=subsampling,
                )
                encoded = buffer.getvalue()
                image = Image.open(BytesIO(encoded))
                payload, grid = self.prepare(image, encoded, target=(64, 64))
                reference = self.processor(
                    image.resize((64, 64), Image.Resampling.BICUBIC),
                    do_resize=False,
                    return_tensors="pt",
                )
                with self.assertLogs(level="INFO") as logs:
                    actual = vision_pixel_values(payload, "cuda", torch.float32)
                record = json.loads(logs.output[-1].split("timing ", 1)[1])
                # A passing accelerator test must not be a hidden CPU fallback.
                self.assertEqual(record["decode_backend"], "device")
                self.assertEqual(actual.device.type, "cuda")
                torch.cuda.synchronize()
                error = (actual.cpu() - reference["pixel_values"]).abs()
                self.assertLess(error.max().item(), 0.1)
                self.assertLess(error.mean().item(), 0.02)
                torch.testing.assert_close(grid, reference["image_grid_thw"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires an accelerator")
    def test_large_device_jpeg_respects_cap_and_bounds_scratch(self):
        from rtp_llm.multimodal.qwen_image_preprocess import limit_image_size

        image, encoded = self.encoded_image(size=(5700, 5700))
        with patch.dict(os.environ, {"RTP_LLM_QWEN_IMAGE_MAX_PIXELS": "2000000"}):
            target = limit_image_size(image, self.processor, 32, (4096, 4096))
        self.assertEqual(target, (1408, 1408))
        payload, grid = self.prepare(image, encoded, target=target)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with self.assertLogs(level="INFO") as logs:
            actual = vision_pixel_values(payload, "cuda", torch.float16)
        torch.cuda.synchronize()
        record = json.loads(logs.output[-1].split("timing ", 1)[1])
        self.assertEqual(record["decode_backend"], "device")
        self.assertEqual(actual.shape, (88 * 88, 1536))
        self.assertEqual(grid.tolist(), [[1, 88, 88]])
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertLess(torch.cuda.max_memory_allocated() - baseline, 1024**3)


if __name__ == "__main__":
    unittest.main()
