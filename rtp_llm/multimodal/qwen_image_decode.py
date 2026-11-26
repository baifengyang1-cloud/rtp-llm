"""Defer supported JPEGs to the ViT device without touching CUDA in workers."""

import json
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from rtp_llm.multimodal.qwen_image_preprocess import (
    QwenImagePixels,
    device_pixel_parameters,
)


def _resize_device_rgb(chw, target_hw):
    from torchvision.transforms.v2.functional import InterpolationMode, resize

    height, width = target_hw
    source_height, source_width = chw.shape[-2:]
    # The CPU uint8 tensor kernel rounds/clamps between its horizontal and
    # vertical passes. Bound fp32 scratch by resizing row strips first.
    if source_width != width:
        strip_height = max(1, 2**22 // max(source_width, width))
        strips = [
            resize(
                rows,
                [rows.shape[-2], width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            for rows in chw.split(strip_height, dim=-2)
        ]
        chw = strips[0] if len(strips) == 1 else torch.cat(strips, dim=-2)
    if source_height != height:
        chw = resize(
            chw,
            [height, width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    return chw


def _decode_device_jpeg(encoded, device):
    from torchvision.io import decode_jpeg

    # torchvision requires compressed bytes on CPU even for device decoding.
    data = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    return decode_jpeg(data, mode="RGB", device=device)


@dataclass
class QwenJpegBytes:
    encoded: bytes
    lookup: torch.Tensor
    patch_size: int
    merge_size: int
    temporal_patch_size: int
    original_hw: Tuple[int, int]
    target_hw: Tuple[int, int]
    torchvision_resize: bool
    request_id: int = 0
    image_id: str = ""

    def _cpu_pixels(self):
        with Image.open(BytesIO(self.encoded)) as image:
            # Preserve the existing Pillow RGB path, including EXIF behaviour:
            # it does not implicitly transpose the image according to EXIF.
            image.load()
            if self.original_hw != self.target_hw:
                if self.torchvision_resize:
                    from torchvision.transforms.v2.functional import (
                        InterpolationMode,
                        resize,
                    )

                    chw = torch.from_numpy(np.array(image)).permute(2, 0, 1)
                    return resize(
                        chw.unsqueeze(0),
                        list(self.target_hw),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    )[0]
                image = image.resize(
                    (self.target_hw[1], self.target_hw[0]), Image.Resampling.BICUBIC
                )
            return torch.from_numpy(np.array(image)).permute(2, 0, 1)

    def to_patches(self, device, dtype):
        device = torch.device(device)
        started = time.monotonic_ns()
        record = {
            "request_id": self.request_id,
            "image_id": self.image_id,
            "device": str(device),
            "decode_backend": "pillow_cpu",
            "outcome": "ok",
            "original_height": self.original_hw[0],
            "original_width": self.original_hw[1],
            "target_height": self.target_hw[0],
            "target_width": self.target_hw[1],
        }
        try:
            chw = None
            if device.type == "cuda":
                try:
                    stage = "decode"
                    step = time.monotonic_ns()
                    chw = _decode_device_jpeg(self.encoded, device)
                    record["decode_host_ms"] = (time.monotonic_ns() - step) / 1e6
                    valid_device = chw.device == device or (
                        device.index is None and chw.device.type == "cuda"
                    )
                    if (
                        not valid_device
                        or chw.dtype != torch.uint8
                        or tuple(chw.shape) != (3, *self.original_hw)
                    ):
                        raise RuntimeError(
                            "device JPEG decoder returned invalid pixels"
                        )
                    stage = "resize"
                    step = time.monotonic_ns()
                    chw = _resize_device_rgb(chw, self.target_hw)
                    record["resize_host_ms"] = (time.monotonic_ns() - step) / 1e6
                    record["decode_backend"] = "device"
                except torch.OutOfMemoryError:
                    # An exhausted device must follow the scheduler's normal
                    # error path; CPU fallback would still need device memory.
                    raise
                except (ImportError, OSError, RuntimeError) as error:
                    record["fallback_reason"] = type(error).__name__
                    record["fallback_stage"] = stage
                    chw = None
            else:
                record["fallback_reason"] = "non_cuda_device"
            if chw is None:
                step = time.monotonic_ns()
                chw = self._cpu_pixels()
                record["cpu_decode_resize_ms"] = (time.monotonic_ns() - step) / 1e6
            step = time.monotonic_ns()
            payload = QwenImagePixels(
                chw.permute(1, 2, 0),
                self.lookup,
                self.patch_size,
                self.merge_size,
                self.temporal_patch_size,
            )
            result = payload.to_patches(device, dtype)
            record["pack_host_ms"] = (time.monotonic_ns() - step) / 1e6
            return result
        except BaseException as error:
            record.update(outcome="error", error=type(error).__name__)
            raise
        finally:
            # Host boundaries deliberately do not synchronize every image.
            # These values are not accelerator kernel execution durations.
            record["total_host_ms"] = (time.monotonic_ns() - started) / 1e6
            logging.info(
                "vit_image_device_preprocess_timing %s",
                json.dumps(record, separators=(",", ":")),
            )


def prepare_device_jpeg(image, encoded, processor, target_hw, processor_resize, timing):
    """Plan CPU-only transport, retaining request sizing and the service cap."""
    if os.getenv("RTP_LLM_QWEN_IMAGE_GPU_DECODE", "0") != "1":
        return None
    if image.format != "JPEG" or image.mode != "RGB":
        return None
    # Reduced IDCT has different pixels. Preserve its existing opt-in CPU path.
    if os.getenv("RTP_LLM_QWEN_IMAGE_JPEG_DRAFT", "0") == "1":
        return None
    maximum = int(os.getenv("RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS", "33554432"))
    if maximum <= 0:
        raise ValueError("RTP_LLM_QWEN_IMAGE_GPU_DECODE_MAX_PIXELS must be positive")
    if image.height * image.width > maximum:
        return None
    if target_hw is None:
        if getattr(processor, "do_resize", False):
            return None
        target_hw = image.height, image.width
    if getattr(processor, "resample", None) != Image.Resampling.BICUBIC:
        return None
    parameters = device_pixel_parameters(processor)
    if parameters is None:
        return None
    _, patch, merge, _ = parameters
    height, width = target_hw
    if min(height, width) <= 0 or height % (patch * merge) or width % (patch * merge):
        return None
    payload = QwenJpegBytes(
        encoded,
        *parameters,
        (image.height, image.width),
        (height, width),
        processor_resize and getattr(processor, "backend", None) == "torchvision",
        timing.get("request_id", 0),
        timing.get("image_id", ""),
    )
    return payload, torch.tensor([[1, height // patch, width // patch]], device="cpu")
