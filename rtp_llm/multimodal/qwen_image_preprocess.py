"""Bound Qwen image processing by its final model grid before float conversion."""

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def default_resize_size(image, processor, factor):
    """Resolve the existing HF policy, without imposing a new pixel budget."""
    if os.getenv("RTP_LLM_QWEN_IMAGE_EARLY_RESIZE", "1") != "1":
        return None
    if image.mode != "RGB" or not getattr(processor, "do_resize", False):
        return None
    if getattr(processor, "resample", None) != Image.Resampling.BICUBIC:
        return None
    minimum = getattr(processor, "min_pixels", None)
    maximum = getattr(processor, "max_pixels", None)
    if type(minimum) is not int or type(maximum) is not int:
        size = getattr(processor, "size", None)
        if isinstance(size, dict):
            minimum, maximum = size.get("shortest_edge"), size.get("longest_edge")
        else:
            minimum = getattr(size, "shortest_edge", None)
            maximum = getattr(size, "longest_edge", None)
    if type(minimum) is not int or type(maximum) is not int:
        return None
    if min(minimum, maximum) <= 0:
        return None
    patch = getattr(processor, "patch_size", None)
    merge = getattr(processor, "merge_size", None)
    if type(patch) is not int or type(merge) is not int or patch * merge != factor:
        return None
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    height, width = smart_resize(
        image.height,
        image.width,
        factor=factor,
        min_pixels=minimum,
        max_pixels=maximum,
    )
    return height, width


def resize_qwen_image(image, processor, target_hw, processor_resize=False):
    """Preserve the processor's resize backend for its default pixel policy."""
    height, width = target_hw
    if processor_resize and getattr(processor, "backend", None) == "torchvision":
        from transformers.image_processing_backends import TorchvisionBackend
        from transformers.image_utils import SizeDict

        pixels = processor.process_image(image, do_convert_rgb=True, device="cpu")
        pixels = TorchvisionBackend.resize(
            processor,
            image=pixels.unsqueeze(0).contiguous(),
            size=SizeDict(height=height, width=width),
            resample=processor.resample,
        )
        return Image.fromarray(pixels[0].permute(1, 2, 0).contiguous().numpy())
    return image.resize((width, height), Image.Resampling.BICUBIC)


def maybe_draft_jpeg(image, target_hw):
    """Optional reduced JPEG IDCT; retain the target planned from original size."""
    if target_hw is None or image.format != "JPEG" or image.mode != "RGB":
        return False
    if os.getenv("RTP_LLM_QWEN_IMAGE_JPEG_DRAFT", "0") != "1":
        return False
    height, width = target_hw
    if image.width * image.height <= max(8_000_000, height * width * 4):
        return False
    original = image.size
    image.draft("RGB", (width, height))
    return image.size != original


@lru_cache(maxsize=16)
def _normalization_table(rescale, scale, normalize, mean, std, normalizer=None):
    # Reuse the exact HF arithmetic for every possible uint8 channel value.
    # Gathering this table on the device avoids changing float rounding at the
    # FP16/BF16 vision boundary, and avoids full-resolution CPU float buffers.
    if normalizer is not None:
        values = (
            torch.arange(256, dtype=torch.uint8, device="cpu")
            .view(1, 1, 1, 256)
            .expand(1, 3, 1, 256)
            .contiguous()
        )
        return normalizer(values, rescale, scale, normalize, mean, std).reshape(3, 256)

    from transformers.image_transforms import normalize as hf_normalize
    from transformers.image_transforms import rescale as hf_rescale
    from transformers.image_utils import ChannelDimension

    values = np.broadcast_to(np.arange(256, dtype=np.uint8), (3, 1, 256)).copy()
    if rescale:
        values = hf_rescale(values, scale, input_data_format=ChannelDimension.FIRST)
    if normalize:
        values = hf_normalize(
            values, mean, std, input_data_format=ChannelDimension.FIRST
        )
    return torch.from_numpy(np.ascontiguousarray(values[:, 0, :], dtype=np.float32))


@dataclass
class QwenImagePixels:
    """CPU-only, pickleable payload; device work starts in the ViT scheduler."""

    pixels: torch.Tensor
    lookup: torch.Tensor
    patch_size: int
    merge_size: int
    temporal_patch_size: int

    def to_patches(self, device, dtype):
        with torch.profiler.record_function("qwen_image_device_preprocess"):
            height, width = self.pixels.shape[:2]
            patch, merge = self.patch_size, self.merge_size
            gh, gw = height // patch, width // patch
            indices = self.pixels.to(device=device).permute(2, 0, 1).reshape(3, -1)
            values = (
                self.lookup.to(device=device, dtype=dtype)
                .gather(1, indices.long())
                .reshape(3, height, width)
            )
            patches = values.reshape(
                3, gh // merge, merge, patch, gw // merge, merge, patch
            ).permute(1, 4, 2, 5, 0, 3, 6)
            patches = patches.unsqueeze(-3).expand(
                -1, -1, -1, -1, -1, self.temporal_patch_size, -1, -1
            )
            return patches.reshape(gh * gw, -1)


def prepare_device_pixels(image, processor):
    """Return a deferred payload only for the supported RGB Qwen contract."""
    if image.mode != "RGB":
        return None
    patch = getattr(processor, "patch_size", None)
    merge = getattr(processor, "merge_size", None)
    temporal = getattr(processor, "temporal_patch_size", None)
    if any(type(x) is not int or x <= 0 for x in (patch, merge, temporal)):
        return None
    if image.height % (patch * merge) or image.width % (patch * merge):
        return None
    rescale = getattr(processor, "do_rescale", None)
    normalize = getattr(processor, "do_normalize", None)
    scale = getattr(processor, "rescale_factor", None)
    mean = getattr(processor, "image_mean", None)
    std = getattr(processor, "image_std", None)
    if type(rescale) is not bool or type(normalize) is not bool:
        return None
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        return None
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        return None
    if len(mean) != 3 or len(std) != 3:
        return None
    if not all(math.isfinite(x) for x in (*mean, *std)) or min(std) <= 0:
        return None
    pixels = QwenImagePixels(
        torch.from_numpy(np.array(image)),
        _normalization_table(
            rescale,
            scale,
            normalize,
            tuple(mean),
            tuple(std),
            (
                processor.rescale_and_normalize
                if getattr(processor, "backend", None) == "torchvision"
                else None
            ),
        ),
        patch,
        merge,
        temporal,
    )
    grid = torch.tensor(
        [[1, image.height // patch, image.width // patch]], device="cpu"
    )
    return pixels, grid


def vision_pixel_values(data, device, dtype):
    if isinstance(data, QwenImagePixels):
        return data.to_patches(device, dtype)
    return data.to(device=device, dtype=dtype)
