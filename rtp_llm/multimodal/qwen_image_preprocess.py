"""Bound Qwen image processing by its final model grid before float conversion."""

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def _processor_pixel_limits(processor):
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
    if min(minimum, maximum) <= 0 or minimum > maximum:
        return None
    return minimum, maximum


def service_image_max_pixels(factor):
    """Zero preserves the model policy; a positive value caps every image."""
    name = "RTP_LLM_QWEN_IMAGE_MAX_PIXELS"
    raw = os.getenv(name, "0")
    try:
        maximum = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if maximum != 0 and maximum < factor * factor:
        raise ValueError(f"{name} must be 0 or at least {factor * factor}")
    return maximum


def limit_image_size(image, processor, factor, target_hw):
    """Apply the service cap after resolving model/request sizing policy."""
    maximum = service_image_max_pixels(factor)
    if maximum == 0:
        return target_hw
    minimum = 0
    if target_hw is None:
        # Also cover non-RGB images and the legacy CPU processor path. Keep a
        # stricter model budget and its small-image policy when available.
        target_hw = image.height, image.width
        limits = _processor_pixel_limits(processor)
        if getattr(processor, "do_resize", False) and limits is not None:
            minimum, model_maximum = limits
            maximum = min(maximum, model_maximum)
            minimum = min(minimum, maximum)
    elif target_hw[0] * target_hw[1] <= maximum:
        return target_hw
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    height, width = smart_resize(
        *target_hw, factor=factor, min_pixels=minimum, max_pixels=maximum
    )
    if height * width > maximum:
        # A model minimum conflicting with the service cap can round upward.
        height, width = smart_resize(
            height, width, factor=factor, min_pixels=0, max_pixels=maximum
        )
    if height * width > maximum:
        # Extremely small caps may not fit the aspect ratio on the patch grid.
        if height >= width:
            height = maximum // (width * factor) * factor
        else:
            width = maximum // (height * factor) * factor
    return height, width


def default_resize_size(image, processor, factor):
    """Resolve the existing HF policy, without imposing a new pixel budget."""
    if os.getenv("RTP_LLM_QWEN_IMAGE_EARLY_RESIZE", "1") != "1":
        return None
    if image.mode != "RGB" or not getattr(processor, "do_resize", False):
        return None
    if getattr(processor, "resample", None) != Image.Resampling.BICUBIC:
        return None
    limits = _processor_pixel_limits(processor)
    if limits is None:
        return None
    minimum, maximum = limits
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


def device_pixel_parameters(processor):
    """Resolve the shared normalization and patch contract without decoding."""
    patch = getattr(processor, "patch_size", None)
    merge = getattr(processor, "merge_size", None)
    temporal = getattr(processor, "temporal_patch_size", None)
    if any(type(x) is not int or x <= 0 for x in (patch, merge, temporal)):
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
    return (
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


def prepare_device_pixels(image, processor):
    """Return a deferred payload only for the supported RGB Qwen contract."""
    if image.mode != "RGB":
        return None
    parameters = device_pixel_parameters(processor)
    if parameters is None:
        return None
    _, patch, merge, _ = parameters
    if image.height % (patch * merge) or image.width % (patch * merge):
        return None
    pixels = QwenImagePixels(torch.from_numpy(np.array(image)), *parameters)
    grid = torch.tensor(
        [[1, image.height // patch, image.width // patch]], device="cpu"
    )
    return pixels, grid


def vision_pixel_values(data, device, dtype):
    # Imported lazily so CPU workers never import a device decoder extension.
    from rtp_llm.multimodal.qwen_image_decode import QwenJpegBytes, QwenPngPixels

    if isinstance(data, (QwenJpegBytes, QwenPngPixels)):
        return data.to_patches(device, dtype)
    if isinstance(data, QwenImagePixels):
        return data.to_patches(device, dtype)
    return data.to(device=device, dtype=dtype)
