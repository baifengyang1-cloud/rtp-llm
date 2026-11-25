"""Synthetic Qwen35 inputs: no downloads and no embedding/cache admission."""

import json
import logging
import os
from io import BytesIO

from PIL import Image

from rtp_llm.multimodal.qwen_image_decode import prepare_device_jpeg
from rtp_llm.multimodal.qwen_image_preprocess import (
    _processor_pixel_limits,
    prepare_device_pixels,
    service_image_max_pixels,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType

# Observed hot image shapes, plus common multi-image forward compositions.
# Other shapes still work normally but can incur their own first-use cost.
_DEFAULT_BATCHES = [
    [[size, size]] for size in (512, 800, 1408, 992, 736, 768, 864, 704, 448)
] + [
    [[512, 384]],
    [[352, 320]],
    [[224, 448]],
    [[480, 704]],
    [[512, 512]] * 2,
    [[512, 512]] * 3,
    [[800, 800]] * 2,
    [[512, 384]] * 2,
]


def warmup_shapes(factor, max_batch_images, max_pixels):
    batches = json.loads(os.getenv("RTP_LLM_VIT_WARMUP_BATCHES", "null"))
    batches = _DEFAULT_BATCHES if batches is None else batches
    if not isinstance(batches, list) or not 1 <= len(batches) <= 64:
        raise ValueError("RTP_LLM_VIT_WARMUP_BATCHES must contain 1 to 64 batches")
    for batch in batches:
        if not isinstance(batch, list) or not 1 <= len(batch) <= 32:
            raise ValueError("Each ViT warmup batch must contain 1 to 32 images")
        for shape in batch:
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(n) is not int or n <= 0 or n % factor for n in shape)
            ):
                raise ValueError(
                    "ViT warmup H/W must be positive multiples of the patch/merge factor"
                )
        if len(batch) > max_batch_images or any(h * w > max_pixels for h, w in batch):
            logging.info(
                "vit_warmup_skipped reason=media_or_pixel_budget shapes=%s", batch
            )
            continue
        yield batch


def qwen_warmup_batches(mm_part, max_batch_images):
    processor = mm_part.mm_processor.image_processor
    factor = mm_part.visual.patch_size * mm_part.visual.spatial_merge_size
    limits = _processor_pixel_limits(processor)
    max_pixels = limits[1] if limits is not None else 1408**2
    service_max = service_image_max_pixels(factor)
    if service_max:
        max_pixels = min(max_pixels, service_max)
    budget = mm_part.get_batch_work_budget(max_batch_images)
    for shapes in warmup_shapes(factor, max_batch_images, max_pixels):
        data = []
        estimate = None
        for h, w in shapes:
            image = Image.new("RGB", (w, h), (127, 63, 191))
            payload = None
            if os.getenv("RTP_LLM_QWEN_IMAGE_GPU_DECODE", "0") == "1":
                encoded = BytesIO()
                # A slightly larger source also warms device resizing.
                Image.new("RGB", (w + factor, h + factor), (127, 63, 191)).save(
                    encoded, format="JPEG"
                )
                encoded = encoded.getvalue()
                with Image.open(BytesIO(encoded)) as jpeg:
                    payload = prepare_device_jpeg(
                        jpeg, encoded, processor, (h, w), True, {}
                    )
            if (
                payload is None
                and os.getenv("RTP_LLM_QWEN35_GPU_PREPROCESS", "0") == "1"
            ):
                payload = prepare_device_pixels(image, processor)
            if payload is None:
                result = processor(images=[image], do_resize=False, return_tensors="pt")
                payload = result["pixel_values"], result["image_grid_thw"]
            item_estimate = mm_part.estimate_work(payload, MMUrlType.IMAGE)
            if item_estimate is not None:
                estimate = (
                    item_estimate if estimate is None else estimate + item_estimate
                )
            data.append(payload)
        if (
            budget is not None
            and estimate is not None
            and not estimate.fits_within(budget)
        ):
            logging.info(
                "vit_warmup_skipped reason=model_work_budget shapes=%s", shapes
            )
            continue
        yield shapes, data
