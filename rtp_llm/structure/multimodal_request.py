"""Media preparation shared with inference, without prompt or generation work."""

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import ChatMessage, ContentPartTypeEnum
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.structure.request_extractor import RequestExtractor
from rtp_llm.utils.base_model_datatypes import MMUrlType

MEDIA_CONFIG_FIELDS = frozenset(
    (
        "resized_shape",
        "min_pixels",
        "max_pixels",
        "fps",
        "min_frames",
        "max_frames",
        "crop_positions",
        "mm_timeout_ms",
        "max_long_side_pixel",
    )
)


def media_config(values):
    if not isinstance(values, dict):
        raise ValueError("media configuration must be an object")
    return {key: value for key, value in values.items() if key in MEDIA_CONFIG_FIELDS}


def native_multimodal_inputs(urls):
    if not isinstance(urls, list) or any(
        not isinstance(url, str) or not url.strip() for url in urls
    ):
        raise ValueError("media urls must be a list of non-empty strings")
    return [
        MultimodalInput(url, MMUrlType.DEFAULT, torch.empty(0), MMPreprocessConfig())
        for url in urls
    ]


def prepare_native_media(body, resolved_config=None):
    """Keep native batching and nested/top-level option precedence."""
    if resolved_config is None:
        resolved_config, _, _ = RequestExtractor.resolve_generate_config(body)
    config = media_config(resolved_config)
    config = GenerateConfig.model_validate(config)
    urls = body.get("images", body.get("urls", []))
    if not isinstance(urls, list):
        raise ValueError("images/urls must be a list")
    batched = bool(urls) and isinstance(urls[0], list)
    groups = urls if batched else [urls]
    prompt = body.get("prompt_batch", body.get("prompt", body.get("text")))
    if isinstance(prompt, list) and (
        "prompt_batch" in body or not prompt or isinstance(prompt[0], str)
    ):
        if urls and (
            not batched and len(prompt) != 1 or batched and len(groups) != len(prompt)
        ):
            raise ValueError("media groups and prompt batch must have the same length")
    return [
        (native_multimodal_inputs(group), config.model_copy(deep=True))
        for group in groups
    ]


def prepare_chat_messages(body):
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    # Tool choices/history are irrelevant to media. Validate roles/content with
    # the normal endpoint's models without inheriting generation validation.
    result = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        parsed = ChatMessage.model_validate(
            {key: message[key] for key in ("role", "content") if key in message}
        )
        if isinstance(parsed.content, list):
            for part in parsed.content:
                if part.type == ContentPartTypeEnum.text:
                    continue
                field = part.type.value
                value = getattr(part, field, None)
                if value is None or (field.endswith("_url") and not value.url.strip()):
                    raise ValueError(f"{field} requires a non-empty media value")
        result.append(parsed)
    return result
