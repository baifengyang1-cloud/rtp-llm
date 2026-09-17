import copy
from typing import Any, List

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs, RendererParams
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.ops import MMPreprocessConfig
from rtp_llm.utils.base_model_datatypes import MMUrlType


class QwenVLRenderer(QwenRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )

    def extract_multimodal_inputs(self, messages):
        urls = []
        for message in messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.type == ContentPartTypeEnum.text:
                        continue
                    if (
                        part.type != ContentPartTypeEnum.image_url
                        or part.image_url is None
                    ):
                        raise ValueError("Qwen-VL supports image_url media only")
                    urls.append(part.image_url.url)
        return RenderedInputs([], input_urls=urls).multimodal_inputs

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        prompt = ""
        images = [item.url for item in self.extract_multimodal_inputs(messages)]
        image_index = 0
        if messages[0].role != RoleEnum.system:
            messages = [
                ChatMessage(
                    role=RoleEnum.system, content="You are a helpful assistant."
                )
            ] + messages

        for message in messages:
            if isinstance(message.content, str):
                prompt += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
            elif isinstance(message.content, list):
                prompt += f"<|im_start|>{message.role}\n"
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        url = content_part.image_url.url
                        image_index += 1
                        prompt += f"Picture {image_index}: <img>{url}</img>\n"
                prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return PromptWithMMInput(prompt=prompt, urls=images)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
        )


class Qwen2VLRenderer(QwenRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )

    def extract_multimodal_inputs(self, messages):
        urls, types, configs = [], [], []
        for message in messages:
            if not isinstance(message.content, list):
                continue
            for part in message.content:
                if part.type == ContentPartTypeEnum.text:
                    continue
                if part.type not in (
                    ContentPartTypeEnum.image_url,
                    ContentPartTypeEnum.video_url,
                ):
                    raise ValueError(f"Unsupported Qwen-VL media type: {part.type}")
                media = (
                    part.image_url
                    if part.type == ContentPartTypeEnum.image_url
                    else part.video_url
                )
                if media is None or not media.url.strip():
                    raise ValueError("media URL is required")
                urls.append(media.url)
                types.append(
                    MMUrlType.IMAGE
                    if part.type == ContentPartTypeEnum.image_url
                    else MMUrlType.VIDEO
                )
                configs.append(
                    get_preprocess_config(part.preprocess_config)
                    if part.preprocess_config
                    else MMPreprocessConfig()
                )
        return RenderedInputs(
            [], input_urls=urls, input_urls_type=types, preprocess_configs=configs
        ).multimodal_inputs

    def _format_tool_call_arguments(self, arguments: Any) -> Any:
        return arguments

    def _render_messages(
        self, request: ChatCompletionRequest, add_vision_id: bool
    ) -> PromptWithMMInput:
        media = self.extract_multimodal_inputs(request.messages)
        urls = [item.url for item in media]
        types = [item.mm_type for item in media]
        preprocess_configs = [item.mm_preprocess_config for item in media]
        final_messages = []
        for message in request.messages:
            msg_dict = {"role": message.role.value}

            if isinstance(message.content, list):
                now_content = []
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_content.append({"type": "text", "text": content_part.text})
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        now_content.append(
                            {"type": "image", "image": content_part.image_url.url}
                        )
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        now_content.append(
                            {"type": "video", "video": content_part.video_url.url}
                        )
                msg_dict["content"] = now_content
            else:
                msg_dict["content"] = message.content

            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "type": "function",
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": self._format_tool_call_arguments(
                                tc.function.arguments
                            ),
                        },
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            final_messages.append(msg_dict)

        final_tools = []
        if request.tools:
            for tool in request.tools:
                final_tools.append(
                    {
                        "type": tool.type,
                        "function": tool.function.model_dump(
                            exclude_none=True, mode="json"
                        ),
                    }
                )

        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "add_vision_id": add_vision_id,
            "tools": final_tools,
        }
        request_chat_template_kwargs = request.get_chat_template_kwargs()
        if request_chat_template_kwargs is not None:
            chat_template_kwargs.update(request_chat_template_kwargs)
        prompt = self.tokenizer.apply_chat_template(
            final_messages, **chat_template_kwargs
        )

        return PromptWithMMInput(
            prompt=prompt,
            urls=urls,
            mm_types=types,
            preprocess_configs=preprocess_configs,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(
            request,
            request.extra_configs.add_vision_id if request.extra_configs else True,
        )
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("qwen_vl", QwenVLRenderer)
register_renderer("qwen_vl_1b8", QwenVLRenderer)
register_renderer("qwen2_vl", Qwen2VLRenderer)
register_renderer("qwen2_5_vl", Qwen2VLRenderer)
register_renderer("qwen3_vl", Qwen2VLRenderer)
register_renderer("qwen3_vl_moe", Qwen2VLRenderer)
