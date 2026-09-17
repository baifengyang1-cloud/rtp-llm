"""The pretrigger HTTP contract, separate from token generation validation."""

import asyncio
from typing import Annotated, Optional

import grpc
from pydantic import BaseModel, Field

from rtp_llm.config.exceptions import (
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import serialize_multimodal_inputs
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.structure.multimodal_request import (
    media_config,
    prepare_chat_messages,
    prepare_native_media,
)
from rtp_llm.structure.request_extractor import RequestExtractor


def resolve_pretrigger_config(body, chat=False):
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    if chat:
        config = body.get("extra_configs")
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("extra_configs must be an object")
    else:
        config, _, _ = RequestExtractor.resolve_generate_config(body)
    GenerateConfig.validate_pretrigger_scheme(
        config.get("pretrigger_scheme", "disable")
    )
    return config


class EncoderResponseOptions(BaseModel):
    """Validate response controls without invoking token-generation validators."""

    n: Optional[Annotated[int, Field(gt=0)]] = None
    num_beams: Annotated[int, Field(gt=0)] = 1
    variable_num_beams: list[Annotated[int, Field(gt=0)]] = []
    num_return_sequences: Annotated[int, Field(ge=0)] = 0
    aux_info: bool = True
    stream: Optional[bool] = False
    yield_generator: bool = False
    return_incremental: bool = False
    is_streaming: bool = False
    user_template: Optional[str] = None
    model: Optional[str] = None


def prepare_encoder_response(body, chat, config):
    """Return validated HTTP options and a terminal response generator factory."""
    from rtp_llm.frontend.frontend_worker import (
        BatchPipelineResponse,
        MultiSequencesPipelineResponse,
        PipelineResponse,
    )
    from rtp_llm.openai.api_datatype import (
        ChatCompletionResponseStreamChoice,
        DeltaMessage,
        UsageInfo,
    )
    from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
    from rtp_llm.openai.renderers.custom_renderer import (
        StreamResponseObject,
        response_choice_count,
    )
    from rtp_llm.utils.complete_response_async_generator import (
        CompleteResponseAsyncGenerator,
    )

    values = {
        key: value
        for key, value in config.items()
        if key in EncoderResponseOptions.model_fields
    }
    values["stream"] = body.get("stream", False)
    if chat:
        for key in ("n", "aux_info", "user_template", "model"):
            if key in body:
                values[key] = body[key]
        # Chat HTTP transport is controlled only by its top-level stream flag.
        values["yield_generator"] = False
    else:
        values["yield_generator"] = RequestExtractor.is_streaming(body)
    options = EncoderResponseOptions.model_validate(values)
    request = dict(body, stream=options.stream)
    if chat:
        # These raw-only fields are discarded by normal ChatCompletionRequest
        # parsing. Do not let them select SSE on the relaxed encoder path.
        for key in ("yield_generator", "generate_config", "generation_config"):
            request.pop(key, None)
    else:
        request["yield_generator"] = options.yield_generator
        if options.return_incremental and not (
            options.stream or options.yield_generator or options.is_streaming
        ):
            raise ValueError("request is non_stream but use incremental decoder")

    if chat:
        count = response_choice_count(options.n, options)

        async def chat_terminal():
            yield StreamResponseObject(
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(role="assistant", content=""),
                        finish_reason="stop",
                    )
                    for i in range(count)
                ],
                usage=UsageInfo(prompt_tokens=0, total_tokens=0, completion_tokens=0),
            )

        return request, lambda: OpenaiEndpoint._complete_stream_response(
            chat_terminal(), None
        )

    batch, input_count = native_response_shape(body)
    count = options.num_return_sequences

    def empty_response():
        if count == 0:
            return PipelineResponse(response="", finished=True, aux_info={})
        return MultiSequencesPipelineResponse(
            response=[""] * count,
            finished=True,
            aux_info=[{} for _ in range(count)] if options.aux_info else [],
        )

    response = (
        BatchPipelineResponse(
            response_batch=[empty_response() for _ in range(input_count)]
        )
        if batch
        else empty_response()
    )

    async def raw_terminal():
        yield response

    async def collect(_):
        # The incremental raw collector cannot collect an empty aux_info dict.
        return response

    return request, lambda: CompleteResponseAsyncGenerator(raw_terminal(), collect)


def native_response_shape(body):
    if "prompt_batch" in body:
        if not isinstance(body["prompt_batch"], list):
            raise ValueError("prompt batch input should be list")
        return True, len(body["prompt_batch"])
    if not any(key in body for key in ("prompt", "text", "messages")):
        urls = body.get("images", body.get("urls", []))
        if isinstance(urls, list) and urls and isinstance(urls[0], list):
            return True, len(urls)
    return False, 1


def prepare_pretrigger(body, chat, endpoint, request_id, resolved_config=None):
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    if resolved_config is None:
        resolved_config = resolve_pretrigger_config(body, chat)
    if chat:
        messages = prepare_chat_messages(body)
        config = GenerateConfig.model_validate(media_config(resolved_config))
        if endpoint is None:
            if any(
                isinstance(m.content, list)
                and any(p.type.value != "text" for p in m.content)
                for m in messages
            ):
                raise NotImplementedError("chat media extraction is unavailable")
            groups = []
        else:
            renderer = (
                endpoint.template_renderer
                if body.get("user_template")
                else endpoint.chat_renderer
            )
            groups = [(renderer.extract_multimodal_inputs(messages), config)]
    else:
        groups = prepare_native_media(body, resolved_config)
    inputs = MultimodalInputsPB(request_id=request_id)
    for media, config in groups:
        inputs.multimodal_inputs.extend(
            serialize_multimodal_inputs(media, config).multimodal_inputs
        )
    return inputs


def pretrigger_error_status(error):
    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return 504
    if isinstance(error, grpc.RpcError):
        return {
            grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
            grpc.StatusCode.DEADLINE_EXCEEDED: 504,
            grpc.StatusCode.INVALID_ARGUMENT: 400,
            grpc.StatusCode.PERMISSION_DENIED: 403,
            grpc.StatusCode.UNAVAILABLE: 503,
            grpc.StatusCode.UNIMPLEMENTED: 503,
        }.get(error.code(), 503)
    if isinstance(error, FtRuntimeException):
        from rtp_llm.server.master_client import VitRoutingPolicyError

        if isinstance(error, VitRoutingPolicyError):
            return 403
        if error.exception_type == ExceptionType.UNSAFE_INPUT_CONTENT:
            return 403
        if error.exception_type == ExceptionType.MASTER_NO_VIT_WORKER:
            return 503
        return {
            ExceptionCategory.CAPACITY: 429,
            ExceptionCategory.TIMEOUT: 504,
            ExceptionCategory.BAD_REQUEST: 400,
            ExceptionCategory.TOO_LONG: 400,
            ExceptionCategory.UNSUPPORTED: 400,
        }.get(error.exception_type.category, 503)
    if isinstance(error, NotImplementedError):
        return 400
    if isinstance(error, (ValueError, TypeError)):
        return 400
    return 500
