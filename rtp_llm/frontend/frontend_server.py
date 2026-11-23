import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Union

from fastapi import Request
from fastapi import Request as RawRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from rtp_llm.utils import vit_diagnostics as diag
from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import (
    update_stop_words_from_env,
    update_tokenizer_special_tokens,
)
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.pretrigger import (
    prepare_encoder_response,
    prepare_pretrigger,
    pretrigger_error_status,
    resolve_pretrigger_config,
)
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import SpecialTokens, TaskType
from rtp_llm.server.misc import format_exception
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.telemetry import CURRENT_TRACE_STATE
from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.telemetry import record_response_attributes, start_server_span
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import check_with_info

USAGE_HEADER = "USAGE"


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _has_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, dict):
        return any(_has_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_payload(item) for item in value)
    if hasattr(value, "model_dump"):
        try:
            return _has_payload(value.model_dump(exclude_none=True))
        except Exception:
            return False
    return bool(value)


def _visible_output_count(response: Any) -> int:
    """Counts visible payload lanes in one streaming response object."""
    choices = _field(response, "choices")
    if isinstance(choices, (list, tuple)):
        count = 0
        for choice in choices:
            delta = _field(choice, "delta")
            if delta is None:
                continue
            if any(
                _has_payload(_field(delta, name))
                for name in (
                    "content",
                    "reasoning_content",
                    "function_call",
                    "tool_calls",
                )
            ):
                count += 1
        return count

    response_batch = _field(response, "response_batch")
    if isinstance(response_batch, (list, tuple)):
        return sum(_visible_output_count(item) for item in response_batch)

    payload = _field(response, "response")
    if isinstance(payload, (list, tuple)):
        return sum(1 for item in payload if _has_payload(item))
    return 1 if _has_payload(payload) else 0


def _output_token_total(response: Any) -> int | None:
    """Returns a cumulative output-token count when the response exposes one."""
    usage = _field(response, "usage")
    completion_tokens = _field(usage, "completion_tokens")
    if (
        isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and completion_tokens >= 0
    ):
        return completion_tokens

    response_batch = _field(response, "response_batch")
    if isinstance(response_batch, (list, tuple)):
        totals = [_output_token_total(item) for item in response_batch]
        if totals and all(total is not None for total in totals):
            return sum(total for total in totals if total is not None)

    aux_info = _field(response, "aux_info")
    aux_items = aux_info if isinstance(aux_info, (list, tuple)) else [aux_info]
    totals = []
    for item in aux_items:
        output_len = _field(item, "output_len")
        if (
            not isinstance(output_len, int)
            or isinstance(output_len, bool)
            or output_len < 0
        ):
            return None
        totals.append(output_len)
    return sum(totals) if totals else None


def _record_http_status(trace_state, status_code: int) -> None:
    # Dual-write both semconv generations: platform views disagree on which key
    # wins, and the HTTP-error counter only reads the legacy http.status_code.
    trace_state.set_attribute(trace_attrs.HTTP_RESPONSE_STATUS_CODE, status_code)
    trace_state.set_attribute(trace_attrs.HTTP_STATUS_CODE, status_code)


def _chat_admission_event(raw_request, request_id, worker, controller):
    from rtp_llm.server.request_headers import (
        extract_correlation_request_id,
        extract_trace_id,
    )

    scope = getattr(raw_request, "scope", {})
    state = scope.setdefault("state", {})
    key = request_id or state.get("vit_trace_id") or -time.time_ns()
    state["vit_trace_id"] = key
    if diag.selected(key):
        diag.event(
            key,
            "frontend.admitted" if request_id else "frontend.rejected",
            worker=worker,
            capacity=controller.max_concurrency,
            available=controller.get_available_concurrency(),
            attempt_id=-state["vit_arrival_ns"] if "vit_arrival_ns" in state else None,
            arrival_wall_ns=state.get("vit_arrival_ns"),
            arrival_mono_ns=state.get("vit_arrival_mono_ns"),
            upstream_id=extract_correlation_request_id(raw_request.headers)[:128],
            upstream_trace=extract_trace_id(raw_request.headers)[:128],
        )


class FrontendServer(object):
    def __init__(
        self,
        rank_id: int,
        server_id: int,
        py_env_configs=None,
    ):
        self.py_env_configs = py_env_configs
        self._access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
            rank_id,
            server_id,
        )
        self._frontend_worker = None
        self._openai_endpoint = None
        self._embedding_endpoint = None
        self.is_embedding = False
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        self.rank_id = str(rank_id)
        self.server_id = str(server_id)
        kmonitor.init()

    def start(self):
        if self.py_env_configs.vit_config.pretrigger_timeout_ms <= 0:
            raise ValueError("pretrigger_timeout_ms must be positive")
        logging.info(
            "Pretrigger submission deadline_ms=%s vit_separation=%s vit_server_count=%s",
            self.py_env_configs.vit_config.pretrigger_timeout_ms,
            self.py_env_configs.vit_config.vit_separation,
            self.py_env_configs.server_config.vit_server_count,
        )
        if (
            self.py_env_configs.profiling_debug_logging_config.debug_start_fake_process
            == 1
        ):
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self._frontend_worker = None
            return

        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=self.py_env_configs.kv_cache_config,
            profiling_debug_logging_config=self.py_env_configs.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
            vit_config=self.py_env_configs.vit_config,
        )

        # Create a temporary tokenizer to initialize special_tokens
        # We'll update it with the actual tokenizer after FrontendWorker is created
        special_tokens = SpecialTokens()
        if self.py_env_configs.generate_env_config:
            update_stop_words_from_env(
                special_tokens, self.py_env_configs.generate_env_config
            )

        # Create FrontendWorker with special_tokens and config
        self._frontend_worker = FrontendWorker(
            self.py_env_configs,
            model_config,
            special_tokens,
        )

        # Update special_tokens with actual tokenizer
        update_tokenizer_special_tokens(special_tokens, self._frontend_worker.tokenizer)

        # Only initialize OpenaiEndpoint for LANGUAGE_MODEL task type
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            # Update model_config with the latest values
            model_config.special_tokens = special_tokens
            model_config.generate_env_config = self.py_env_configs.generate_env_config
            model_config.render_config = self.py_env_configs.render_config
            model_config.model_name = self.py_env_configs.model_args.model_type
            model_config.template_type = None

            self._openai_endpoint = OpenaiEndpoint(
                model_config=model_config,
                misc_config=self.py_env_configs.misc_config,
                vit_config=self.py_env_configs.vit_config,
                tokenizer=self._frontend_worker.tokenizer,
                backend_rpc_server_visitor=self._frontend_worker.backend_rpc_server_visitor,
            )
        else:
            from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint

            self._embedding_endpoint = EmbeddingEndpoint(
                model_config=model_config,
                grpc_config=self.py_env_configs.grpc_config,
                server_config=self.py_env_configs.server_config,
                tokenizer=self._frontend_worker.tokenizer,
            )
            self.is_embedding = True

    async def close(self):
        try:
            if self._frontend_worker is not None:
                await self._frontend_worker.close()
        finally:
            if self._embedding_endpoint is not None:
                await self._embedding_endpoint.close()

    def stop(self):
        if self._frontend_worker is not None:
            self._frontend_worker.stop()

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        start_time = time.time()
        try:
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(
                AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")}
            )
            sequence = self._global_controller.increment() % 4096  # 12 bits
            request[request_id_field_name] = generate_request_id(
                self.py_env_configs.server_config.ip,
                self.py_env_configs.server_config.server_port,
                self.server_id,
                sequence,
            )
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            assert (
                self._embedding_endpoint is not None
            ), "embedding pipeline should not be None"
            result, logable_result = await self._embedding_endpoint.embedding(request)
            # do not log result since too big
            if logable_result is not None:
                self._access_logger.log_success_access(request, logable_result)
            end_time = time.time()
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000
            )
            kmonitor.report(
                AccMetrics.SUCCESS_QPS_METRIC,
                1,
                {"source": request.get("source", "unknown")},
            )
            usage = result.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            return ORJSONResponse(result, headers={USAGE_HEADER: json.dumps(usage)})
        except BaseException as e:
            return self._handle_exception(request, e)
        finally:
            self._global_controller.decrement()

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146
    async def stream_response(
        self,
        request: Dict[str, Any],
        response: CompleteResponseAsyncGenerator,
    ):
        # HTTP SERVER span owner for streaming requests: the four exits below
        # (success / cancel / error / finally) all funnel into the idempotent
        # finish() (manual instrumentation, no ASGI middleware).
        trace_state = CURRENT_TRACE_STATE.get()
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            first = True
            async for res in response:
                if first:
                    diag.event(
                        request.get(request_id_field_name), "frontend.first_result"
                    )
                    first = False
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield f"data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response
            )
            if trace_state is not None:
                _record_http_status(trace_state, 200)
                trace_state.finish()
        except (asyncio.CancelledError, GeneratorExit) as e:
            try:
                await response.aclose()
            except Exception as close_error:
                logging.warning(
                    "close streaming response after cancellation failed: %s",
                    close_error,
                )
            if trace_state is not None:
                # StreamingResponse headers were already committed as 200 even
                # though body delivery was cancelled afterwards.
                _record_http_status(trace_state, 200)
                trace_state.finish(error=e, error_type="Cancelled")
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                },
            )
            raise
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            if trace_state is not None:
                # SSE headers already went out as 200; the error only reaches
                # the client inside the stream body, so 200 is the true status.
                _record_http_status(trace_state, 200)
                trace_state.finish(error=e)
            format_e = format_exception(e)
            self._access_logger.log_exception_access(request, e, format_e)
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                    "error_code": str(format_e.get("error_code_str", -1)),
                },
            )
            yield response_data_prefix + json.dumps(
                format_e, ensure_ascii=False
            ) + "\r\n\r\n"
        finally:
            if trace_state is not None:
                # safety net for exits not covered above; idempotent
                trace_state.finish()
            self._global_controller.decrement()
            diag.event(
                request.get(request_id_field_name),
                "frontend.slot_released",
                streaming=True,
            )

    async def inference(self, req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
        request_headers: Dict[str, str] = {}
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            try:
                config = resolve_pretrigger_config(req)
            except (ValueError, TypeError) as error:
                return self._handle_exception(req, error, status_code=400)
            if config.get("pretrigger_scheme", "disable") == "encoder":
                return await self._encoder_inference(req, raw_request, False, config)
            sequence = self._global_controller.increment() % 4096  # 12 bits
            req[request_id_field_name] = generate_request_id(
                self.py_env_configs.server_config.ip,
                self.py_env_configs.server_config.server_port,
                self.server_id,
                sequence,
            )
            request_headers = extract_request_headers(
                getattr(raw_request, "headers", None)
            )
        except Exception as e:
            return self._handle_exception(req, e)

        def generate_call():
            assert self._frontend_worker is not None
            if request_headers:
                return self._frontend_worker.inference(**req, headers=request_headers)
            return self._frontend_worker.inference(**req)

        try:
            rep = await self._infer_wrap(req, raw_request, generate_call)
        except BaseException as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def _infer_wrap(
        self,
        req: Dict[str, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        try:
            rep = await self._infer_impl(req, raw_request, generate_call)
        except BaseException as e:
            diag.event(
                req.get(request_id_field_name), "frontend.error", error=type(e).__name__
            )
            rep = self._handle_exception(req, e)
            # Finish the span here while the real exception is still in hand:
            # the caller only sees the swallowed ORJSONResponse and would
            # otherwise mark a 500 span as OK (OTel semconv: 5xx on SERVER
            # spans must set status Error).
            trace_state = CURRENT_TRACE_STATE.get()
            if trace_state is not None:
                _record_http_status(trace_state, getattr(rep, "status_code", 500))
                if isinstance(e, asyncio.CancelledError):
                    # keep error.type low-cardinality, mirroring stream_response
                    trace_state.finish(error=e, error_type="Cancelled")
                else:
                    trace_state.finish(error=e)
        return rep

    def _start_chat_trace(self, model, raw_request, request_id):
        # Trace entry point: only chat completions get an HTTP SERVER span.
        # Returns None when telemetry is disabled; all calls below are no-ops then.
        loaded_model = (
            self._openai_endpoint.model_name
            if self._openai_endpoint is not None
            else ""
        ) or self.py_env_configs.model_args.model_type
        request_model = model or loaded_model
        initial_trace_attributes = {
            trace_attrs.GEN_AI_SPAN_KIND: "LLM",
            trace_attrs.GEN_AI_OPERATION_NAME: "chat",
            trace_attrs.GEN_AI_SYSTEM: "rtp_llm",
            trace_attrs.LINGJI_FLAG: True,
            trace_attrs.ACS_ARMS_TENANT_SPAN_POLICY: "mask",
        }
        if request_model:
            initial_trace_attributes[trace_attrs.GEN_AI_REQUEST_MODEL] = str(
                request_model
            )
        trace_state = start_server_span(
            "POST /v1/chat/completions",
            raw_request.headers,
            initial_attributes=initial_trace_attributes,
        )
        if trace_state is not None:
            # `request_id` is the Bailian Unitrace index key: spans without it
            # are accepted upstream but unsearchable (verified 2026-07-26)
            trace_state.set_attribute("request_id", str(request_id))
            trace_state.set_attribute(trace_attrs.HTTP_REQUEST_METHOD, "POST")
            trace_state.set_attribute(trace_attrs.HTTP_METHOD, "POST")

        return trace_state

    async def _encoder_inference(self, body, raw_request: Request, chat, config):
        """Admission completes before the shared JSON/SSE response lifecycle starts."""
        try:
            sequence = self._global_controller.increment() % 4096
        except ConcurrencyException as error:
            if diag.SAMPLE_EVERY:
                _chat_admission_event(
                    raw_request, None, self.server_id, self._global_controller
                )
            return self._handle_exception(body, error, status_code=429)
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        if diag.SAMPLE_EVERY:
            _chat_admission_event(
                raw_request, request_id, self.server_id, self._global_controller
            )
        body = dict(body)
        body[request_id_field_name] = request_id
        started = time.monotonic()
        outcome = "error"
        response = None
        trace_state = None
        try:
            if chat:
                trace_state = self._start_chat_trace(
                    body.get("model"), raw_request, request_id
                )
            body, generate_call = prepare_encoder_response(body, chat, config)
            inputs = prepare_pretrigger(
                body, chat, self._openai_endpoint, request_id, config
            )
            if inputs.multimodal_inputs:
                if self.py_env_configs.server_config.vit_server_count != 1:
                    from rtp_llm.config.exceptions import (
                        ExceptionType,
                        FtRuntimeException,
                    )

                    raise FtRuntimeException(
                        ExceptionType.MASTER_NO_VIT_WORKER,
                        "pretrigger requires vit_server_count=1",
                    )
                timeout = self.py_env_configs.vit_config.pretrigger_timeout_ms / 1000.0
                deadline = time.monotonic() + timeout
                await asyncio.wait_for(
                    self._frontend_worker.backend_rpc_server_visitor.submit_pretrigger(
                        inputs, dict(raw_request.headers), deadline
                    ),
                    timeout=timeout,
                )
                outcome = "accepted"
            else:
                outcome = "noop"
            response = await self._infer_impl(body, raw_request, generate_call)
            return response
        except asyncio.CancelledError as error:
            outcome = "cancelled"
            if trace_state is not None:
                trace_state.finish(error=error, error_type="Cancelled")
            raise
        except Exception as error:
            code = pretrigger_error_status(error)
            outcome = str(code)
            logging.warning(
                "pretrigger request_id=%s status=%s: %s", request_id, code, error
            )
            response = self._handle_exception(body, error, status_code=code)
            if trace_state is not None:
                _record_http_status(trace_state, code)
                trace_state.finish(error=error)
            return response
        finally:
            # SSE completion/cancellation releases this slot in stream_response.
            if not isinstance(response, StreamingResponse):
                if trace_state is not None:
                    _record_http_status(
                        trace_state, getattr(response, "status_code", 500)
                    )
                    trace_state.finish()
                self._global_controller.decrement()
                diag.event(request_id, "frontend.slot_released", outcome=outcome)
            try:
                kmonitor.report(
                    AccMetrics.PRETRIGGER_OUTCOME_QPS_METRIC, 1, {"outcome": outcome}
                )
                kmonitor.report(
                    GaugeMetrics.PRETRIGGER_SUBMISSION_RT_METRIC,
                    (time.monotonic() - started) * 1000,
                )
            except Exception:
                logging.exception("Failed to report pretrigger metrics")

    async def chat_completion(
        self,
        request: Union[ChatCompletionRequest, Dict[str, Any]],
        raw_request: Request,
    ):
        body = (
            request.model_dump(exclude_none=True)
            if isinstance(request, ChatCompletionRequest)
            else request
        )
        try:
            config = resolve_pretrigger_config(body, chat=True)
        except (ValueError, TypeError) as error:
            return self._handle_exception(body, error, status_code=400)
        if config.get("pretrigger_scheme", "disable") == "encoder":
            return await self._encoder_inference(body, raw_request, True, config)
        if not isinstance(request, ChatCompletionRequest):
            try:
                request = ChatCompletionRequest.model_validate(body)
            except ValidationError as error:
                # Preserve FastAPI's conventional request-validation response.
                errors = [
                    dict(item, loc=("body", *item["loc"])) for item in error.errors()
                ]
                raise RequestValidationError(errors, body=body) from error
        try:
            sequence = self._global_controller.increment() % 4096
        except ConcurrencyException:
            if diag.SAMPLE_EVERY:
                _chat_admission_event(
                    raw_request, None, self.server_id, self._global_controller
                )
            raise
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )

        if diag.SAMPLE_EVERY:
            _chat_admission_event(
                raw_request, request_id, self.server_id, self._global_controller
            )
        trace_state = self._start_chat_trace(request.model, raw_request, request_id)

        def generate_call():
            assert self._openai_endpoint != None
            response = self._openai_endpoint.chat_completion(
                request_id, request, raw_request
            )
            assert isinstance(
                response, CompleteResponseAsyncGenerator
            ), f"error type: {type(response)}"
            return response

        try:
            if request.prompt_logprobs is not None:
                request.stream = False
            elif request.extra_configs is not None and getattr(
                request.extra_configs, "return_prompt_logits", False
            ):
                request.stream = False
            request_dict = request.model_dump(exclude_none=True)
            request_dict[request_id_field_name] = request_id
            rep = await self._infer_wrap(request_dict, raw_request, generate_call)
        except BaseException as e:
            if trace_state is not None:
                # re-raise ends in FastAPI's generic exception handler -> 500
                _record_http_status(trace_state, 500)
                trace_state.finish(error=e)
            self._global_controller.decrement()
            diag.event(request_id, "frontend.slot_released", error=type(e).__name__)
            raise e

        if not isinstance(rep, StreamingResponse):
            # non-streaming lifecycle ends here; streaming spans are finished
            # by stream_response's four exits
            if trace_state is not None:
                # Swallowed-error spans were already finished (status ERROR)
                # inside _infer_wrap; these calls are dropped/no-op for them
                # and only take effect on the success path.
                _record_http_status(trace_state, getattr(rep, "status_code", 200))
                trace_state.finish()
            self._global_controller.decrement()
            diag.event(
                request_id,
                "frontend.slot_released",
                status=getattr(rep, "status_code", 200),
            )

        return rep

    async def batch_chat_completion(self, request, raw_request: Request):
        from rtp_llm.openai.api_datatype import BatchChatCompletionResponse

        sequence = self._global_controller.increment() % 4096
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        try:
            assert self._openai_endpoint is not None
            responses = await self._openai_endpoint.batch_chat_completion(
                request_id, request
            )
            return ORJSONResponse(
                content=BatchChatCompletionResponse(
                    responses=[r.model_dump(exclude_none=True) for r in responses]
                ).model_dump()
            )
        finally:
            self._global_controller.decrement()

    async def batch_infer(self, req: dict, raw_request: Request):
        from rtp_llm.frontend.frontend_worker import BatchPipelineResponse

        # Concurrency accounting: a batch counts as ONE scheduling unit because the engine
        # atomically enqueues all prompts via BatchGenerateCall. Per-item counting would over-
        # reject under the same concurrency_limit; the trade-off is that a large batch occupies
        # only one slot regardless of N.
        sequence = self._global_controller.increment() % 4096
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        try:
            assert self._frontend_worker is not None
            prompts = req.get("prompt_batch", [])
            generate_config = req.get("generate_config", {})
            result = await self._frontend_worker.batch_infer(
                prompts=prompts,
                request_id=request_id,
                generate_config=generate_config,
            )
            return ORJSONResponse(content=result.model_dump(exclude_none=True))
        finally:
            self._global_controller.decrement()

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            assert self._openai_endpoint != None
            return self._openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def _handle_exception(
        self, request: Dict[str, Any], e: BaseException, status_code=500
    ):
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                },
            )
            self._access_logger.log_exception_access(request, e)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                    "error_code": error_code_str,
                },
            )
            self._access_logger.log_exception_access(request, e, exception_json)

        rep = ORJSONResponse(exception_json, status_code=status_code)
        return rep

    async def _call_generate_with_report(
        self,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
        is_streaming: bool,
    ):
        # Captured here (request handler context) instead of inside the
        # generator: StreamingResponse may iterate it from another task whose
        # contextvars snapshot no longer holds CURRENT_TRACE_STATE.
        trace_state = CURRENT_TRACE_STATE.get()

        async def __gen_response_with_report(start_time: float, response_generator):
            last_iterate_time = current_time_ms()
            first_response = True
            last_observed_output_tokens = 0
            fallback_token_debt = 0
            iter_count = 0
            async for response in response_generator:
                end_time = current_time_ms()
                if first_response:
                    first_response = False
                    if trace_state is not None and is_streaming:
                        # This marks frontend delivery availability. It is not
                        # the engine TTFT boundary and is intentionally absent
                        # from non-streaming requests, whose first response is
                        # already the complete body.
                        trace_state.add_event(trace_attrs.EVENT_FIRST_RESPONSE_CHUNK)
                    kmonitor.report(
                        GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC,
                        end_time - last_iterate_time,
                    )
                else:
                    step_output_len = 1
                    if hasattr(response, "aux_info"):
                        if isinstance(response.aux_info, list):
                            step_output_len = 0
                            for info in response.aux_info:
                                step_output_len += info.get("step_output_len", 1)
                        elif isinstance(response.aux_info, dict):
                            step_output_len = max(
                                response.aux_info.get("step_output_len", 1),
                                step_output_len,
                            )

                    kmonitor.report(
                        GaugeMetrics.RESPONSE_ITER_RT_METRIC,
                        (end_time - last_iterate_time) / step_output_len,
                    )
                kmonitor.report(
                    AccMetrics.ITER_QPS_METRIC,
                    1,
                    {
                        "rank_id": self.rank_id,
                        "server_id": self.server_id,
                    },
                )
                if trace_state is not None and is_streaming:
                    try:
                        output_tokens = _output_token_total(response)
                        visible_outputs = _visible_output_count(response)
                        if visible_outputs > 0:
                            visible_tokens = 0
                            if output_tokens is None:
                                visible_tokens = visible_outputs
                                fallback_token_debt += visible_tokens
                            elif output_tokens < last_observed_output_tokens:
                                # A restarted/rebased stream invalidates prior
                                # cumulative accounting. Count this delivery by
                                # its visible lower bound and rebase the cursor.
                                last_observed_output_tokens = output_tokens
                                fallback_token_debt = 0
                                visible_tokens = visible_outputs
                            else:
                                observed_token_delta = (
                                    output_tokens - last_observed_output_tokens
                                )
                                last_observed_output_tokens = output_tokens
                                repaid_debt = min(
                                    observed_token_delta, fallback_token_debt
                                )
                                fallback_token_debt -= repaid_debt
                                visible_tokens = observed_token_delta - repaid_debt
                                if observed_token_delta == 0:
                                    visible_tokens = visible_outputs
                                    fallback_token_debt += visible_tokens
                            if visible_tokens > 0:
                                trace_state.record_frontend_output_tokens(
                                    visible_tokens
                                )
                    except Exception:  # noqa: BLE001 - observation must not block data
                        logging.debug(
                            "frontend token observation failed", exc_info=True
                        )
                last_iterate_time = end_time
                iter_count += 1
                yield response
            kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
            kmonitor.report(
                GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
            )
            kmonitor.report(
                AccMetrics.SUCCESS_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                },
            )

        assert self._frontend_worker is not None
        start_time = current_time_ms()
        response_generator = generate_call()
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, response_generator),
            response_generator._collect_complete_response_func,
        )

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        # Single point covering streaming + non-streaming: write request-level
        # gen_ai.* business attributes onto the HTTP SERVER span from the fully
        # aggregated response (usage / finish_reason / AuxInfo). No-op when
        # telemetry is off.
        record_response_attributes(complete_response)
        self._access_logger.log_success_access(req, complete_response)

        return complete_response

    async def _infer_impl(
        self,
        req: Dict[Any, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
    ):
        assert self._frontend_worker is not None
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": req.get("source", "unkown"),
            },
        )
        self._access_logger.log_query_access(req)
        is_streaming = self._frontend_worker.is_streaming(req)
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        diag.event(
            req.get(request_id_field_name),
            "frontend.generate_start",
            streaming=is_streaming,
        )
        res = await self._call_generate_with_report(generate_call, is_streaming)

        if is_streaming:
            return StreamingResponse(
                self.stream_response(req, res), media_type="text/event-stream"
            )
        first = True
        async for x in res:
            if first:
                diag.event(req.get(request_id_field_name), "frontend.first_result")
                first = False
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                raise asyncio.CancelledError("client disconnects")

        complete_response = await self._collect_complete_response_and_record_access_log(
            req, res
        )
        return ORJSONResponse(content=complete_response)

    def tokenize(self, req: str | Dict[str, Any]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if ChatCompletionRequest.is_openai_request(req):
                chat_request = ChatCompletionRequest(**req)
                token_ids = self._openai_endpoint.render_chat(chat_request).input_ids
            else:
                prompt = req.pop("prompt")
                token_ids = self._frontend_worker.pipeline.encode(prompt)
            return ORJSONResponse({"token_ids": token_ids})
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def tokenizer_encode(self, req: Union[str, Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop("prompt")
            assert self._frontend_worker is not None
            if req.get("return_offsets_mapping", None) == True:
                mapping = self._frontend_worker.tokenizer_offset_mapping(prompt)
                response = TokenizerEncodeResponse(
                    offset_mapping=mapping["offset_mapping"],
                    token_ids=mapping["input_ids"],
                )
            else:
                token_ids, tokens = self._frontend_worker.tokenizer_encode(prompt)
                response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def check_health(self):
        assert self._frontend_worker is not None
        return (
            self._frontend_worker.backend_rpc_server_visitor.is_backend_service_ready(
                refresh=False
            )
        )
