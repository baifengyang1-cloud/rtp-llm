import asyncio
import json
import time
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import (
    resolved_multimodal_cache_keys,
    serialize_multimodal_inputs,
)
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.frontend.pretrigger import (
    prepare_pretrigger,
    pretrigger_error_status,
    resolve_pretrigger_config,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.qwen35_renderer import Qwen35Renderer
from rtp_llm.structure.multimodal_request import native_multimodal_inputs
from rtp_llm.structure.request_extractor import RequestExtractor


class MediaPreparationTest(TestCase):
    def setUp(self):
        self.renderer = object.__new__(Qwen35Renderer)
        self.renderer.tokenizer = MagicMock()
        self.renderer.tokenizer.apply_chat_template.return_value = "prompt"
        self.renderer.tokenizer.encode.return_value = [1]
        self.endpoint = SimpleNamespace(chat_renderer=self.renderer)

    def test_native_ignores_generation_and_preserves_media_overrides(self):
        body = {
            "images": ["fake://image"],
            "generate_config": {"max_new_tokens": "ignored", "min_pixels": 100},
            "min_pixels": 200,
            "stream": "ignored",
        }
        actual = prepare_pretrigger(body, False, None, 1)
        normal = serialize_multimodal_inputs(
            native_multimodal_inputs(body["images"]), GenerateConfig(min_pixels=200), 1
        )
        self.assertEqual(actual, normal)
        self.assertEqual(
            resolved_multimodal_cache_keys(actual),
            resolved_multimodal_cache_keys(normal),
        )

    def test_native_batch_and_malformed_media(self):
        actual = prepare_pretrigger({"urls": [["a"], ["b", "c"]]}, False, None, 1)
        self.assertEqual(
            [i.multimodal_url for i in actual.multimodal_inputs], ["a", "b", "c"]
        )
        for body in (
            {"images": None},
            {"urls": "a"},
            {"images": [""]},
            {"urls": [["a"], 1]},
            {"prompt_batch": ["a", "b"], "urls": [["image"]]},
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                prepare_pretrigger(body, False, None, 1)

    def test_chat_parity_with_mixed_per_media_defaults(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "a"}},
                    {
                        "type": "video_url",
                        "video_url": {"url": "b"},
                        "preprocess_config": {"fps": 2, "max_pixels": 1024},
                    },
                ],
            }
        ]
        body = {
            "messages": messages,
            "extra_configs": {"min_pixels": 128, "max_new_tokens": "ignored"},
            "max_tokens": -100,
            "tool_choice": {"invalid": True},
        }
        actual = prepare_pretrigger(body, True, self.endpoint, 3)
        self.renderer.tokenizer.encode.assert_not_called()
        self.renderer.tokenizer.apply_chat_template.assert_not_called()
        rendered = self.renderer.render_chat(ChatCompletionRequest(messages=messages))
        normal = serialize_multimodal_inputs(
            rendered.multimodal_inputs, GenerateConfig(min_pixels=128), 3
        )
        self.assertEqual(actual, normal)
        self.assertEqual(
            resolved_multimodal_cache_keys(actual),
            resolved_multimodal_cache_keys(normal),
        )

    def test_empty_and_malformed_chat(self):
        self.assertFalse(
            prepare_pretrigger({}, True, self.endpoint, 1).multimodal_inputs
        )
        for content in (
            {"type": "image_url"},
            {"type": "video_url", "video_url": {"url": ""}},
            {"type": "audio_url", "audio_url": {"url": "a"}},
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                prepare_pretrigger(
                    {"messages": [{"role": "assistant", "content": [content]}]},
                    True,
                    self.endpoint,
                    1,
                )


class SubmissionTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = object.__new__(FrontendServer)
        self.server._global_controller = MagicMock()
        self.server._global_controller.increment.return_value = 1
        self.server.server_id = "0"
        self.server.rank_id = "0"
        self.server.is_embedding = False
        self.server._access_logger = MagicMock()
        self.server.py_env_configs = SimpleNamespace(
            server_config=SimpleNamespace(
                ip="127.0.0.1", server_port=8000, vit_server_count=1
            ),
            vit_config=SimpleNamespace(pretrigger_timeout_ms=1000),
            model_args=SimpleNamespace(model_type="qwen35_dense"),
        )
        self.server._openai_endpoint = None
        self.visitor = SimpleNamespace(submit_pretrigger=AsyncMock())
        self.server._frontend_worker = SimpleNamespace(
            backend_rpc_server_visitor=self.visitor,
            is_streaming=lambda req: FrontendWorker.is_streaming(None, req),
            inference=MagicMock(side_effect=AssertionError("generation must not run")),
        )
        self.request = SimpleNamespace(
            headers={"x-api-key": "key"}, is_disconnected=AsyncMock(return_value=False)
        )
        self.report = patch("rtp_llm.frontend.frontend_server.kmonitor.report").start()
        self.addCleanup(patch.stopall)

    async def raw(self, body):
        body = dict(body)
        body["generate_config"] = dict(
            body.get("generate_config", {}), pretrigger_scheme="encoder"
        )
        return await self.server.inference(body, self.request)

    async def chat(self, body):
        body = dict(body)
        body["extra_configs"] = dict(
            body.get("extra_configs", {}), pretrigger_scheme="encoder"
        )
        return await self.server.chat_completion(body, self.request)

    async def test_noop_and_acknowledgement(self):
        response = await self.raw({})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.body),
            {"response": "", "finished": True, "aux_info": {}},
        )
        self.visitor.submit_pretrigger.assert_not_called()
        response = await self.raw({"images": ["a"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.body),
            {"response": "", "finished": True, "aux_info": {}},
        )
        inputs, headers, deadline = self.visitor.submit_pretrigger.call_args.args
        self.assertEqual(headers["x-api-key"], "key")
        self.assertGreater(deadline, time.monotonic())
        self.assertEqual(self.server._global_controller.decrement.call_count, 2)

    async def test_cancellation_releases_frontend_accounting(self):
        started = asyncio.Event()

        async def blocked(*args):
            started.set()
            await asyncio.Event().wait()

        self.visitor.submit_pretrigger.side_effect = blocked
        task = asyncio.create_task(self.raw({"images": ["a"]}))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.server._global_controller.decrement.assert_called_once()

    async def test_deadline_and_admission_errors(self):
        self.server.py_env_configs.vit_config.pretrigger_timeout_ms = 20

        async def blocked(*args):
            await asyncio.Event().wait()

        self.visitor.submit_pretrigger.side_effect = blocked
        response = await self.raw({"images": ["a"]})
        self.assertEqual(response.status_code, 504)
        for code, expected in (
            (ExceptionType.CONCURRENCY_LIMIT_ERROR, 429),
            (ExceptionType.MASTER_NO_VIT_WORKER, 503),
        ):
            self.visitor.submit_pretrigger.side_effect = FtRuntimeException(
                code, "test"
            )
            response = await self.raw({"images": ["a"]})
            self.assertEqual(response.status_code, expected)
        self.assertEqual(self.server._global_controller.decrement.call_count, 3)

    async def test_invalid_scheme_before_generation_or_submission(self):
        for invalid in (None, True, False, 0, 1, [], {}, "", "Encoder", "bogus"):
            for chat in (False, True):
                with self.subTest(invalid=invalid, chat=chat):
                    key = "extra_configs" if chat else "generate_config"
                    method = (
                        self.server.chat_completion if chat else self.server.inference
                    )
                    response = await method(
                        {key: {"pretrigger_scheme": invalid}}, self.request
                    )
                    self.assertEqual(response.status_code, 400)
                    message = json.loads(response.body)["message"]
                    self.assertIn("pretrigger_scheme", message)
                    self.assertIn('"disable" or "encoder"', message)
        self.visitor.submit_pretrigger.assert_not_called()
        self.server._frontend_worker.inference.assert_not_called()
        self.server._global_controller.increment.assert_not_called()

    async def test_raw_shapes_preserved_independently_of_media(self):
        for sequences in (0, 1, 3):
            for auxiliary in (False, True):
                for body, batch_size in (
                    ({}, None),
                    ({"prompt_batch": [""]}, 1),
                    ({"prompt_batch": ["", ""]}, 2),
                    ({"images": [[], ["a"], []]}, 3),
                    ({"prompt": ["a", "b"]}, None),
                ):
                    with self.subTest(
                        sequences=sequences, auxiliary=auxiliary, body=body
                    ):
                        body = dict(
                            body,
                            generate_config={
                                "num_return_sequences": sequences,
                                "aux_info": auxiliary,
                            },
                        )
                        response = await self.raw(body)
                        self.assertEqual(response.status_code, 200)
                        actual = json.loads(response.body)
                        item = {
                            "response": [""] * sequences if sequences else "",
                            "finished": True,
                            "aux_info": (
                                ([{} for _ in range(sequences)] if auxiliary else [])
                                if sequences
                                else {}
                            ),
                        }
                        expected = (
                            {"response_batch": [item] * batch_size}
                            if batch_size is not None
                            else item
                        )
                        self.assertEqual(actual, expected)

    async def test_chat_choices_and_empty_complete_response(self):
        for overrides, count in (
            ({}, 1),
            ({"n": 2}, 2),
            ({"n": 2, "extra_configs": {"num_beams": 3}}, 3),
            ({"extra_configs": {"variable_num_beams": [4]}}, 1),
            ({"extra_configs": {"variable_num_beams": [4, 2]}}, 2),
            ({"extra_configs": {"num_return_sequences": 3}}, 1),
        ):
            response = await self.chat(
                dict(
                    overrides,
                    model="ignored-model",
                    max_tokens="ignored",
                    tool_choice="required",
                )
            )
            self.assertEqual(response.status_code, 200)
            body = json.loads(response.body)
            self.assertEqual(body["id"], "chat-")
            self.assertEqual(body["object"], "chat.completion")
            self.assertEqual(body["model"], "")
            self.assertEqual(
                body["usage"],
                dict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
            self.assertEqual(
                body["choices"],
                [
                    dict(
                        index=i,
                        message={"role": "assistant", "partial": False},
                        finish_reason="stop",
                    )
                    for i in range(count)
                ],
            )
            self.assertEqual(
                set(body), {"id", "object", "created", "model", "usage", "choices"}
            )
        self.visitor.submit_pretrigger.assert_not_called()

    async def test_transport_framing_and_access_logging(self):
        for chat, body, prefix, sentinel in (
            (True, {"stream": True, "prompt_logprobs": 2}, "data: ", False),
            (False, {"stream": True}, "data: ", False),
            (
                False,
                {"yield_generator": True, "return_incremental": True},
                "data:",
                True,
            ),
            (False, {"stream": True, "yield_generator": True}, "data: ", False),
            (False, {"generation_config": {"yield_generator": True}}, "data:", True),
        ):
            with self.subTest(chat=chat, body=body):
                self.server._global_controller.decrement.reset_mock()
                self.server._access_logger.reset_mock()
                response = await (self.chat(body) if chat else self.raw(body))
                self.assertIsInstance(response, StreamingResponse)
                self.server._global_controller.decrement.assert_not_called()
                chunks = [chunk async for chunk in response.body_iterator]
                self.assertEqual(len(chunks), 2 if sentinel else 1)
                self.assertTrue(chunks[0].startswith(prefix))
                self.assertTrue(chunks[0].endswith("\r\n\r\n"))
                event = json.loads(chunks[0][len(prefix) :])
                if chat:
                    self.assertEqual(event["id"], "chat")
                    self.assertNotIn("model", event)
                    self.assertEqual(
                        event["choices"],
                        [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": "stop",
                            }
                        ],
                    )
                    self.assertEqual(event["usage"]["total_tokens"], 0)
                else:
                    self.assertEqual(
                        event, {"response": "", "finished": True, "aux_info": {}}
                    )
                if sentinel:
                    self.assertEqual(chunks[-1], "data:[done]\r\n\r\n")
                self.server._global_controller.decrement.assert_called_once()
                self.server._access_logger.log_success_access.assert_called_once()
                self.server._access_logger.log_exception_access.assert_not_called()

    async def test_engine_streaming_does_not_select_sse(self):
        for body in (
            {"generate_config": {"is_streaming": True}},
            {"yield_generator": False, "generate_config": {"yield_generator": True}},
            {
                "generation_config": {"yield_generator": False},
                "generate_config": {"yield_generator": True},
            },
        ):
            response = await self.raw(body)
            self.assertNotIsInstance(response, StreamingResponse)
            self.assertEqual(response.status_code, 200)

    async def test_response_controls_validated_before_admission(self):
        for chat, body in (
            (True, {"model": {"invalid": True}}),
            (True, {"n": 0}),
            (True, {"n": -1}),
            (True, {"extra_configs": {"num_beams": 0}}),
            (True, {"extra_configs": {"variable_num_beams": [1, -1]}}),
            (False, {"num_return_sequences": -1}),
            (False, {"stream": "invalid"}),
            (False, {"yield_generator": []}),
            (False, {"prompt_batch": "bad"}),
            (False, {"return_incremental": True}),
        ):
            with self.subTest(chat=chat, body=body):
                response = await (self.chat(body) if chat else self.raw(body))
                self.assertEqual(response.status_code, 400)
        self.visitor.submit_pretrigger.assert_not_called()

    async def test_streaming_waits_for_admission_and_preserves_errors(self):
        started, admitted = asyncio.Event(), asyncio.Event()

        async def blocked(*args):
            started.set()
            await admitted.wait()

        self.visitor.submit_pretrigger.side_effect = blocked
        task = asyncio.create_task(self.raw({"images": ["a"], "stream": True}))
        await started.wait()
        self.assertFalse(task.done())
        admitted.set()
        response = await task
        self.assertIsInstance(response, StreamingResponse)
        _ = [chunk async for chunk in response.body_iterator]
        self.visitor.submit_pretrigger.side_effect = FtRuntimeException(
            ExceptionType.CONCURRENCY_LIMIT_ERROR, "full"
        )
        response = await self.raw({"images": ["a"], "stream": True})
        self.assertNotIsInstance(response, StreamingResponse)
        self.assertEqual(response.status_code, 429)
        self.assertIn("error_code", json.loads(response.body))

    async def test_disable_keeps_conventional_generation_and_validation(self):
        for scheme in ({}, {"pretrigger_scheme": "disable"}):
            with patch.object(
                self.server, "_infer_wrap", new=AsyncMock(return_value="normal")
            ) as infer:
                response = await self.server.inference(
                    {"prompt": "hello", "generate_config": scheme}, self.request
                )
                self.assertEqual(response, "normal")
                infer.assert_awaited_once()
                response = await self.server.chat_completion(
                    {"messages": [], "extra_configs": scheme}, self.request
                )
                self.assertEqual(response, "normal")
            with self.assertRaises(RequestValidationError):
                await self.server.chat_completion(
                    {"extra_configs": scheme}, self.request
                )
            with self.assertRaises(RequestValidationError):
                await self.server.chat_completion(
                    {
                        "messages": [],
                        "tool_choice": "required",
                        "extra_configs": scheme,
                    },
                    self.request,
                )
        self.visitor.submit_pretrigger.assert_not_called()

    async def test_frontend_capacity_rejection_does_not_release_an_unowned_slot(self):
        from rtp_llm.utils.concurrency_controller import ConcurrencyException

        self.server._global_controller.increment.side_effect = ConcurrencyException(
            "full"
        )
        for method in (self.raw, self.chat):
            response = await method({})
            self.assertEqual(response.status_code, 429)
        self.server._global_controller.decrement.assert_not_called()
        self.visitor.submit_pretrigger.assert_not_called()

    async def test_chat_ignores_raw_transport_controls(self):
        response = await self.chat(
            {
                "yield_generator": True,
                "generation_config": None,
                "extra_configs": {"return_prompt_logits": True},
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertNotIsInstance(response, StreamingResponse)
        self.assertEqual(json.loads(response.body)["object"], "chat.completion")

    async def test_chat_media_admission_matches_no_media_response(self):
        renderer = object.__new__(Qwen35Renderer)
        renderer.tokenizer = MagicMock()
        self.server._openai_endpoint = SimpleNamespace(
            chat_renderer=renderer, model_name="qwen35_dense"
        )
        no_media = await self.chat({})
        with_media = await self.chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "a"}},
                            {
                                "type": "video_url",
                                "video_url": {"url": "b"},
                                "preprocess_config": {"fps": 2},
                            },
                        ],
                    }
                ],
                "tool_choice": {"invalid": "ignored"},
                "extra_configs": {"max_new_tokens": "ignored", "min_pixels": 128},
            }
        )
        self.assertEqual(with_media.status_code, 200)
        expected, actual = json.loads(no_media.body), json.loads(with_media.body)
        expected.pop("created")
        actual.pop("created")
        self.assertEqual(actual, expected)
        self.visitor.submit_pretrigger.assert_awaited_once()
        renderer.tokenizer.encode.assert_not_called()
        renderer.tokenizer.apply_chat_template.assert_not_called()

    def make_app(self):
        from rtp_llm.frontend.frontend_app import FrontendApp
        from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager

        owner = FrontendApp.__new__(FrontendApp)
        owner.frontend_server = self.server
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        return owner, owner.create_app()

    async def http_post(self, app, path, body, sent=None):
        sent = [] if sent is None else sent
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": json.dumps(body).encode(),
                    "more_body": False,
                }
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        )
        start = next(item for item in sent if item["type"] == "http.response.start")
        payload = b"".join(
            item.get("body", b"")
            for item in sent
            if item["type"] == "http.response.body"
        )
        return start["status"], payload

    async def test_existing_http_routes_and_removed_routes(self):
        owner, app = self.make_app()
        for path in ("/chat/completions", "/v1/chat/completions"):
            status, payload = await self.http_post(
                app, path, {"extra_configs": {"pretrigger_scheme": "encoder"}}
            )
            self.assertEqual(status, 200)
            self.assertEqual(
                json.loads(payload)["choices"][0]["message"],
                {"role": "assistant", "partial": False},
            )
            status, _ = await self.http_post(
                app, path, {"extra_configs": {"pretrigger_scheme": None}}
            )
            self.assertEqual(status, 400)
            status, _ = await self.http_post(app, path, {})
            self.assertEqual(status, 422)
        for path in (
            "/pretrigger",
            "/pretrigger/chat/completions",
            "/pretrigger/v1/chat/completions",
        ):
            status, _ = await self.http_post(app, path, {})
            self.assertEqual(status, 404)
        self.assertEqual(owner.shutdown_manager.active_request_count(), 0)

    async def test_http_admission_before_headers_and_shutdown_tracking(self):
        owner, app = self.make_app()
        started, admitted = asyncio.Event(), asyncio.Event()

        async def blocked(*args):
            started.set()
            await admitted.wait()

        self.visitor.submit_pretrigger.side_effect = blocked
        sent = []
        task = asyncio.create_task(
            self.http_post(
                app,
                "/",
                {
                    "images": ["a"],
                    "stream": True,
                    "generate_config": {"pretrigger_scheme": "encoder"},
                },
                sent,
            )
        )
        await started.wait()
        self.assertEqual(sent, [])
        self.assertEqual(owner.shutdown_manager.active_request_count(), 1)
        admitted.set()
        status, payload = await task
        self.assertEqual(status, 200)
        self.assertEqual(
            payload, b'data: {"response":"","finished":true,"aux_info":{}}\r\n\r\n'
        )
        self.assertEqual(owner.shutdown_manager.active_request_count(), 0)
        self.server._global_controller.decrement.assert_called_once()


class SchemeResolutionTest(TestCase):
    def test_native_alias_and_override_resolution_matches_extractor(self):
        for body, expected in (
            ({}, "disable"),
            ({"generation_config": {"pretrigger_scheme": "encoder"}}, "encoder"),
            (
                {
                    "generate_config": {"pretrigger_scheme": "encoder"},
                    "generation_config": {"pretrigger_scheme": None},
                },
                "encoder",
            ),
            (
                {
                    "generate_config": {"pretrigger_scheme": "encoder"},
                    "pretrigger_scheme": "disable",
                },
                "disable",
            ),
            (
                {
                    "generate_config": {"pretrigger_scheme": None},
                    "pretrigger_scheme": "encoder",
                },
                "encoder",
            ),
        ):
            with self.subTest(body=body):
                config = resolve_pretrigger_config(body)
                actual, _ = RequestExtractor(GenerateConfig())._format_generate_config(
                    dict(body)
                )
                self.assertEqual(config.get("pretrigger_scheme", "disable"), expected)
                self.assertEqual(actual.pretrigger_scheme, expected)

    def test_shared_config_validates_strict_scheme(self):
        for invalid in (None, True, 1, [], {}, "Encoder"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                GenerateConfig(pretrigger_scheme=invalid)


class PretriggerRoutingTest(IsolatedAsyncioTestCase):
    def setUp(self):
        from rtp_llm.config.generate_config import RoleAddr, RoleType
        from rtp_llm.ops import VitSeparation
        from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
        from rtp_llm.server.master_client import FlexlbResponse

        self.response_type = FlexlbResponse
        self.address = RoleAddr(
            role=RoleType.VIT, ip="vit", http_port=8000, grpc_port=8001
        )
        self.visitor = object.__new__(BackendRPCServerVisitor)
        self.visitor.vit_separation = VitSeparation.VIT_SEPARATION_REMOTE
        self.visitor.master_client = SimpleNamespace(route_vit=AsyncMock())
        self.visitor.model_rpc_client = SimpleNamespace(submit_embedding=AsyncMock())
        self.visitor.host_service = SimpleNamespace(
            get_backend_role_addrs=MagicMock(return_value=[self.address])
        )
        self.inputs = prepare_pretrigger({"images": ["a"]}, False, None, 1)

    async def test_discovery_only_after_connection_failure(self):
        self.visitor.master_client.route_vit.return_value = (
            self.response_type.connection_failed_response()
        )
        await self.visitor.submit_pretrigger(self.inputs, {}, time.monotonic() + 5)
        self.visitor.host_service.get_backend_role_addrs.assert_called_once()
        self.visitor.model_rpc_client.submit_embedding.assert_awaited_once()
        address, inputs, timeout = (
            self.visitor.model_rpc_client.submit_embedding.call_args.args
        )
        self.assertEqual(address, self.address)
        self.assertIs(inputs, self.inputs)
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 5)

    async def test_rejections_do_not_bypass_flexlb(self):
        for code in (429, 403, int(ExceptionType.MASTER_NO_VIT_WORKER)):
            self.visitor.master_client.route_vit.return_value = (
                self.response_type.error_response(code, "reject")
            )
            with self.assertRaises(FtRuntimeException):
                await self.visitor.submit_pretrigger(
                    self.inputs, {}, time.monotonic() + 5
                )
        self.visitor.host_service.get_backend_role_addrs.assert_not_called()
        self.visitor.model_rpc_client.submit_embedding.assert_not_called()

    async def test_selected_vit_and_expired_budget(self):
        self.visitor.master_client.route_vit.return_value = self.response_type.ok(
            [self.address]
        )
        await self.visitor.submit_pretrigger(self.inputs, {}, time.monotonic() + 5)
        self.visitor.host_service.get_backend_role_addrs.assert_not_called()
        self.visitor.model_rpc_client.submit_embedding.reset_mock()
        with self.assertRaises(asyncio.TimeoutError):
            await self.visitor.submit_pretrigger(self.inputs, {}, time.monotonic() - 1)
        self.visitor.model_rpc_client.submit_embedding.assert_not_called()

    async def test_missing_discovery_endpoint(self):
        self.visitor.master_client.route_vit.return_value = (
            self.response_type.connection_failed_response()
        )
        self.visitor.host_service.get_backend_role_addrs.return_value = []
        with self.assertRaises(FtRuntimeException) as caught:
            await self.visitor.submit_pretrigger(self.inputs, {}, time.monotonic() + 5)
        self.assertEqual(pretrigger_error_status(caught.exception), 503)
        self.visitor.model_rpc_client.submit_embedding.assert_not_called()


if __name__ == "__main__":
    main()
