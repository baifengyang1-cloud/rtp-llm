import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import grpc

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    HIGHER_PRIORITY_AHEAD,
    RESOURCE_EXHAUSTED,
    SAME_PRIORITY_AHEAD,
    SCHEDULE_FAILURE_REASON_UNSPECIFIED,
    FlexlbScheduleResponsePB,
    FlexlbServerStatusPB,
)
from rtp_llm.server.master_client import MasterClient


class _FakeMasterConfig:
    master_max_connect_pool_size = 4
    master_session_timeout_s = 1
    master_default_timeout_ms = 3600000


class _FakeHostService:
    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return None


class _FakeHostServiceWithSlave(_FakeHostService):
    def get_slave_addr(self):
        return "slave:1234"


class _FakeGenerateConfig:
    max_new_tokens = 17
    num_beams = 2
    force_disable_sp_run = True
    ttft_timeout_ms = 3000
    timeout_ms = -1
    traffic_reject_priority = 12


class _FakeInput:
    prompt_length = 5

    def __init__(self, headers=None):
        self.generate_config = _FakeGenerateConfig()
        self.headers = {"x-request-id": "req-1"} if headers is None else headers


class _CaptureMasterClient(MasterClient):
    def __init__(self):
        super().__init__(
            host_service=_FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.calls = []

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        self.calls.append(
            {
                "addr": addr,
                "request_pb": request_pb,
                "timeout_s": timeout_s,
                "request_id": request_id,
            }
        )
        return FlexlbScheduleResponsePB(
            success=True,
            code=200,
            server_status=[
                FlexlbServerStatusPB(
                    role="PREFILL",
                    server_ip="10.0.0.7",
                    http_port=8080,
                    grpc_port=9000,
                )
            ],
            enqueued_by_master=True,
        )


class _DeadlineMasterClient(MasterClient):
    def __init__(self):
        super().__init__(
            host_service=_FakeHostServiceWithSlave(),
            master_config=_FakeMasterConfig(),
        )
        self.calls = []

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        self.calls.append(addr)
        raise FtRuntimeException(
            ExceptionType.DEADLINE_EXCEEDED, "schedule deadline exceeded"
        )


class _RejectingMasterClient(MasterClient):
    def __init__(self, code, reason, *, include_reason=True):
        super().__init__(
            host_service=_FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.code = code
        self.reason = reason
        self.include_reason = include_reason

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        fields = {
            "code": int(self.code),
            "error_message": "private scheduler diagnostic",
            "queue_length": 3,
        }
        if self.include_reason:
            fields["admission_reject_reason"] = int(self.reason)
        return SimpleNamespace(**fields)


class _FakeInputPB:
    def SerializeToString(self):
        return b"serialized-input"


class MasterClientBatchPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_vit_only_deadline_retries_slave_without_cancelling_admission(self):
        client = MasterClient(
            host_service=_FakeHostServiceWithSlave(), master_config=_FakeMasterConfig()
        )
        error = grpc.aio.AioRpcError(
            grpc.StatusCode.DEADLINE_EXCEEDED, (), (), "route timed out"
        )
        stub = MagicMock()
        stub.Schedule = AsyncMock(side_effect=error)
        client._get_channel = MagicMock()
        client._close_channel = AsyncMock()
        client._best_effort_cancel = AsyncMock()
        with patch("rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub):
            response = await client.get_backend_role_addrs(
                block_cache_keys=[],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=100,
                vit_only=True,
            )
        self.assertTrue(response.connection_failed)
        self.assertEqual(stub.Schedule.await_count, 2)
        self.assertTrue(
            all(call.args[0].vit_only for call in stub.Schedule.call_args_list)
        )
        client._best_effort_cancel.assert_not_awaited()

    async def test_admission_deadline_remains_terminal(self):
        client = MasterClient(
            host_service=_FakeHostServiceWithSlave(), master_config=_FakeMasterConfig()
        )
        error = grpc.aio.AioRpcError(
            grpc.StatusCode.DEADLINE_EXCEEDED, (), (), "admission timed out"
        )
        stub = MagicMock()
        stub.Schedule = AsyncMock(side_effect=error)
        client._get_channel = MagicMock()
        client._close_channel = AsyncMock()
        client._best_effort_cancel = AsyncMock()
        with patch("rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub):
            with self.assertRaises(FtRuntimeException) as raised:
                await client.get_backend_role_addrs(
                    block_cache_keys=[],
                    cache_key_block_size=1024,
                    input=_FakeInput(),
                    request_id=100,
                    input_pb=_FakeInputPB(),
                )
        self.assertEqual(
            raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
        )
        self.assertEqual(stub.Schedule.await_count, 1)
        client._best_effort_cancel.assert_awaited_once()

    def test_python_reason_enum_matches_schedule_wire_values(self):
        self.assertEqual(
            int(AdmissionRejectReason.UNSPECIFIED),
            SCHEDULE_FAILURE_REASON_UNSPECIFIED,
        )
        self.assertEqual(
            int(AdmissionRejectReason.HIGHER_PRIORITY_AHEAD),
            HIGHER_PRIORITY_AHEAD,
        )
        self.assertEqual(
            int(AdmissionRejectReason.SAME_PRIORITY_AHEAD),
            SAME_PRIORITY_AHEAD,
        )
        self.assertEqual(
            int(AdmissionRejectReason.RESOURCE_EXHAUSTED),
            RESOURCE_EXHAUSTED,
        )

    async def test_schedule_payload_contains_batch_fields_and_pb(self):
        client = _CaptureMasterClient()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1, 2, 3],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=99,
            input_pb=_FakeInputPB(),
        )

        self.assertTrue(response.is_ok)
        self.assertTrue(response.enqueued_by_master)
        self.assertEqual(response.role_addrs[0].ip, "10.0.0.7")

        call = client.calls[0]
        request_pb = call["request_pb"]
        self.assertEqual(call["addr"], "master:1234")
        self.assertEqual(call["timeout_s"], 3.0)
        self.assertEqual(call["request_id"], 99)
        self.assertEqual(list(request_pb.block_cache_keys), [1, 2, 3])
        self.assertEqual(request_pb.seq_len, 5)
        self.assertEqual(request_pb.generate_timeout, 3000)
        self.assertEqual(request_pb.request_id, 99)
        self.assertEqual(request_pb.max_new_tokens, 17)
        self.assertEqual(request_pb.num_beams, 2)
        self.assertTrue(request_pb.force_disable_sp_run)
        self.assertEqual(request_pb.generate_input, b"serialized-input")
        self.assertEqual(request_pb.cache_key_block_size, 1024)
        self.assertEqual(request_pb.priority, 50)

    async def test_schedule_payload_priority_from_qos_header(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "70"}),
            request_id=101,
            input_pb=_FakeInputPB(),
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 70)

    async def test_schedule_payload_priority_defaults_when_header_missing(self):
        client = _CaptureMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={}),
            request_id=102,
            input_pb=_FakeInputPB(),
        )

        self.assertEqual(client.calls[0]["request_pb"].priority, 50)

    async def test_schedule_payload_priority_invalid_header_no_raise(self):
        client = _CaptureMasterClient()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1],
            cache_key_block_size=1024,
            input=_FakeInput(headers={"x-dashscope-inner-qos-level": "high"}),
            request_id=103,
            input_pb=_FakeInputPB(),
        )

        self.assertTrue(response.is_ok)
        self.assertEqual(client.calls[0]["request_pb"].priority, 50)

    async def test_schedule_deadline_does_not_retry_slave(self):
        client = _DeadlineMasterClient()

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs(
                block_cache_keys=[1],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=100,
                input_pb=_FakeInputPB(),
            )

        self.assertEqual(
            raised.exception.exception_type, ExceptionType.DEADLINE_EXCEEDED
        )
        self.assertEqual(client.calls, ["master:1234"])

    async def test_schedule_failure_preserves_typed_admission_reason(self):
        cases = (
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
            ),
            (
                ExceptionType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
            ),
            (
                ExceptionType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED,
            ),
        )
        for exception_type, reason in cases:
            with self.subTest(exception_type=exception_type, reason=reason):
                client = _RejectingMasterClient(exception_type, reason)
                with self.assertRaises(FtRuntimeException) as raised:
                    await client.get_backend_role_addrs(
                        block_cache_keys=[1],
                        cache_key_block_size=1024,
                        input=_FakeInput(),
                        request_id=104,
                        input_pb=_FakeInputPB(),
                    )

                self.assertEqual(exception_type, raised.exception.exception_type)
                self.assertEqual(
                    reason,
                    raised.exception.admission_reject_reason,
                )
                self.assertEqual(
                    "private scheduler diagnostic",
                    raised.exception.message,
                )

    async def test_missing_reason_field_falls_back_to_unspecified(self):
        client = _RejectingMasterClient(
            ExceptionType.ADMISSION_UNAVAILABLE,
            AdmissionRejectReason.UNSPECIFIED,
            include_reason=False,
        )

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs(
                block_cache_keys=[1],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=105,
                input_pb=_FakeInputPB(),
            )

        self.assertEqual(
            AdmissionRejectReason.UNSPECIFIED,
            raised.exception.admission_reject_reason,
        )

    async def test_unknown_reason_is_preserved_as_invalid(self):
        client = _RejectingMasterClient(
            ExceptionType.PRIORITY_PREEMPTED,
            999,
        )

        with self.assertRaises(FtRuntimeException) as raised:
            await client.get_backend_role_addrs(
                block_cache_keys=[1],
                cache_key_block_size=1024,
                input=_FakeInput(),
                request_id=106,
                input_pb=_FakeInputPB(),
            )

        self.assertEqual(
            AdmissionRejectReason.INVALID,
            raised.exception.admission_reject_reason,
        )


class PretriggerVitRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_rpc_policy_and_capacity_errors_do_not_allow_discovery(self):
        from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
            FlexlbScheduleRequestPB,
        )

        client = MasterClient(
            host_service=_FakeHostServiceWithSlave(), master_config=_FakeMasterConfig()
        )
        for status, expected in (
            (grpc.StatusCode.RESOURCE_EXHAUSTED, 429),
            (grpc.StatusCode.PERMISSION_DENIED, 403),
            (grpc.StatusCode.UNAUTHENTICATED, 403),
            (grpc.StatusCode.INTERNAL, 503),
        ):
            with self.subTest(status=status):
                stub = SimpleNamespace(
                    Schedule=AsyncMock(
                        side_effect=grpc.aio.AioRpcError(status, (), (), "rejected")
                    )
                )
                with patch.object(client, "_get_channel", return_value=object()), patch(
                    "rtp_llm.server.master_client.FlexlbServiceStub", return_value=stub
                ):
                    response = await client._send_schedule_request(
                        "master:1234", FlexlbScheduleRequestPB(vit_only=True), 1, 123
                    )
                self.assertEqual(response.code, expected)

    async def test_media_only_payload_and_shared_deadline(self):
        from unittest.mock import AsyncMock, patch

        client = MasterClient(
            host_service=_FakeHostServiceWithSlave(), master_config=_FakeMasterConfig()
        )
        success = FlexlbScheduleResponsePB(
            code=200,
            server_status=[
                FlexlbServerStatusPB(
                    role="VIT", server_ip="vit", http_port=8000, grpc_port=8001
                )
            ],
        )
        client._send_schedule_request = AsyncMock(side_effect=[None, success])
        with patch(
            "rtp_llm.server.master_client.time.monotonic", side_effect=[10.0, 10.4]
        ):
            result = await client.route_vit(
                ["a", "b"],
                123,
                {"x-api-key": "key", "x-dashscope-inner-qos-level": "42"},
                10.6,
            )
        self.assertTrue(result.is_ok)
        first, second = client._send_schedule_request.call_args_list
        self.assertAlmostEqual(first.args[2], 0.6)
        self.assertAlmostEqual(second.args[2], 0.2)
        payload = first.args[1]
        self.assertEqual(list(payload.media_keys), ["a", "b"])
        self.assertEqual(list(payload.block_cache_keys), [])
        self.assertEqual(payload.seq_len, 0)
        self.assertEqual(payload.api_key, "key")
        self.assertEqual(payload.priority, 42)
        self.assertTrue(payload.vit_only)
        self.assertFalse(payload.generate_input)
        self.assertEqual(payload.generate_timeout, 0)

    async def test_explicit_rejection_never_tries_slave(self):
        import time
        from unittest.mock import AsyncMock

        client = MasterClient(
            host_service=_FakeHostServiceWithSlave(), master_config=_FakeMasterConfig()
        )
        client._send_schedule_request = AsyncMock(
            return_value=FlexlbScheduleResponsePB(code=429, error_message="full")
        )
        result = await client.route_vit(["a"], 124, {}, time.monotonic() + 5)
        self.assertFalse(result.connection_failed)
        self.assertEqual(result.error_code, 429)
        self.assertEqual(client._send_schedule_request.await_count, 1)


if __name__ == "__main__":
    unittest.main()
