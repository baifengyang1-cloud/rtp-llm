import threading
from concurrent import futures
from unittest import TestCase

import grpc
import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.multimodal.greennet_hook import (
    GreenNetHandle,
    GreenNetProvider,
    GreenNetVerdict,
)
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.vit_proxy_server import (
    LoadBalancer,
    VitProxyRpcServer,
    WorkerConnectionPool,
)
from rtp_llm.server.vit_rpc_server import MultimodalRpcServer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class CpuEmbedding(MultiModalEmbeddingInterface):
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        return mm_inputs

    def embedding(self, data, **kwargs):
        self.calls += 1
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("test forward was not released")
        return torch.ones(2, 16), None


class RejectProvider(GreenNetProvider):
    def is_enabled(self):
        return True

    async def preprocess_and_submit(self, request, inputs):
        class Handle(GreenNetHandle):
            rewritten_inputs = inputs

            async def wait_result(self):
                return GreenNetVerdict(False, code=2, message="rejected test input")

            def cancel(self):
                pass

        return Handle()


class VitPretriggerRpcTest(TestCase):
    def setUp(self):
        self.engines = []
        self.servers = []
        self.channels = []
        self.parts = []
        self.pool = None

    def tearDown(self):
        for part in self.parts:
            part.release.set()
        for channel in self.channels:
            channel.close()
        for server in self.servers:
            server.stop(0).wait(5)
        if self.pool:
            self.pool.close_all()
        for engine in self.engines:
            engine.stop()

    def _engine(self, provider=None):
        part = CpuEmbedding()
        config = VitConfig()
        config.use_local_preprocess = True
        config.vit_concurrency = 2
        config.vit_max_queue_size = 2
        config.disable_access_log = True
        model = ModelConfig()
        model.mm_related_params.preprocess_batch_size = 1
        engine = MMProcessEngine(
            part, model, config, ProfilingDebugLoggingConfig(), device="cpu"
        )
        if provider:
            engine._greennet_provider = provider
        self.engines.append(engine)
        self.parts.append(part)
        return engine, part

    def _serve(self, servicer):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        add_MultimodalRpcServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        self.servers.append(server)
        return f"127.0.0.1:{port}"

    def _stub(self, address):
        channel = grpc.insecure_channel(address)
        self.channels.append(channel)
        return MultimodalRpcServiceStub(channel)

    @staticmethod
    def _request(request_id=101, url="fake://shared"):
        request = MultimodalInputsPB(request_id=request_id)
        request.multimodal_inputs.add(
            multimodal_url=url, multimodal_type=int(MMUrlType.IMAGE)
        )
        return request

    def test_pretrigger_metadata_ack_and_reuse_on_standalone_worker(self):
        engine, part = self._engine()
        part.release.clear()
        stub = self._stub(self._serve(MultimodalRpcServer(engine)))
        stub.AsyncSubmitEmbedding(
            self._request(701), timeout=2, metadata=(("x-rtp-pretrigger", "1"),)
        )
        self.assertTrue(part.started.wait(2))
        self.assertEqual(part.calls, 1)
        part.release.set()
        response = stub.RemoteMultimodalEmbedding(self._request(702), timeout=2)
        self.assertEqual(list(response.split_size), [2])
        self.assertEqual(part.calls, 1)
        entry = next(iter(engine._embedding_cache._entries.values()))
        self.assertTrue(entry.pretrigger_initiated)
        self.assertFalse(entry.claim_pretrigger_reuse())

    def test_pretrigger_metadata_rejects_proxy_topology(self):
        engine, part = self._engine()
        address = self._serve(MultimodalRpcServer(engine))
        self.pool = WorkerConnectionPool([address])
        proxy = VitProxyRpcServer(LoadBalancer([address]), self.pool)
        stub = self._stub(self._serve(proxy))
        with self.assertRaises(grpc.RpcError) as caught:
            stub.AsyncSubmitEmbedding(
                self._request(703), timeout=2, metadata=(("x-rtp-pretrigger", "1"),)
            )
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(part.calls, 0)

    def test_invalid_input_does_not_leave_cache_claims(self):
        engine, _ = self._engine()
        inputs = [
            MultimodalInput(url, MMUrlType.IMAGE, torch.empty(0), MMPreprocessConfig())
            for url in ("fake://valid", "")
        ]
        with self.assertRaises(ValueError):
            engine.async_submit(inputs, request_id=33)
        self.assertEqual(engine._async_tasks, {})
        self.assertEqual(engine._async_request_tasks, {})
        self.assertEqual(engine._async_admitted, 0)
        self.assertIsNone(engine._embedding_cache.peek(inputs[0].cache_key()))

    def test_stopped_engine_rejects_new_claims(self):
        engine, _ = self._engine()
        engine.stop()
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            engine.async_submit([])


if __name__ == "__main__":
    import unittest

    unittest.main()
