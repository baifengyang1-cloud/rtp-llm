"""Boundary regressions runnable without loading the inference engine libraries."""

import ast
import asyncio
import logging
import importlib.util
import struct
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest

from pydantic import BaseModel
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_nodes(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    selected = [n for n in nodes if getattr(n, "name", None) in names]
    assert len(selected) == len(names)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )


class CompactLogprobsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = dict(
            torch=torch,
            Dict=Dict,
            List=List,
            Optional=Optional,
            BaseModel=BaseModel,
            logging=logging,
        )
        load_nodes(
            "rtp_llm/openai/api_datatype.py",
            {"TopLogprob", "ChatCompletionTokenLogprob"},
            ns,
        )
        cls.ns = dict(
            ns, GenerateOutput=object, StreamStatus=object, StreamStatusSync=object
        )
        load_nodes(
            "rtp_llm/openai/renderers/custom_renderer.py",
            {
                "_generate_compact_log_probs",
                "_generate_log_probs",
                "_generate_log_probs_sync",
            },
            cls.ns,
            class_name="CustomChatRenderer",
        )
        renderer_type = type(
            "Renderer",
            (),
            {
                n: cls.ns[n]
                for n in (
                    "_generate_compact_log_probs",
                    "_generate_log_probs",
                    "_generate_log_probs_sync",
                )
            },
        )
        cls.renderer = renderer_type()
        cls.renderer.tokenizer = SimpleNamespace(
            decode=lambda ids: "词" + str(ids[0] if isinstance(ids, list) else ids)
        )

    def test_compact_matches_dense_for_async_and_sync_consumers(self):
        for probs, selected, k in [
            ([0.1, 0.6, 0.3, 0.0], 0, 2),
            ([0.0, 1.0, 0.0], 1, 3),
            ([0.25, 0.75], 0, 1),
        ]:
            with self.subTest(probs=probs):
                p = torch.tensor(probs)
                values, ids = p.topk(k)
                compact = {
                    "token_ids": torch.cat((torch.tensor([selected]), ids)),
                    "logprobs": torch.cat((p[selected : selected + 1], values)).log(),
                }
                status = SimpleNamespace(
                    request=SimpleNamespace(logprobs=True, top_logprobs=k)
                )
                token = torch.tensor([selected])
                dense = self.renderer._generate_log_probs_sync(status, p, token)
                result = self.renderer._generate_log_probs_sync(status, compact, token)
                self.assertEqual(dense.model_dump(), result.model_dump())
                output = SimpleNamespace(
                    all_probs=None, compact_logprobs=compact, output_ids=token
                )
                result = asyncio.run(self.renderer._generate_log_probs(status, output))
                self.assertEqual(dense.model_dump(), result.model_dump())
                self.assertEqual(result.bytes, list(result.token.encode("utf-8")))

    def test_mismatched_selected_token_fails_instead_of_reporting_wrong_probability(
        self,
    ):
        status = SimpleNamespace(request=SimpleNamespace(logprobs=True, top_logprobs=1))
        compact = {
            "token_ids": torch.tensor([1, 2]),
            "logprobs": torch.tensor([-1.0, -2.0]),
        }
        with self.assertRaises(ValueError):
            self.renderer._generate_compact_log_probs(
                status, compact, torch.tensor([3])
            )

    def test_disabled_logprobs_does_not_require_probability_tensors(self):
        status = SimpleNamespace(request=SimpleNamespace(logprobs=False))
        self.assertIsNone(self.renderer._generate_log_probs_sync(status, None, None))

    def test_request_negotiation_preserves_explicit_dense_and_opt_out(self):
        source = ast.parse((ROOT / "rtp_llm/openai/openai_endpoint.py").read_text())
        cls = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "OpenaiEndpoint"
        )
        method = next(
            n
            for n in cls.body
            if getattr(n, "name", "") == "_extract_generation_config"
        )
        node = next(
            n
            for n in method.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "request.logprobs != None"
        )
        code = compile(
            ast.Module(body=[node], type_ignores=[]), "request negotiation", "exec"
        )
        modes = SimpleNamespace(NONE=0, DEFAULT=1, ORIGINAL=2)
        for enabled, mode, fields, old_k, expected_mode, expected_k in [
            (True, 0, set(), -1, 1, 5),
            (True, 0, {"logprobs_top_k"}, -1, 1, -1),
            (True, 2, {"return_all_probs"}, -1, 2, -1),
            (False, 2, set(), 5, 0, -1),
            (None, 0, set(), -1, 0, -1),
        ]:
            config = SimpleNamespace(
                return_all_probs=mode, logprobs_top_k=old_k, model_fields_set=fields
            )
            request = SimpleNamespace(
                logprobs=enabled, logprobs_mode=None, top_logprobs=5
            )
            exec(code, dict(config=config, request=request, ReturnAllProbsMode=modes))
            self.assertEqual(config.return_all_probs, expected_mode)
            self.assertEqual(config.logprobs_top_k, expected_k)

    def test_rpc_roundtrip_and_legacy_dense_fallback(self):
        generated = (
            ROOT.parent
            / "build_logs/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py"
        )
        if generated.exists():
            spec = importlib.util.spec_from_file_location("compact_test_pb", generated)
            pb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pb)
        else:
            from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
        ns = dict(
            torch=torch,
            TensorPB=pb.TensorPB,
            Dict=Dict,
            List=List,
            Optional=Optional,
            Any=Any,
            dataclass=dataclass,
            field=field,
            RoleAddr=object,
            GENERATION_PREFILL_CUDA_GRAPH_STATUS_NOT_REQUESTED="not_requested",
            logging=logging,
            GenerateInput=object,
            GenerateOutputsPB=pb.GenerateOutputsPB,
            StreamState=object,
        )
        load_nodes(
            "rtp_llm/utils/grpc_util.py", {"trans_tensor", "trans_grpc_dtype"}, ns
        )
        load_nodes(
            "rtp_llm/utils/base_model_datatypes.py",
            {"AuxInfo", "GenerateOutput", "GenerateOutputs"},
            ns,
        )
        load_nodes("rtp_llm/cpp/model_rpc/model_rpc_client.py", {"trans_output"}, ns)
        inp = SimpleNamespace(
            token_ids=torch.tensor([4, 5]),
            generate_config=SimpleNamespace(
                logits_index=None, aux_info=True, role_addrs=[], calculate_loss=0
            ),
        )
        for compact in [False, True]:
            message = pb.GenerateOutputsPB()
            out = message.flatten_output
            out.finished.append(False)
            out.output_ids.data_type = pb.TensorPB.INT32
            out.output_ids.shape.extend([1, 1, 1])
            out.output_ids.int32_data = struct.pack("<i", 2)
            aux = out.aux_info.add()
            if compact:
                aux.logprob_token_ids.data_type = pb.TensorPB.INT32
                aux.logprob_token_ids.shape.extend([1, 2])
                aux.logprob_token_ids.int32_data = struct.pack("<ii", 2, 1)
                aux.logprob_values.shape.extend([1, 2])
                aux.logprob_values.fp32_data = struct.pack("<ff", -1.0, -0.5)
            else:
                out.all_probs.shape.extend([1, 1, 4])
                out.all_probs.fp32_data = struct.pack("<ffff", 0.1, 0.6, 0.3, 0.0)
            received = pb.GenerateOutputsPB.FromString(message.SerializeToString())
            result = ns["trans_output"](
                inp, received, SimpleNamespace(cached_logits_dict={})
            ).generate_outputs[0]
            if compact:
                self.assertIsNone(result.all_probs)
                self.assertEqual(
                    result.compact_logprobs["token_ids"].tolist(), [[2, 1]]
                )
                self.assertEqual(
                    result.compact_logprobs["logprobs"].tolist(), [[-1.0, -0.5]]
                )
                self.assertFalse(received.flatten_output.HasField("all_probs"))
            else:
                self.assertIsNone(result.compact_logprobs)
                self.assertTrue(
                    torch.allclose(
                        result.all_probs, torch.tensor([[0.1, 0.6, 0.3, 0.0]])
                    )
                )
        config = pb.GenerateConfigPB()
        self.assertFalse(config.HasField("logprobs_top_k"))
        config.logprobs_top_k.value = 0
        self.assertTrue(
            pb.GenerateConfigPB.FromString(config.SerializeToString()).HasField(
                "logprobs_top_k"
            )
        )


if __name__ == "__main__":
    unittest.main()
