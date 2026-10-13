import itertools
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple
from unittest import SkipTest, TestCase, main

import torch

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import per_tensor_quant_fp8  # isort:skip


class PerTensorFp8QuantFusedTest(TestCase):
    # Keep in sync with small_input_limit in scaled_fp8_quant.cu.
    SINGLE_CTA_LIMIT = 8192
    NUM_TOKENS = [128, 256, 512]
    HIDDEN_SIZES = [128, 768, 769, 770, 5120, 8192]
    INPUT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    # Allow FP32 division/reciprocal rounding differences; zero remains exact.
    SCALE_RTOL = 2 * torch.finfo(torch.float32).eps

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.manual_seed(2026)

    def torch_scaled_fp8_quant(
        self, tensor: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        multiplier = scale.clamp_min(1e-9).reciprocal()
        return (
            (tensor.float() * multiplier)
            .clamp(min=finfo.min, max=finfo.max)
            .to(torch.float8_e4m3fn)
        )

    def call_per_tensor_quant_fp8(
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        is_static = scale is not None
        if scale is None:
            # Dynamic quantization must initialize its own scale.
            scale = torch.full(
                (1,), float("nan"), device=input.device, dtype=torch.float32
            )
        per_tensor_quant_fp8(input, output, scale, is_static)
        return output, scale

    def assert_dynamic_scale(self, input: torch.Tensor, scale: torch.Tensor) -> None:
        expected = input.detach().cpu().float().abs().max().reshape(1) / 448.0
        torch.testing.assert_close(
            scale.detach().cpu().reshape(1),
            expected,
            rtol=self.SCALE_RTOL,
            atol=0,
        )

    def assert_quantized_output(
        self, input: torch.Tensor, output: torch.Tensor, scale: torch.Tensor
    ) -> None:
        # Use the returned scale, matching the original test reference.
        expected = self.torch_scaled_fp8_quant(input, scale)
        torch.testing.assert_close(
            output.contiguous().view(torch.uint8).cpu(),
            expected.contiguous().view(torch.uint8).cpu(),
            rtol=0,
            atol=0,
            msg="FP8 output bytes differ",
        )

    def test_profiler_kernel_performance(self):
        """Compare original vs current op using Chrome trace GPU durations.

        Excludes allocation, warmup, capture, CPU launch time and GPU gaps.
        Includes baseline scale fill and any fallback scale memset.
        """
        if os.environ.get("FP8_QUANT_PERF", "1") == "0":
            self.skipTest("FP8_QUANT_PERF=0")
        from rtp_llm.ops.compute_ops import per_tensor_quant_fp8_baseline

        iterations = int(os.environ.get("FP8_QUANT_PERF_ITERS", "100"))
        rounds = int(os.environ.get("FP8_QUANT_PERF_ROUNDS", "3"))
        self.assertGreater(iterations, 0)
        self.assertGreater(rounds, 0)
        artifact_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        temporary = tempfile.TemporaryDirectory(prefix="fp8-quant-perf-")
        output_dir = Path(artifact_dir or temporary.name) / "fp8_quant_perf"
        output_dir.mkdir(parents=True, exist_ok=True)
        shapes = [
            (8, 1024),
            (16, 1024),
            (32, 1024),
            (8, 2048),
            (16, 2048),
            (32, 2048),
        ]
        results = []
        print("\nFP8 profiler performance: BF16, no torch.compile, allocations/capture/warmup excluded")
        print("shape | mode | baseline GPU us | current GPU us | GPU speedup | baseline event us | current event us")
        try:
            for shape in shapes:
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                outputs = {name: torch.empty_like(x, dtype=torch.float8_e4m3fn)
                           for name in ("baseline", "current")}
                scales = {name: torch.empty(1, device="cuda")
                          for name in ("baseline", "current")}

                def baseline():
                    scales["baseline"].zero_()
                    per_tensor_quant_fp8_baseline(x, outputs["baseline"], scales["baseline"], False)

                def current():
                    per_tensor_quant_fp8(x, outputs["current"], scales["current"], False)

                calls = {"baseline": baseline, "current": current}
                for name, call in calls.items():
                    call()
                    self.assert_dynamic_scale(x, scales[name])
                    self.assert_quantized_output(x, outputs[name], scales[name])
                torch.testing.assert_close(scales["current"], scales["baseline"],
                                           rtol=self.SCALE_RTOL, atol=0)
                for mode in ("direct", "graph"):
                    measured = calls.copy()
                    graphs = {}
                    warm_stream = torch.cuda.Stream()
                    warm_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warm_stream):
                        for _ in range(30):
                            for call in calls.values():
                                call()
                    torch.cuda.current_stream().wait_stream(warm_stream)
                    torch.cuda.synchronize()
                    if mode == "graph":
                        for name, call in calls.items():
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                call()
                            graphs[name] = graph
                            measured[name] = graph.replay
                    # CUDA events measure stream latency, including launch gaps.
                    # Keep this separate from profiler sums of actual GPU work.
                    event_samples = {"baseline": [], "current": []}
                    for repeat in range(rounds):
                        order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                        for name in order:
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            start.record()
                            for _ in range(iterations):
                                measured[name]()
                            end.record()
                            end.synchronize()
                            event_samples[name].append(start.elapsed_time(end) * 1000 / iterations)
                    samples = {"baseline": [], "current": []}
                    for repeat in range(rounds):
                        # Alternate order to reduce ordering bias.
                        order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                        for name in order:
                            torch.cuda.synchronize()
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                                for _ in range(iterations):
                                    measured[name]()
                                torch.cuda.synchronize()
                            trace = output_dir / f"{shape[0]}x{shape[1]}_{mode}_{name}_{repeat}.json"
                            prof.export_chrome_trace(str(trace))
                            events = json.loads(trace.read_text())["traceEvents"]
                            gpu = [e for e in events if e.get("ph") == "X"
                                   and e.get("cat") in ("kernel", "gpu_memset")]
                            kernels = [e for e in gpu if e["cat"] == "kernel"]
                            memsets = [e for e in gpu if e["cat"] == "gpu_memset"]
                            fused = name == "current" and x.numel() <= self.SINGLE_CTA_LIMIT
                            expected_kernels = 3 if name == "baseline" else (1 if fused else 2)
                            self.assertEqual(len(kernels), iterations * expected_kernels,
                                             f"Unexpected launches in {trace}")
                            self.assertEqual(len(memsets), 0, f"Unexpected scale memset in {trace}")
                            for e in kernels:
                                self.assertTrue("per_tensor_" in e["name"]
                                                or (name == "baseline" and "FillFunctor<float>" in e["name"]),
                                                f"Unexpected kernel: {e['name']}")
                            families = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
                            for e in gpu:
                                families[e["name"]]["count"] += 1
                                families[e["name"]]["sum_us"] += e["dur"]
                            samples[name].append({
                                "kernel_sum_us_per_iter": sum(e["dur"] for e in kernels) / iterations,
                                "memset_sum_us_per_iter": sum(e["dur"] for e in memsets) / iterations,
                                "gpu_sum_us_per_iter": sum(e["dur"] for e in gpu) / iterations,
                                "kernel_count_per_iter": len(kernels) / iterations,
                                "memset_count_per_iter": len(memsets) / iterations,
                                "families": dict(families), "trace": trace.name,
                            })
                    stats = {}
                    for name, values in samples.items():
                        times = sorted(v["gpu_sum_us_per_iter"] for v in values)
                        stats[name] = {"median_us": statistics.median(times),
                                       "min_us": min(times), "max_us": max(times),
                                       "kernel_median_us": statistics.median(
                                           v["kernel_sum_us_per_iter"] for v in values),
                                       "memset_median_us": statistics.median(
                                           v["memset_sum_us_per_iter"] for v in values),
                                       "event_median_us": statistics.median(event_samples[name]),
                                       "event_min_us": min(event_samples[name]),
                                       "event_max_us": max(event_samples[name]),
                                       "event_samples_us": event_samples[name],
                                       "launch_validation": "passed",
                                       "samples": values}
                    speedup = stats["baseline"]["median_us"] / stats["current"]["median_us"]
                    results.append({"shape": shape, "mode": mode, "statistics": stats,
                                    "speedup": speedup,
                                    "event_speedup": stats["baseline"]["event_median_us"] / stats["current"]["event_median_us"]})
                    print(f"{shape} | {mode} | {stats['baseline']['median_us']:.3f} | "
                          f"{stats['current']['median_us']:.3f} | {speedup:.3f}x | "
                          f"{stats['baseline']['event_median_us']:.3f} | "
                          f"{stats['current']['event_median_us']:.3f}", flush=True)
                    for name in ("baseline", "current"):
                        family_totals = defaultdict(lambda: {"count": 0, "sum_us": 0.0})
                        for sample in samples[name]:
                            for family, info in sample["families"].items():
                                family_totals[family]["count"] += info["count"]
                                family_totals[family]["sum_us"] += info["sum_us"]
                        for family, info in family_totals.items():
                            print(f"  {name}: {info['sum_us'] / (rounds * iterations):.3f} us/call, "
                                  f"{info['count'] / (rounds * iterations):.1f} launches/call: {family}",
                                  flush=True)
            summary = {"timing": "Chrome trace dur (us), kernel + gpu_memset; no gaps",
                       "baseline": "382e155932 original kernels + caller scale.zero_()",
                       "candidate": f"register-cached single CTA <={self.SINGLE_CTA_LIMIT}; otherwise partial-scale absmax + per-token quant; no memset",
                       "single_cta_limit": self.SINGLE_CTA_LIMIT,
                       "event_timing": "CUDA event stream latency (us/call), includes launch gaps",
                       "speedup_metric": "median kernel + memset duration; includes required scale initialization",
                       "device": torch.cuda.get_device_name(x.device),
                       "dtype": "bfloat16", "iterations": iterations, "rounds": rounds,
                       "warmup": 30, "torch_compile": False, "results": results}
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"FP8 performance artifacts: {output_dir}", flush=True)
            if not artifact_dir:
                print("Set TEST_UNDECLARED_OUTPUTS_DIR to retain traces and summary outside Bazel.")
        finally:
            temporary.cleanup()

    def test_per_tensor_fp8_quant(self):
        for num_tokens, hidden_size, dtype in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.INPUT_DTYPES
        ):
            with self.subTest(
                num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype
            ):
                x = torch.randn(
                    (num_tokens, hidden_size), device="cuda", dtype=dtype
                )
                output, scale = self.call_per_tensor_quant_fp8(x)
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)
                static_scale = torch.tensor([0.1], device="cuda")
                original_scale = static_scale.clone()
                output, scale = self.call_per_tensor_quant_fp8(x, static_scale)
                torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
                self.assert_quantized_output(x, output, scale)

    def test_dynamic_small_inputs_and_vector_tail(self):
        # All lengths satisfy the binding's 16-byte input alignment constraint;
        # 8/24 BF16 or FP16 and 4/12 FP32 also exercise the quantizer's tail.
        for dtype in self.INPUT_DTYPES:
            alignment = 16 // torch.empty((), dtype=dtype).element_size()
            for length in (alignment, 3 * alignment, 264):
                with self.subTest(dtype=dtype, length=length):
                    x = torch.zeros(length, device="cuda", dtype=dtype)
                    # Ensure the maximum is in the tail, not the vector prefix.
                    x[-1] = -7.0
                    output, scale = self.call_per_tensor_quant_fp8(x)
                    self.assert_dynamic_scale(x, scale)
                    self.assert_quantized_output(x, output, scale)

    def test_dynamic_unaligned_inputs_and_odd_lengths(self):
        for dtype, length in itertools.product(self.INPUT_DTYPES, (1, 257, self.SINGLE_CTA_LIMIT - 1, self.SINGLE_CTA_LIMIT + 1, 32767, 32769, 65537)):
            with self.subTest(dtype=dtype, length=length):
                storage = torch.full((length + 1,), 0.125, device="cuda", dtype=dtype)
                x = storage[1:]
                x[-1] = -32.0
                output, scale = self.call_per_tensor_quant_fp8(x)
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)

    def test_register_cached_input_tiles(self):
        # Put the maximum in each cached tile in turn, including its last lane.
        for dtype in self.INPUT_DTYPES:
            vector_size = 16 // torch.empty((), dtype=dtype).element_size()
            tile_size = 256 * vector_size
            x = torch.empty(self.SINGLE_CTA_LIMIT, device="cuda", dtype=dtype)
            for position in range(tile_size - 1, x.numel(), tile_size):
                with self.subTest(dtype=dtype, position=position):
                    x.fill_(0.125)
                    x[position] = -32.0
                    output, scale = self.call_per_tensor_quant_fp8(x)
                    self.assert_dynamic_scale(x, scale)
                    self.assert_quantized_output(x, output, scale)

    def test_dynamic_scale_reciprocal_rounding(self):
        x = torch.full((8,), 255 / 256, device="cuda", dtype=torch.bfloat16)
        output, scale = self.call_per_tensor_quant_fp8(x)
        self.assert_dynamic_scale(x, scale)
        self.assert_quantized_output(x, output, scale)

    def test_dynamic_scale_overwrites_previous_value(self):
        x = torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.full((1,), float("nan"), device="cuda")
        # Exercise repeated calls with decreasing, zero, and increasing inputs.
        for magnitude in (16.0, 0.25, 0.0, 8.0):
            with self.subTest(magnitude=magnitude):
                value = x * magnitude
                per_tensor_quant_fp8(value, output, scale, False)
                self.assert_dynamic_scale(value, scale)
                self.assert_quantized_output(value, output, scale)

    def test_large_dynamic_scale_overwrites_previous_value(self):
        x = torch.ones((33, 1024), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.full((1,), float("nan"), device="cuda")
        for magnitude in (32.0, 0.125, 0.0, -8.0):
            with self.subTest(magnitude=magnitude):
                x.fill_(magnitude)
                per_tensor_quant_fp8(x, output, scale, False)
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)

    def test_per_tensor_fp8_static_quant_precision(self):
        values = [
            -500,
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            0.0,
            0.5,
            1.5,
            2.5,
            3.5,
            500,
            2**-10,
            2**-9,
            3 * 2**-10,
            -2**-9,
            448,
        ]
        for dtype, scale_value in itertools.product(
            self.INPUT_DTYPES, (0.1, 2.5, 22.5, 222.0, 5012.0)
        ):
            with self.subTest(dtype=dtype, scale=scale_value):
                # Bound FP16 inputs to finite values while preserving saturation
                # coverage through the other input dtypes and smaller scales.
                x = torch.tensor(values, device="cuda", dtype=torch.float32)
                x = x * scale_value
                if dtype == torch.float16:
                    x = x.clamp(-65504, 65504)
                x = x.to(dtype)
                scale = torch.tensor([scale_value], device="cuda")
                original_scale = scale.clone()
                output, scale = self.call_per_tensor_quant_fp8(x, scale)
                torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
                self.assert_quantized_output(x, output, scale)

        x = torch.tensor([2**31, -(2**31), 0, 1], device="cuda", dtype=torch.float32)
        scale = torch.ones(1, device="cuda")
        output, scale = self.call_per_tensor_quant_fp8(x, scale)
        self.assert_quantized_output(x, output, scale)

    def test_static_scale_safety_clamp(self):
        x = torch.tensor(
            [0, 1e-10, -1e-10, 1e-9, -1e-9, 1, -1, 2],
            device="cuda",
            dtype=torch.float32,
        )
        for scale_value in (0.0, 1e-12, 1e-9):
            with self.subTest(scale=scale_value):
                scale = torch.tensor([scale_value], device="cuda")
                original_scale = scale.clone()
                output, scale = self.call_per_tensor_quant_fp8(x, scale)
                torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
                self.assert_quantized_output(x, output, scale)

    def test_quant_dispatch_boundaries(self):
        for dtype in self.INPUT_DTYPES:
            alignment = 16 // torch.empty((), dtype=dtype).element_size()
            for boundary in (self.SINGLE_CTA_LIMIT, 32768):
                for length in (boundary - alignment, boundary, boundary + alignment):
                    with self.subTest(dtype=dtype, length=length):
                        x = torch.full((length,), 0.125, device="cuda", dtype=dtype)
                        x[-1] = -32.0
                        output, scale = self.call_per_tensor_quant_fp8(x)
                        self.assert_dynamic_scale(x, scale)
                        self.assert_quantized_output(x, output, scale)

    def test_hybrid_row_and_flat_paths(self):
        for dtype in self.INPUT_DTYPES:
            for shape in ((32, 1024), (33, 1024), (64, 1024), (64, 769)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.full(shape, 0.125, device="cuda", dtype=dtype)
                    x[-1, -1] = -32.0
                    output, scale = self.call_per_tensor_quant_fp8(x)
                    self.assert_dynamic_scale(x, scale)
                    self.assert_quantized_output(x, output, scale)
                    static_scale = torch.tensor([0.1], device="cuda")
                    output, scale = self.call_per_tensor_quant_fp8(x, static_scale)
                    self.assert_quantized_output(x, output, scale)

    def test_dynamic_global_max_at_different_positions(self):
        # Exercise maxima owned by different blocks and grid-stride iterations.
        for dtype in self.INPUT_DTYPES:
            x = torch.empty(1024 * 1024 + 24, device="cuda", dtype=dtype)
            for position in (0, 256, 8192, x.numel() // 2, x.numel() - 1):
                with self.subTest(dtype=dtype, position=position):
                    x.fill_(0.125)
                    x[position] = -32.0
                    output, scale = self.call_per_tensor_quant_fp8(x)
                    self.assert_dynamic_scale(x, scale)
                    self.assert_quantized_output(x, output, scale)

    def test_dynamic_large_cuda_graph_replay(self):
        # Large inputs exceed the resident grid and must use grid-stride loops.
        x = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(1, device="cuda")
        for _ in range(3):
            per_tensor_quant_fp8(x, output, scale, False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            per_tensor_quant_fp8(x, output, scale, False)
        for position, magnitude in ((0, 32.0), (x.numel() - 1, -8.0), (256, 0.5)):
            with self.subTest(position=position, magnitude=magnitude):
                x.zero_()
                x.view(-1)[position] = magnitude
                graph.replay()
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)

    def test_dynamic_scale_cuda_graph_replay(self):
        x = torch.ones((8, 1024), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(1, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                per_tensor_quant_fp8(x, output, scale, False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            per_tensor_quant_fp8(x, output, scale, False)
        for magnitude in (8.0, 0.125, 0.0, -16.0):
            with self.subTest(magnitude=magnitude):
                x.fill_(magnitude)
                graph.replay()
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)


if __name__ == "__main__":
    main()
