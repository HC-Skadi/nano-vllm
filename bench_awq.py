"""Microbenchmark fused and unfused vLLM AWQ execution.

This benchmark isolates the AWQ linear operator.  Both paths consume the same
synthetic W4 packed weights, zero-points, scales, and FP16 activations:

* fused: ``vllm._custom_ops.awq_gemm``
* unfused: ``vllm._custom_ops.awq_dequantize`` followed by ``torch.matmul``
* dense (optional): one cached FP16 dequantization followed by ``torch.matmul``

The unfused path intentionally dequantizes on every invocation; the optional
dense path is reported separately because it has a different memory/compute
trade-off.  Per-call CUDA events are collected with rotating execution order,
then median and p90 latency are reported.

The operator signatures and the M=256 dispatch crossover follow vLLM's AWQ
implementation as inspected at:
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py
(file commit 4d591db470c0b53304f9dd2369d4feb8275a94bc).

Example:
    python bench_awq.py --k 2560 --n 6144 --group-size 128 --dense
    python bench_awq.py --json awq-results.json
    python bench_awq.py --json - > awq-results.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
from typing import Callable

import torch


DEFAULT_M_VALUES = (1, 8, 32, 128, 256, 512, 2048)
PACK_FACTOR = 8  # Eight 4-bit values are packed in each int32.
SOURCE_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    "4d591db470c0b53304f9dd2369d4feb8275a94bc/"
    "vllm/model_executor/layers/quantization/awq.py"
)
SOURCE_FILE_COMMIT = "4d591db470c0b53304f9dd2369d4feb8275a94bc"


class BenchmarkUnavailable(RuntimeError):
    """Raised when the required CUDA AWQ backend cannot be used."""


def parse_m_values(values: list[str]) -> list[int]:
    """Accept both ``--m 1 8`` and ``--m 1,8`` forms."""
    parsed: list[int] = []
    for value in values:
        for item in value.split(","):
            try:
                number = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid M value {item!r}"
                ) from exc
            if number <= 0:
                raise argparse.ArgumentTypeError("all M values must be positive")
            if number not in parsed:
                parsed.append(number)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one M value is required")
    return parsed


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
    }


def load_vllm_ops():
    """Import vLLM lazily so ``--help`` works without a vLLM installation."""
    try:
        import vllm
        from vllm import _custom_ops as ops
    except (ImportError, OSError, RuntimeError) as exc:
        raise BenchmarkUnavailable(
            "vLLM with compiled CUDA extensions is required; install a vLLM "
            "wheel compatible with this PyTorch/CUDA environment"
        ) from exc

    missing = [
        name
        for name in ("awq_gemm", "awq_dequantize")
        if not callable(getattr(ops, name, None))
    ]
    if missing:
        raise BenchmarkUnavailable(
            "the installed vLLM does not expose required custom ops: "
            + ", ".join(missing)
        )
    return vllm, ops


def validate_environment(args: argparse.Namespace) -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise BenchmarkUnavailable(
            "CUDA is unavailable (also check CUDA_VISIBLE_DEVICES and container "
            "GPU access)"
        )

    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise BenchmarkUnavailable(
            "vLLM's AWQ CUDA kernel requires SM75 or newer, but device 0 is "
            f"SM{capability[0]}{capability[1]}"
        )

    if args.k <= 0 or args.n <= 0:
        raise ValueError("K and N must be positive")
    if args.n % PACK_FACTOR:
        raise ValueError(f"N must be divisible by the W4 pack factor {PACK_FACTOR}")
    effective_group_size = args.k if args.group_size == -1 else args.group_size
    if effective_group_size <= 0:
        raise ValueError("group size must be positive or -1 (one group per column)")
    if args.k % effective_group_size:
        raise ValueError(
            f"K={args.k} must be divisible by group size {effective_group_size}"
        )
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("iters must be positive")
    return capability


def make_packed_weights(
    k: int,
    n: int,
    group_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create valid-shape random AWQ GEMM tensors on CUDA.

    Uniform random int32 bit patterns are equivalent to independently packed
    random 4-bit lanes, while avoiding a large temporary unpacked tensor.
    """
    packed_n = n // PACK_FACTOR
    num_groups = k // group_size
    int32 = torch.iinfo(torch.int32)
    qweight = torch.randint(
        int32.min,
        int32.max,
        (k, packed_n),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    qzeros = torch.randint(
        int32.min,
        int32.max,
        (num_groups, packed_n),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    scales = torch.empty(
        (num_groups, n), dtype=torch.float16, device="cuda"
    ).uniform_(0.001, 0.05, generator=generator)
    return qweight, qzeros, scales


def collect_timings(
    functions: dict[str, Callable[[], torch.Tensor]],
    warmup: int,
    iters: int,
) -> dict[str, list[float]]:
    """Time functions with CUDA events and rotate order to reduce bias."""
    names = list(functions)
    for iteration in range(warmup):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            functions[name]()
    torch.cuda.synchronize()

    event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    for iteration in range(iters):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            event_pairs[name].append((start, end))

    torch.cuda.synchronize()
    return {
        name: [start.elapsed_time(end) for start, end in pairs]
        for name, pairs in event_pairs.items()
    }


def error_summary(
    actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    error = (actual.float() - reference.float()).abs()
    return {
        "max_abs": error.max().item(),
        "mean_abs": error.mean().item(),
    }


def benchmark_shape(
    m: int,
    k: int,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    ops,
    generator: torch.Generator,
    warmup: int,
    iters: int,
    include_dense: bool,
) -> dict[str, object]:
    x = torch.randn(
        (m, k), dtype=torch.float16, device="cuda", generator=generator
    )

    # Keep this exact argument order in sync with vLLM AWQLinearMethod.
    def fused() -> torch.Tensor:
        return ops.awq_gemm(x, qweight, scales, qzeros, PACK_FACTOR)

    def unfused() -> torch.Tensor:
        weight = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        return torch.matmul(x, weight)

    functions: dict[str, Callable[[], torch.Tensor]] = {
        "fused": fused,
        "unfused": unfused,
    }
    try:
        timings = collect_timings(functions, warmup, iters)
        fused_output = fused()
        unfused_output = unfused()
        dense_output = None
        if include_dense:
            # Allocate and time this baseline only after the primary A/B so it
            # cannot alter their cache residency or execution ordering.
            dense_weight = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)

            def dense() -> torch.Tensor:
                return torch.matmul(x, dense_weight)

            dense_timings = collect_timings(
                {"dense_fp16": dense},
                warmup,
                iters,
            )
            timings.update(dense_timings)
            dense_output = dense()
        torch.cuda.synchronize()
    except (RuntimeError, OSError) as exc:
        capability = torch.cuda.get_device_capability(0)
        raise BenchmarkUnavailable(
            "vLLM AWQ ops were found but failed for "
            f"M={m}, K={k}, N={qweight.shape[1] * PACK_FACTOR}, "
            f"SM{capability[0]}{capability[1]}; check vLLM/PyTorch/CUDA ABI "
            f"compatibility and kernel shape support: {exc}"
        ) from exc

    result: dict[str, object] = {
        "m": m,
        "fused": latency_summary(timings["fused"]),
        "unfused": latency_summary(timings["unfused"]),
        "speedup_unfused_over_fused": (
            statistics.median(timings["unfused"])
            / statistics.median(timings["fused"])
        ),
        "fused_vs_unfused_error": error_summary(fused_output, unfused_output),
    }
    if include_dense and dense_output is not None:
        result["dense_fp16"] = latency_summary(timings["dense_fp16"])
        result["fused_vs_dense_error"] = error_summary(
            fused_output, dense_output
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vLLM fused AWQ GEMM with per-call AWQ dequantize + "
            "torch.matmul using identical synthetic packed weights."
        )
    )
    parser.add_argument(
        "--m",
        "--ms",
        nargs="+",
        default=[str(value) for value in DEFAULT_M_VALUES],
        metavar="M",
        help="token counts, space- or comma-separated (default: %(default)s)",
    )
    parser.add_argument("--k", type=int, default=1024, help="input features")
    parser.add_argument("--n", type=int, default=4096, help="output features")
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="AWQ group size; -1 means K (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="warm-up calls per path"
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="measured calls per path"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="also time cached dequantized FP16 weight + torch.matmul",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="write JSON to PATH; use '-' (or omit PATH) for JSON-only stdout",
    )
    return parser


def print_table(results: list[dict[str, object]], include_dense: bool) -> None:
    header = (
        f"{'M':>6} {'fused med':>11} {'fused p90':>11} "
        f"{'unfused med':>13} {'unfused p90':>13} {'speedup':>9} "
        f"{'max error':>12} {'mean error':>12}"
    )
    if include_dense:
        header += f" {'dense med':>11} {'dense p90':>11}"
    print(header)
    for result in results:
        fused = result["fused"]
        unfused = result["unfused"]
        error = result["fused_vs_unfused_error"]
        assert isinstance(fused, dict)
        assert isinstance(unfused, dict)
        assert isinstance(error, dict)
        line = (
            f"{result['m']:>6} "
            f"{fused['median_ms']:>11.4f} {fused['p90_ms']:>11.4f} "
            f"{unfused['median_ms']:>13.4f} {unfused['p90_ms']:>13.4f} "
            f"{result['speedup_unfused_over_fused']:>8.2f}x "
            f"{error['max_abs']:>12.6g} {error['mean_abs']:>12.6g}"
        )
        if include_dense:
            dense = result["dense_fp16"]
            assert isinstance(dense, dict)
            line += (
                f" {dense['median_ms']:>11.4f} {dense['p90_ms']:>11.4f}"
            )
        print(line)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        m_values = parse_m_values(args.m)
        capability = validate_environment(args)
        vllm, ops = load_vllm_ops()
        effective_group_size = args.k if args.group_size == -1 else args.group_size

        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        with torch.inference_mode():
            qweight, qzeros, scales = make_packed_weights(
                args.k, args.n, effective_group_size, generator
            )
            results = [
                benchmark_shape(
                    m,
                    args.k,
                    qweight,
                    qzeros,
                    scales,
                    ops,
                    generator,
                    args.warmup,
                    args.iters,
                    args.dense,
                )
                for m in m_values
            ]

        payload = {
            "benchmark": "vllm_awq_fused_vs_unfused",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "url": SOURCE_URL,
                "awq_file_commit": SOURCE_FILE_COMMIT,
                "fused": (
                    "vllm._custom_ops.awq_gemm(x, qweight, scales, "
                    "qzeros, 8)"
                ),
                "unfused": (
                    "vllm._custom_ops.awq_dequantize(qweight, scales, "
                    "qzeros, 0, 0, 0) + torch.matmul"
                ),
            },
            "method": {
                "timing": "per-call CUDA events",
                "ordering": "rotated across paths on every iteration",
                "unfused_dequantization": "inside every timed invocation",
                "vllm_auto_dispatch_reference": (
                    "fused for M < 256; unfused for M >= 256; this benchmark "
                    "forces and reports both paths at every M"
                ),
                "dense_dequantization": (
                    "once before timing" if args.dense else "not measured"
                ),
                "latency_statistics": "median and linearly-interpolated p90",
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "vllm": getattr(vllm, "__version__", "unknown"),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": f"{capability[0]}.{capability[1]}",
            },
            "config": {
                "m_values": m_values,
                "k": args.k,
                "n": args.n,
                "group_size": args.group_size,
                "effective_group_size": effective_group_size,
                "bits": 4,
                "pack_factor": PACK_FACTOR,
                "dtype": "float16",
                "warmup": args.warmup,
                "iters": args.iters,
                "dense": args.dense,
                "seed": args.seed,
            },
            "results": results,
        }

        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        if args.json == "-":
            print(serialized)
            return

        print(
            f"GPU: {payload['environment']['gpu']} "
            f"(SM{capability[0]}{capability[1]}); "
            f"torch={torch.__version__}, vllm={payload['environment']['vllm']}"
        )
        print(
            f"W4A16 K={args.k}, N={args.n}, group={effective_group_size}; "
            "unfused includes dequantization on every call; latency is ms"
        )
        print_table(results, args.dense)
        if args.json:
            output_path = Path(args.json)
            output_path.write_text(serialized + "\n", encoding="utf-8")
            print(f"JSON written to {output_path}")
    except (argparse.ArgumentTypeError, BenchmarkUnavailable, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
