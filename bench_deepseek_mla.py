"""Compare DeepSeek-V2 expanded-KV and weight-absorbed MLA backends.

Each backend runs in a fresh child process so checkpoint memory, the CUDA
allocator, and cache allocation from the first run cannot affect the second.

Example:
    python bench_deepseek_mla.py \
        --model /models/DeepSeek-V2-Lite-Chat \
        --requests 8 --input-tokens 512 --output-tokens 128 \
        --repeats 3 --json benchmarks/deepseek_mla_a100.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

import torch


BACKENDS = ("expanded", "latent")


class BenchmarkUnavailable(RuntimeError):
    pass


@dataclass
class RunResult:
    ttft_ms: float
    tpot_ms: float
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    output_tokens_per_second: float
    requests_per_second: float
    elapsed_seconds: float
    peak_memory_gib: float


def make_prompts(
    requests: int,
    input_tokens: int,
    vocab_size: int,
    seed: int,
) -> list[list[int]]:
    generator = random.Random(seed)
    lower = 10 if vocab_size > 32 else 1
    upper = max(lower + 1, vocab_size - 10)
    return [
        [generator.randrange(lower, upper) for _ in range(input_tokens)]
        for _ in range(requests)
    ]


def run_batch(llm, prompts, sampling_params) -> tuple[RunResult, list[list[int]]]:
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prefill_seconds = 0.0
    decode_seconds = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    decode_steps = 0
    completed: dict[int, list[int]] = {}

    while not llm.is_finished():
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        outputs, num_tokens = llm.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        if num_tokens > 0:
            prefill_seconds += step_seconds
            prefill_tokens += num_tokens
        else:
            decode_seconds += step_seconds
            decode_tokens += -num_tokens
            decode_steps += 1
        for sequence_id, token_ids in outputs:
            completed[sequence_id] = token_ids

    elapsed = time.perf_counter() - started
    ordered_outputs = [completed[key] for key in sorted(completed)]
    total_output_tokens = sum(len(tokens) for tokens in ordered_outputs)
    result = RunResult(
        ttft_ms=prefill_seconds * 1000.0,
        tpot_ms=(
            decode_seconds / decode_steps * 1000.0 if decode_steps else 0.0
        ),
        prefill_tokens_per_second=prefill_tokens / prefill_seconds,
        decode_tokens_per_second=(
            decode_tokens / decode_seconds if decode_seconds else 0.0
        ),
        output_tokens_per_second=total_output_tokens / elapsed,
        requests_per_second=len(prompts) / elapsed,
        elapsed_seconds=elapsed,
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    return result, ordered_outputs


def median_runs(runs: list[RunResult]) -> dict[str, float]:
    return {
        field: statistics.median(getattr(run, field) for run in runs)
        for field in RunResult.__dataclass_fields__
    }


def worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise BenchmarkUnavailable(
            "CUDA is unavailable; run inside a GPU-enabled environment"
        )
    if not Path(args.model).is_dir():
        raise ValueError(f"model directory does not exist: {args.model}")

    torch.manual_seed(args.seed)
    from nanovllm import LLM, SamplingParams

    initialization_started = time.perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.input_tokens + args.output_tokens,
        max_num_batched_tokens=args.requests * args.input_tokens,
        max_num_seqs=args.requests,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.cache_blocks,
        trust_remote_code=args.trust_remote_code,
        deepseek_mla_backend=args.backend,
    )
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialization_started

    config = llm.model_runner.config
    vocab_size = int(config.hf_config.vocab_size)
    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    warmup_prompts = make_prompts(
        args.requests,
        args.input_tokens,
        vocab_size,
        args.seed + 10_000,
    )
    run_batch(llm, warmup_prompts, params)

    runs = []
    reference_outputs = None
    for repeat in range(args.repeats):
        prompts = make_prompts(
            args.requests,
            args.input_tokens,
            vocab_size,
            args.seed + repeat,
        )
        result, outputs = run_batch(llm, prompts, params)
        runs.append(result)
        if reference_outputs is None:
            reference_outputs = outputs

    cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for cache_pair in llm.model_runner.kv_cache
        for tensor in cache_pair
    )
    payload = {
        "backend": args.backend,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dtype": str(config.runtime_dtype),
        "initialization_seconds": initialization_seconds,
        "num_kvcache_blocks": config.num_kvcache_blocks,
        "kvcache_block_size": config.kvcache_block_size,
        "cache_bytes": cache_bytes,
        "cache_mib": cache_bytes / 2**20,
        "cache_bytes_per_block": cache_bytes / config.num_kvcache_blocks,
        "median": median_runs(runs),
        "runs": [asdict(run) for run in runs],
        "token_ids": reference_outputs,
    }
    Path(args.result_file).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def token_match(left: list[list[int]], right: list[list[int]]) -> float:
    left_flat = [token for sequence in left for token in sequence]
    right_flat = [token for sequence in right for token in sequence]
    compared = min(len(left_flat), len(right_flat))
    if compared == 0:
        return 100.0 if len(left_flat) == len(right_flat) else 0.0
    matches = sum(
        lhs == rhs
        for lhs, rhs in zip(left_flat[:compared], right_flat[:compared])
    )
    return matches / max(len(left_flat), len(right_flat)) * 100.0


def print_results(results: list[dict]) -> None:
    print(f"GPU: {results[0]['gpu']}")
    print(f"dtype: {results[0]['dtype']}")
    print()
    print(
        f"{'backend':<10} {'TTFT ms':>10} {'TPOT ms':>10} "
        f"{'prefill tok/s':>14} {'decode tok/s':>13} "
        f"{'output tok/s':>13} {'cache MiB':>11} {'blocks':>8}"
    )
    for result in results:
        median = result["median"]
        print(
            f"{result['backend']:<10} "
            f"{median['ttft_ms']:>10.3f} {median['tpot_ms']:>10.3f} "
            f"{median['prefill_tokens_per_second']:>14.1f} "
            f"{median['decode_tokens_per_second']:>13.1f} "
            f"{median['output_tokens_per_second']:>13.1f} "
            f"{result['cache_mib']:>11.1f} "
            f"{result['num_kvcache_blocks']:>8}"
        )

    by_backend = {result["backend"]: result for result in results}
    if set(BACKENDS).issubset(by_backend):
        expanded = by_backend["expanded"]
        latent = by_backend["latent"]
        expanded_median = expanded["median"]
        latent_median = latent["median"]
        print()
        print(
            "latent/expanded decode throughput: "
            f"{latent_median['decode_tokens_per_second'] / expanded_median['decode_tokens_per_second']:.3f}x"
        )
        print(
            "latent/expanded prefill throughput: "
            f"{latent_median['prefill_tokens_per_second'] / expanded_median['prefill_tokens_per_second']:.3f}x"
        )
        print(
            "cache reduction: "
            f"{(1.0 - latent['cache_bytes_per_block'] / expanded['cache_bytes_per_block']) * 100.0:.2f}%"
        )
        print(
            "sampled token match: "
            f"{token_match(expanded['token_ids'], latent['token_ids']):.3f}%"
        )


def coordinator(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise BenchmarkUnavailable(
            "CUDA is unavailable in this environment. The benchmark was not "
            "run; use a GPU-enabled shell to collect BF16 results."
        )
    if args.requests <= 0 or args.input_tokens <= 0 or args.output_tokens <= 0:
        raise ValueError("requests and token counts must be positive")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.cache_blocks == 0:
        blocks_per_sequence = math.ceil(
            (args.input_tokens + args.output_tokens) / 256
        )
        args.cache_blocks = args.requests * blocks_per_sequence + 2

    results = []
    with tempfile.TemporaryDirectory(prefix="deepseek-mla-bench-") as directory:
        for backend in args.backends:
            result_file = Path(directory, f"{backend}.json")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--backend",
                backend,
                "--result-file",
                str(result_file),
                "--model",
                args.model,
                "--requests",
                str(args.requests),
                "--input-tokens",
                str(args.input_tokens),
                "--output-tokens",
                str(args.output_tokens),
                "--repeats",
                str(args.repeats),
                "--cache-blocks",
                str(args.cache_blocks),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--temperature",
                str(args.temperature),
                "--seed",
                str(args.seed),
            ]
            if args.trust_remote_code:
                command.append("--trust-remote-code")
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(
                    f"{backend} worker failed:\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )
            results.append(json.loads(result_file.read_text(encoding="utf-8")))

    print_results(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(args.model).resolve()),
        "requests": args.requests,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "tensor_parallel_size": args.tensor_parallel_size,
        "cache_blocks": args.cache_blocks,
        "results": results,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=BACKENDS
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-blocks", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.worker:
            worker(args)
        else:
            coordinator(args)
    except (BenchmarkUnavailable, ValueError, RuntimeError) as exc:
        raise SystemExit(f"benchmark unavailable: {exc}") from exc


if __name__ == "__main__":
    main()
