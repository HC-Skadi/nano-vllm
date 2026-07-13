"""Compare the custom Triton attention kernels with FlashAttention."""

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch
from flash_attn import (
    flash_attn_varlen_func as flash_varlen,
    flash_attn_with_kvcache as flash_decode,
)

from nanovllm.layers.attention import (
    flash_attn_varlen_func as triton_varlen,
    flash_attn_with_kvcache as triton_decode,
)


NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 256
DTYPE = torch.bfloat16
SCALE = HEAD_DIM**-0.5


@dataclass
class Result:
    case: str
    backend: str
    latency_ms: float
    max_abs_error: float
    mean_abs_error: float
    exact_percent: float


def benchmark(fn: Callable, warmup: int, repeats: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        output = fn()
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end) / repeats


def error_stats(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float]:
    error = (output.float() - reference.float()).abs()
    exact = output == reference
    return error.max().item(), error.mean().item(), exact.float().mean().item() * 100


def random_q(tokens: int) -> torch.Tensor:
    return torch.randn(tokens, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)


def random_kv(tokens: int, noncontiguous_v: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    if noncontiguous_v:
        storage = torch.randn(tokens, NUM_KV_HEADS * 2, HEAD_DIM, device="cuda", dtype=DTYPE)
        v = storage[:, NUM_KV_HEADS:, :]
    else:
        v = torch.randn_like(k)
    return k, v


def make_cache(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks_per_seq = [math.ceil(length / BLOCK_SIZE) for length in lengths]
    used = sum(blocks_per_seq)
    k_cache = torch.zeros(used + 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    v_cache = torch.zeros_like(k_cache)
    physical = list(reversed(range(used)))
    tables = []
    cursor = 0
    max_blocks = max(blocks_per_seq)
    for length, count in zip(lengths, blocks_per_seq):
        table = physical[cursor:cursor + count]
        cursor += count
        tables.append(table + [-1] * (max_blocks - count))
        k, v = random_kv(length)
        for logical, block in enumerate(table):
            start = logical * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, length)
            k_cache[block, :end - start] = k[start:end]
            v_cache[block, :end - start] = v[start:end]
    return k_cache, v_cache, torch.tensor(tables, device="cuda", dtype=torch.int32)


def compare_case(
    name: str,
    triton_fn: Callable,
    flash_fn: Callable,
    warmup: int,
    repeats: int,
) -> list[Result]:
    flash_out, flash_ms = benchmark(flash_fn, warmup, repeats)
    triton_out, triton_ms = benchmark(triton_fn, warmup, repeats)
    max_err, mean_err, exact = error_stats(triton_out, flash_out)
    return [
        Result(name, "FlashAttention", flash_ms, 0.0, 0.0, 100.0),
        Result(name, "Triton", triton_ms, max_err, mean_err, exact),
    ]


def packed_case(warmup: int, repeats: int) -> list[Result]:
    lengths = [128, 512]
    total = sum(lengths)
    q = random_q(total)
    k, v = random_kv(total, noncontiguous_v=True)
    cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    return compare_case(
        "packed-prefill",
        lambda: triton_varlen(q, k, v, cu, cu, max(lengths), max(lengths), SCALE, True),
        lambda: flash_varlen(q, k, v, cu, cu, max(lengths), max(lengths), softmax_scale=SCALE, causal=True),
        warmup,
        repeats,
    )


def prefix_case(warmup: int, repeats: int) -> list[Result]:
    q_lengths = [32, 64]
    k_lengths = [256, 512]
    q = random_q(sum(q_lengths))
    k, v, table = make_cache(k_lengths)
    cu_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, k_lengths[0], sum(k_lengths)], device="cuda", dtype=torch.int32)
    return compare_case(
        "prefix-prefill",
        lambda: triton_varlen(q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths), SCALE, True, table),
        lambda: flash_varlen(q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths), softmax_scale=SCALE, causal=True, block_table=table),
        warmup,
        repeats,
    )


def decode_case(warmup: int, repeats: int) -> list[Result]:
    lengths = [128, 512]
    q = random_q(len(lengths)).unsqueeze(1)
    k, v, table = make_cache(lengths)
    seqlens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return compare_case(
        "paged-decode",
        lambda: triton_decode(q, k, v, seqlens, table, SCALE, True),
        lambda: flash_decode(q, k, v, cache_seqlens=seqlens, block_table=table, softmax_scale=SCALE, causal=True),
        warmup,
        repeats,
    )


def print_results(results: list[Result]) -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype={DTYPE}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print()
    print(f"{'case':<17} {'backend':<15} {'latency(ms)':>12} {'speedup':>9} {'max error':>12} {'mean error':>12} {'exact':>9}")
    baselines = {r.case: r.latency_ms for r in results if r.backend == "FlashAttention"}
    for result in results:
        speedup = baselines[result.case] / result.latency_ms
        print(
            f"{result.case:<17} {result.backend:<15} {result.latency_ms:>12.4f} "
            f"{speedup:>8.2f}x {result.max_abs_error:>12.6g} "
            f"{result.mean_abs_error:>12.6g} {result.exact_percent:>8.3f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(0)
    results = []
    results += packed_case(args.warmup, args.repeats)
    results += prefix_case(args.warmup, args.repeats)
    results += decode_case(args.warmup, args.repeats)
    print_results(results)


if __name__ == "__main__":
    main()
