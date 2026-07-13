"""Classic attention benchmark matrix for Triton vs FlashAttention.

Each point uses homogeneous sequence lengths, alternates backend order across
rounds, and reports the median of per-round CUDA-event measurements.
"""

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
from flash_attn import (
    flash_attn_varlen_func as flash_varlen,
    flash_attn_with_kvcache as flash_decode,
)

from bench_attention import (
    HEAD_DIM,
    SCALE,
    error_stats,
    make_cache,
    random_kv,
    random_q,
)
from nanovllm.layers.attention import (
    flash_attn_varlen_func as triton_varlen,
    flash_attn_with_kvcache as triton_decode,
)


@dataclass
class ClassicResult:
    stage: str
    batch: int
    q_len: int
    kv_len: int
    flash_ms: float
    triton_ms: float
    max_error: float
    mean_error: float
    exact_percent: float

    @property
    def speedup(self) -> float:
        return self.flash_ms / self.triton_ms


def timed(fn: Callable, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def compare(
    stage: str,
    batch: int,
    q_len: int,
    kv_len: int,
    triton_fn: Callable,
    flash_fn: Callable,
    warmup: int,
    repeats: int,
    rounds: int,
) -> ClassicResult:
    # Compile both kernels and stabilize GPU clocks before collecting data.
    for _ in range(warmup):
        triton_output = triton_fn()
        flash_output = flash_fn()
    torch.cuda.synchronize()

    triton_times = []
    flash_times = []
    for round_id in range(rounds):
        # Alternate order to reduce cache, clock, and thermal ordering bias.
        if round_id % 2 == 0:
            flash_times.append(timed(flash_fn, repeats))
            triton_times.append(timed(triton_fn, repeats))
        else:
            triton_times.append(timed(triton_fn, repeats))
            flash_times.append(timed(flash_fn, repeats))

    max_error, mean_error, exact = error_stats(triton_output, flash_output)
    return ClassicResult(
        stage,
        batch,
        q_len,
        kv_len,
        statistics.median(flash_times),
        statistics.median(triton_times),
        max_error,
        mean_error,
        exact,
    )


def packed(batch: int, seq_len: int, args) -> ClassicResult:
    total = batch * seq_len
    q = random_q(total)
    k, v = random_kv(total, noncontiguous_v=True)
    cu = torch.arange(0, total + 1, seq_len, device="cuda", dtype=torch.int32)
    return compare(
        "prefill",
        batch,
        seq_len,
        seq_len,
        lambda: triton_varlen(q, k, v, cu, cu, seq_len, seq_len, SCALE, True),
        lambda: flash_varlen(
            q, k, v, cu, cu, seq_len, seq_len,
            softmax_scale=SCALE, causal=True,
        ),
        args.warmup,
        args.repeats,
        args.rounds,
    )


def prefix(batch: int, q_len: int, kv_len: int, args) -> ClassicResult:
    q = random_q(batch * q_len)
    k_cache, v_cache, table = make_cache([kv_len] * batch)
    cu_q = torch.arange(0, batch * q_len + 1, q_len, device="cuda", dtype=torch.int32)
    cu_k = torch.arange(0, batch * kv_len + 1, kv_len, device="cuda", dtype=torch.int32)
    return compare(
        "prefix",
        batch,
        q_len,
        kv_len,
        lambda: triton_varlen(
            q, k_cache, v_cache, cu_q, cu_k, q_len, kv_len,
            SCALE, True, table,
        ),
        lambda: flash_varlen(
            q, k_cache, v_cache, cu_q, cu_k, q_len, kv_len,
            softmax_scale=SCALE, causal=True, block_table=table,
        ),
        args.warmup,
        args.repeats,
        args.rounds,
    )


def decode(batch: int, kv_len: int, args) -> ClassicResult:
    q = random_q(batch).unsqueeze(1)
    k_cache, v_cache, table = make_cache([kv_len] * batch)
    seqlens = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
    return compare(
        "decode",
        batch,
        1,
        kv_len,
        lambda: triton_decode(q, k_cache, v_cache, seqlens, table, SCALE, True),
        lambda: flash_decode(
            q, k_cache, v_cache, cache_seqlens=seqlens,
            block_table=table, softmax_scale=SCALE, causal=True,
        ),
        args.warmup,
        args.repeats,
        args.rounds,
    )


def print_results(results: list[ClassicResult]) -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"head_dim={HEAD_DIM}, dtype=torch.bfloat16; times are median round latency")
    print()
    print(
        f"{'stage':<8} {'B':>3} {'Q':>5} {'KV':>5} "
        f"{'Flash ms':>10} {'Triton ms':>11} {'speedup':>9} "
        f"{'max err':>10} {'mean err':>11} {'exact':>8}"
    )
    for r in results:
        print(
            f"{r.stage:<8} {r.batch:>3} {r.q_len:>5} {r.kv_len:>5} "
            f"{r.flash_ms:>10.4f} {r.triton_ms:>11.4f} {r.speedup:>8.2f}x "
            f"{r.max_error:>10.6g} {r.mean_error:>11.6g} {r.exact_percent:>7.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(0)

    results = []
    # Classic TTFT points: single long prompt and batched short prompts.
    for batch, seq_len in [(1, 128), (1, 512), (1, 1024), (1, 2048), (8, 128)]:
        results.append(packed(batch, seq_len, args))

    # Cached-prefill points: vary the uncached suffix and batch independently.
    for batch, q_len, kv_len in [
        (1, 16, 1024),
        (1, 64, 1024),
        (1, 256, 1024),
        (8, 16, 1024),
    ]:
        results.append(prefix(batch, q_len, kv_len, args))

    # Classic TPOT points: context scaling at B=1 and batch scaling at KV=1024.
    for batch, kv_len in [
        (1, 128),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (8, 1024),
        (16, 1024),
        (32, 1024),
    ]:
        results.append(decode(batch, kv_len, args))

    print_results(results)


if __name__ == "__main__":
    main()
