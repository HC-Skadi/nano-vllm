"""Tune BLOCK_N and num_warps for the M=1 vector decode kernel."""

import statistics

import torch

from bench_attention import HEAD_DIM, NUM_Q_HEADS, SCALE, make_cache, random_q
from nanovllm.layers.attention import flash_attn_decode_vector_kernel


def timed(fn, repeats=100, rounds=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    return statistics.median(values)


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'B':>3} {'KV':>5} {'BLOCK_N':>8} {'warps':>5} {'ms':>9}")
    for batch, kv_len in [(1, 128), (1, 512), (1, 1024), (1, 2048), (8, 1024)]:
        q = random_q(batch).unsqueeze(1)
        k, v, table = make_cache([kv_len] * batch)
        seqlens = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
        out = torch.empty(batch, NUM_Q_HEADS, HEAD_DIM, device="cuda", dtype=q.dtype)
        for block_n in (16, 32, 64):
            for warps in (4, 8):
                fn = lambda bn=block_n, nw=warps: flash_attn_decode_vector_kernel[
                    (batch * NUM_Q_HEADS,)
                ](
                    q[:, 0], k, v, out, seqlens, table,
                    NUM_Q_HEADS, k.shape[2], table.shape[1], k.shape[1],
                    HEAD_DIM, SCALE, BLOCK_N=bn,
                    BLOCK_D=HEAD_DIM, num_warps=nw,
                )
                print(f"{batch:>3} {kv_len:>5} {block_n:>8} {warps:>5} {timed(fn):>9.4f}")


if __name__ == "__main__":
    main()
