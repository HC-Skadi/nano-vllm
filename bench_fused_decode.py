"""Compare separate KV-store + decode against fused KV-store/decode."""

import statistics

import torch

from bench_attention import HEAD_DIM, NUM_Q_HEADS, SCALE, make_cache, random_kv, random_q
from nanovllm.layers.attention import flash_attn_with_kvcache, store_kvcache


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
    print(f"{'B':>3} {'KV':>5} {'separate ms':>12} {'fused ms':>10} {'speedup':>9}")
    for batch, kv_len in [(1, 128), (1, 1024), (8, 1024), (16, 1024), (32, 1024)]:
        q = random_q(batch).unsqueeze(1)
        new_k, new_v = random_kv(batch)
        k_cache, v_cache, table = make_cache([kv_len - 1] * batch)
        seqlens = torch.full((batch,), kv_len, device="cuda", dtype=torch.int32)
        slots = torch.tensor(
            [table[i, (kv_len - 1) // k_cache.shape[1]] * k_cache.shape[1] + (kv_len - 1) % k_cache.shape[1]
             for i in range(batch)],
            device="cuda",
            dtype=torch.int32,
        )

        def separate():
            store_kvcache(new_k, new_v, k_cache, v_cache, slots)
            return flash_attn_with_kvcache(q, k_cache, v_cache, seqlens, table, SCALE, True)

        def fused():
            return flash_attn_with_kvcache(
                q, k_cache, v_cache, seqlens, table, SCALE, True,
                new_k=new_k, new_v=new_v, slot_mapping=slots,
            )

        separate_ms = timed(separate)
        fused_ms = timed(fused)
        print(f"{batch:>3} {kv_len:>5} {separate_ms:>12.4f} {fused_ms:>10.4f} {separate_ms / fused_ms:>8.2f}x")


if __name__ == "__main__":
    main()
