import math
import unittest

import torch
from flash_attn import (
    flash_attn_varlen_func as reference_varlen,
    flash_attn_with_kvcache as reference_decode,
)

from nanovllm.layers.attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    store_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonAttentionTest(unittest.TestCase):
    dtype = torch.bfloat16
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    block_size = 256

    def setUp(self):
        torch.manual_seed(0)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    def random_q(self, tokens):
        return torch.randn(
            tokens, self.num_q_heads, self.head_dim,
            device="cuda", dtype=self.dtype,
        )

    def random_kv(self, tokens):
        shape = (tokens, self.num_kv_heads, self.head_dim)
        return (
            torch.randn(shape, device="cuda", dtype=self.dtype),
            torch.randn(shape, device="cuda", dtype=self.dtype),
        )

    def make_paged_cache(self, lengths):
        blocks_per_sequence = [math.ceil(length / self.block_size) for length in lengths]
        total_used_blocks = sum(blocks_per_sequence)
        num_blocks = total_used_blocks + 2
        k_cache = torch.zeros(
            num_blocks, self.block_size, self.num_kv_heads, self.head_dim,
            device="cuda", dtype=self.dtype,
        )
        v_cache = torch.zeros_like(k_cache)
        physical_blocks = list(reversed(range(total_used_blocks)))
        max_blocks = max(blocks_per_sequence)
        tables = []
        cursor = 0
        for batch_id, length in enumerate(lengths):
            k, v = self.random_kv(length)
            table = physical_blocks[cursor:cursor + blocks_per_sequence[batch_id]]
            cursor += blocks_per_sequence[batch_id]
            tables.append(table + [-1] * (max_blocks - len(table)))
            for logical_block, physical_block in enumerate(table):
                start = logical_block * self.block_size
                end = min(start + self.block_size, length)
                k_cache[physical_block, :end - start] = k[start:end]
                v_cache[physical_block, :end - start] = v[start:end]
        block_table = torch.tensor(tables, device="cuda", dtype=torch.int32)
        return k_cache, v_cache, block_table

    def test_varlen_packed_gqa_bf16(self):
        lengths = [19, 37]
        total = sum(lengths)
        q = self.random_q(total)
        k, _ = self.random_kv(total)
        # Match the framework's qkv.split(...).view(...) layout: V has a
        # larger token stride even though head_dim remains contiguous.
        v_storage = torch.randn(
            total, self.num_kv_heads * 2, self.head_dim,
            device="cuda", dtype=self.dtype,
        )
        v = v_storage[:, self.num_kv_heads:, :]
        self.assertFalse(v.is_contiguous())
        cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)

        actual = flash_attn_varlen_func(
            q, k, v, cu, cu, max(lengths), max(lengths), self.scale, True,
        )
        expected = reference_varlen(
            q, k, v, cu, cu, max(lengths), max(lengths),
            softmax_scale=self.scale, causal=True,
        )
        self.assert_close(actual, expected)

    def test_varlen_prefix_cache_gqa_bf16(self):
        q_lengths = [7, 5]
        k_lengths = [19, 37]
        q = self.random_q(sum(q_lengths))
        k_cache, v_cache, block_table = self.make_paged_cache(k_lengths)
        cu_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, k_lengths[0], sum(k_lengths)], device="cuda", dtype=torch.int32)

        actual = flash_attn_varlen_func(
            q, k_cache, v_cache, cu_q, cu_k,
            max(q_lengths), max(k_lengths), self.scale, True, block_table,
        )
        expected = reference_varlen(
            q, k_cache, v_cache, cu_q, cu_k,
            max(q_lengths), max(k_lengths), softmax_scale=self.scale,
            causal=True, block_table=block_table,
        )
        self.assert_close(actual, expected)

    def test_decode_paged_gqa_bf16(self):
        lengths = [19, 300]
        q = self.random_q(len(lengths)).unsqueeze(1)
        k_cache, v_cache, block_table = self.make_paged_cache(lengths)
        cache_seqlens = torch.tensor(lengths, device="cuda", dtype=torch.int32)

        actual = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens, block_table, self.scale, True,
        )
        expected = reference_decode(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens,
            block_table=block_table, softmax_scale=self.scale, causal=True,
        )
        self.assert_close(actual, expected)

    def test_store_kvcache_then_decode(self):
        lengths = [19, 37]
        q = self.random_q(len(lengths)).unsqueeze(1)
        k_cache, v_cache, block_table = self.make_paged_cache(lengths)
        new_k, new_v = self.random_kv(len(lengths))
        slots = torch.tensor(
            [block_table[i, 0] * self.block_size + lengths[i] for i in range(len(lengths))],
            device="cuda", dtype=torch.int32,
        )
        store_kvcache(new_k, new_v, k_cache, v_cache, slots)
        updated_lengths = torch.tensor([x + 1 for x in lengths], device="cuda", dtype=torch.int32)

        actual = flash_attn_with_kvcache(
            q, k_cache, v_cache, updated_lengths, block_table, self.scale, True,
        )
        expected = reference_decode(
            q, k_cache, v_cache, cache_seqlens=updated_lengths,
            block_table=block_table, softmax_scale=self.scale, causal=True,
        )
        self.assert_close(actual, expected)

    def test_fused_store_kvcache_and_decode(self):
        old_lengths = [18, 36]
        q = self.random_q(len(old_lengths)).unsqueeze(1)
        k_cache, v_cache, block_table = self.make_paged_cache(old_lengths)
        reference_k_cache = k_cache.clone()
        reference_v_cache = v_cache.clone()
        new_k, _ = self.random_kv(len(old_lengths))
        v_storage = torch.randn(
            len(old_lengths), self.num_kv_heads * 2, self.head_dim,
            device="cuda", dtype=self.dtype,
        )
        new_v = v_storage[:, self.num_kv_heads:, :]
        self.assertFalse(new_v.is_contiguous())
        slots = torch.tensor(
            [block_table[i, 0] * self.block_size + old_lengths[i] for i in range(len(old_lengths))],
            device="cuda", dtype=torch.int32,
        )
        updated_lengths = torch.tensor(
            [length + 1 for length in old_lengths], device="cuda", dtype=torch.int32,
        )

        actual = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            updated_lengths,
            block_table,
            self.scale,
            True,
            new_k=new_k,
            new_v=new_v,
            slot_mapping=slots,
        )
        store_kvcache(new_k, new_v, reference_k_cache, reference_v_cache, slots)
        expected = reference_decode(
            q,
            reference_k_cache,
            reference_v_cache,
            cache_seqlens=updated_lengths,
            block_table=block_table,
            softmax_scale=self.scale,
            causal=True,
        )
        self.assert_close(actual, expected)
        torch.testing.assert_close(k_cache, reference_k_cache, atol=0, rtol=0)
        torch.testing.assert_close(v_cache, reference_v_cache, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
