import math
import unittest

import torch

from nanovllm.layers.attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    store_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepseekKVCacheCudaTest(unittest.TestCase):

    dtype = torch.bfloat16
    head_dim = 192

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)

    def test_store_supports_non_power_of_two_expanded_mla_width(self):
        num_tokens = 3
        num_heads = 16
        head_dim = self.head_dim
        block_size = 256
        num_blocks = 2
        shape = (num_tokens, num_heads, head_dim)
        key = torch.randn(shape, device="cuda", dtype=self.dtype)
        value = torch.randn_like(key)
        k_cache = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            device="cuda",
            dtype=key.dtype,
        )
        v_cache = torch.zeros_like(k_cache)
        slots = torch.tensor([0, 257, 511], device="cuda", dtype=torch.int32)

        store_kvcache(key, value, k_cache, v_cache, slots)

        flat_k_cache = k_cache.view(-1, num_heads, head_dim)
        flat_v_cache = v_cache.view(-1, num_heads, head_dim)
        torch.testing.assert_close(flat_k_cache[slots.long()], key)
        torch.testing.assert_close(flat_v_cache[slots.long()], value)

    def test_prefill_supports_expanded_mla_head_dim(self):
        lengths = [5, 7]
        num_heads = 2
        total_tokens = sum(lengths)
        shape = (total_tokens, num_heads, self.head_dim)
        query = torch.randn(shape, device="cuda", dtype=self.dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cu_seqlens = torch.tensor(
            [0, lengths[0], total_tokens], device="cuda", dtype=torch.int32
        )
        scale = 1.0 / math.sqrt(self.head_dim)

        actual = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            max(lengths),
            max(lengths),
            scale,
            True,
        )

        expected_parts = []
        start = 0
        for length in lengths:
            q = query[start : start + length]
            k = key[start : start + length]
            v = value[start : start + length]
            scores = torch.einsum("qhd,khd->hqk", q, k).float() * scale
            causal_mask = torch.triu(
                torch.ones(length, length, device="cuda", dtype=torch.bool),
                diagonal=1,
            )
            probabilities = scores.masked_fill(
                causal_mask, float("-inf")
            ).softmax(dim=-1).to(self.dtype)
            expected_parts.append(torch.einsum("hqk,khd->qhd", probabilities, v))
            start += length
        self.assert_close(actual, torch.cat(expected_parts))

    def test_large_batch_decode_dot_kernel_supports_expanded_mla_head_dim(self):
        batch = 16  # selects the tl.dot decode path rather than the vector path
        num_heads = 2
        block_size = 256
        lengths = torch.arange(3, 3 + batch, device="cuda", dtype=torch.int32)
        query = torch.randn(
            batch,
            1,
            num_heads,
            self.head_dim,
            device="cuda",
            dtype=self.dtype,
        )
        cache_shape = (batch, block_size, num_heads, self.head_dim)
        key_cache = torch.zeros(cache_shape, device="cuda", dtype=self.dtype)
        value_cache = torch.zeros_like(key_cache)
        for batch_idx, length in enumerate(lengths.tolist()):
            key_cache[batch_idx, :length].normal_()
            value_cache[batch_idx, :length].normal_()
        block_table = torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).unsqueeze(1)
        scale = 1.0 / math.sqrt(self.head_dim)

        actual = flash_attn_with_kvcache(
            query,
            key_cache,
            value_cache,
            lengths,
            block_table,
            scale,
            True,
        )

        expected = []
        for batch_idx, length in enumerate(lengths.tolist()):
            scores = torch.einsum(
                "hd,thd->ht", query[batch_idx, 0], key_cache[batch_idx, :length]
            ).float() * scale
            probabilities = scores.softmax(dim=-1).to(self.dtype)
            expected.append(
                torch.einsum(
                    "ht,thd->hd", probabilities, value_cache[batch_idx, :length]
                )
            )
        expected = torch.stack(expected).unsqueeze(1)
        self.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
