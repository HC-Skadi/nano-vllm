import math
import unittest

import torch

from nanovllm.layers.attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    store_kvcache,
)
from nanovllm.layers.mla import LatentMLAAttention
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepseekKVCacheCudaTest(unittest.TestCase):

    dtype = torch.bfloat16
    head_dim = 192

    def assert_close(self, actual, expected):
        # BF16 contractions can differ by one or two ULPs when the eager
        # reference and Triton/batched-GEMM paths accumulate in a different
        # order.  The RTX 3060 observed worst-case absolute error is 0.015625.
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)

    def tearDown(self):
        reset_context()

    def test_native_latent_cache_matches_full_attention_at_lite_dimensions(self):
        torch.manual_seed(0)
        num_heads = 2
        latent_dim = 512
        rope_dim = 64
        total_tokens = 8
        module = LatentMLAAttention(
            num_heads,
            latent_dim,
            rope_dim,
            (128 + rope_dim) ** -0.5,
        ).cuda()
        query_latent = torch.randn(
            total_tokens,
            num_heads,
            latent_dim,
            device="cuda",
            dtype=self.dtype,
        )
        query_rope = torch.randn(
            total_tokens,
            num_heads,
            rope_dim,
            device="cuda",
            dtype=self.dtype,
        )
        latent = torch.randn(
            total_tokens, latent_dim, device="cuda", dtype=self.dtype
        )
        key_rope = torch.randn(
            total_tokens, rope_dim, device="cuda", dtype=self.dtype
        )

        expected = module(query_latent, query_rope, latent, key_rope)
        module.k_cache = torch.zeros(
            1,
            256,
            1,
            latent_dim + rope_dim,
            device="cuda",
            dtype=self.dtype,
        )

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 5], device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 5], device="cuda", dtype=torch.int32),
            max_seqlen_q=5,
            max_seqlen_k=5,
            slot_mapping=torch.arange(5, device="cuda", dtype=torch.int32),
        )
        prefill = module(
            query_latent[:5], query_rope[:5], latent[:5], key_rope[:5]
        )

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 7], device="cuda", dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=7,
            slot_mapping=torch.tensor([5, 6], device="cuda", dtype=torch.int32),
            block_tables=torch.tensor([[0]], device="cuda", dtype=torch.int32),
        )
        prefix = module(
            query_latent[5:7], query_rope[5:7], latent[5:7], key_rope[5:7]
        )

        set_context(
            False,
            slot_mapping=torch.tensor([7], device="cuda", dtype=torch.int32),
            context_lens=torch.tensor([8], device="cuda", dtype=torch.int32),
            block_tables=torch.tensor([[0]], device="cuda", dtype=torch.int32),
        )
        decode = module(
            query_latent[7:], query_rope[7:], latent[7:], key_rope[7:]
        )

        self.assert_close(prefill, expected[:5])
        self.assert_close(prefix, expected[5:7])
        self.assert_close(decode, expected[7:])
        cached = module.k_cache.view(-1, latent_dim + rope_dim)[:total_tokens]
        self.assert_close(cached[:, :latent_dim], latent)
        self.assert_close(cached[:, latent_dim:], key_rope)

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
