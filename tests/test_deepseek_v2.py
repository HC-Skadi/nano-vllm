import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

from nanovllm.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    DeepseekV2Model,
    DeepseekV2RotaryEmbedding,
    yarn_get_mscale,
)
from nanovllm.utils.loader import load_model


def tiny_config(**overrides):
    values = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "kv_lora_rank": 8,
        "q_lora_rank": None,
        "n_group": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "routed_scaling_factor": 1.0,
        "topk_group": 1,
        "topk_method": "greedy",
        "scoring_func": "softmax",
        "norm_topk_prob": False,
        "v_head_dim": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@contextmanager
def single_rank_dist():
    with (
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        yield


class CaptureAttention(nn.Module):

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self.query = query.detach().clone()
        self.key = key.detach().clone()
        self.value = value.detach().clone()
        return value


class CausalReferenceAttention(nn.Module):

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(
            torch.ones(
                scores.shape[-2:], device=scores.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        return torch.matmul(probabilities, value).transpose(0, 1)


class EagerRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return normalized.to(hidden_states.dtype) * self.weight


class DeepseekV2Test(unittest.TestCase):

    def test_rejects_awq_quantization_explicitly(self):
        with self.assertRaisesRegex(ValueError, "AWQ quantization is not supported"):
            DeepseekV2ForCausalLM(tiny_config(), quant_config=object())

    def setUp(self):
        torch.manual_seed(0)

    def test_tiny_config_selects_dense_then_moe_layers(self):
        with single_rank_dist():
            model = DeepseekV2Model(tiny_config())

        self.assertIsInstance(model.layers[0].mlp, DeepseekV2MLP)
        self.assertIsInstance(model.layers[1].mlp, DeepseekV2MoE)
        self.assertEqual(len(model.layers[1].mlp.experts), 4)
        self.assertIsInstance(
            model.layers[1].mlp.shared_experts, DeepseekV2MLP
        )

    def test_yarn_rotary_matches_adjacent_complex_pair_formula(self):
        scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 16,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 0.5,
            "mscale_all_dim": 0.25,
        }
        rotary = DeepseekV2RotaryEmbedding(
            4,
            max_position_embeddings=64,
            base=10000.0,
            rope_scaling=scaling,
        )

        # For this small configuration the first frequency extrapolates and the
        # second interpolates by the YaRN factor.
        torch.testing.assert_close(
            rotary.inv_freq, torch.tensor([1.0, 0.01 / 4.0])
        )
        expected_mscale = yarn_get_mscale(4.0, 0.5) / yarn_get_mscale(
            4.0, 0.25
        )
        self.assertAlmostEqual(rotary.rotary_mscale, expected_mscale)

        # RotaryEmbedding must retain nn.Module's device/dtype migration
        # contract; in particular, helper names must not shadow Module._apply.
        rotary = rotary.to(dtype=torch.float64)
        self.assertEqual(rotary.inv_freq.dtype, torch.float64)

        positions = torch.tensor([0, 3])
        query = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, -3.0]],
                [[-2.0, 1.0, 0.25, 0.75], [4.0, 3.0, 2.0, 1.0]],
            ],
            dtype=torch.float64,
        )
        key = query[:, :1].clone()
        actual_query, actual_key = rotary(positions, query, key)
        self.assertEqual(actual_query.dtype, torch.float64)
        self.assertEqual(actual_key.dtype, torch.float64)

        angles = torch.outer(positions.float(), rotary.inv_freq)
        phase = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)

        def reference(hidden_states):
            complex_states = torch.view_as_complex(
                hidden_states.reshape(*hidden_states.shape[:-1], 2, 2)
            )
            adjacent = torch.view_as_real(
                complex_states * phase * expected_mscale
            ).flatten(-2)
            # The implementation leaves the result in half/half layout, which
            # is a shared permutation of Q and K and therefore preserves QK^T.
            return (
                adjacent.reshape(*adjacent.shape[:-1], 2, 2)
                .transpose(-1, -2)
                .reshape_as(adjacent)
            )

        torch.testing.assert_close(actual_query, reference(query))
        torch.testing.assert_close(actual_key, reference(key))

    def test_mla_shapes_partial_rope_and_value_padding(self):
        config = tiny_config()
        with single_rank_dist():
            attention = DeepseekV2Attention(config)
        attention.kv_a_layernorm = nn.Identity()
        capture = CaptureAttention()
        attention.attn = capture
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.uniform_(-0.1, 0.1)

        hidden_states = torch.randn(3, config.hidden_size)
        positions = torch.tensor([0, 1, 5])
        raw_query = attention.q_proj(hidden_states).view(
            3, config.num_attention_heads, 12
        )
        expected_rotary, _ = attention.rotary_emb(
            positions,
            raw_query[..., 8:],
            torch.zeros(3, 1, 4),
        )
        output = attention(positions, hidden_states)

        self.assertEqual(capture.query.shape, (3, 4, 12))
        self.assertEqual(capture.key.shape, (3, 4, 12))
        self.assertEqual(capture.value.shape, (3, 4, 12))
        self.assertEqual(output.shape, (3, config.hidden_size))
        torch.testing.assert_close(capture.query[..., :8], raw_query[..., :8])
        torch.testing.assert_close(capture.query[..., 8:], expected_rotary)
        torch.testing.assert_close(
            capture.value[..., 8:], torch.zeros_like(capture.value[..., 8:])
        )
        for head in range(1, config.num_attention_heads):
            torch.testing.assert_close(
                capture.key[:, head, 8:], capture.key[:, 0, 8:]
            )

    def test_mla_matches_self_contained_eager_reference(self):
        config = tiny_config()
        with single_rank_dist():
            attention = DeepseekV2Attention(config)
        eager_norm = EagerRMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        attention.kv_a_layernorm = eager_norm
        attention.attn = CausalReferenceAttention(attention.attn.scale)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.uniform_(-0.15, 0.15)

        hidden_states = torch.randn(5, config.hidden_size)
        positions = torch.arange(hidden_states.shape[0])
        with torch.no_grad():
            actual = attention(positions, hidden_states)

            query = torch.nn.functional.linear(
                hidden_states, attention.q_proj.weight
            ).view(5, config.num_attention_heads, 12)
            query_nope, query_rope = query.split([8, 4], dim=-1)

            compressed = torch.nn.functional.linear(
                hidden_states,
                attention.kv_a_proj_with_mqa.weight,
            )
            latent, key_rope = compressed.split([8, 4], dim=-1)
            latent_fp32 = latent.float()
            latent = (
                latent_fp32
                * torch.rsqrt(
                    latent_fp32.square().mean(-1, keepdim=True)
                    + eager_norm.eps
                )
            ).to(latent.dtype) * eager_norm.weight
            expanded = torch.nn.functional.linear(
                latent, attention.kv_b_proj.weight
            ).view(5, config.num_attention_heads, 16)
            key_nope, value = expanded.split([8, 8], dim=-1)

            # Transformers 5.2 expresses DeepSeek-V2 RoPE as multiplication of
            # adjacent complex pairs.  Nano stores the rotated result in a
            # shared half/half permutation; QK^T is invariant to that shared
            # permutation, so use the original complex formula as the oracle.
            angles = torch.outer(positions.float(), attention.rotary_emb.inv_freq)
            phase = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)
            query_rope = torch.view_as_real(
                torch.view_as_complex(query_rope.reshape(5, 4, 2, 2)) * phase
            ).flatten(-2)
            key_rope = torch.view_as_real(
                torch.view_as_complex(key_rope.reshape(5, 1, 2, 2)) * phase
            ).flatten(-2)
            query = torch.cat((query_nope, query_rope), dim=-1)
            key = torch.cat(
                (key_nope, key_rope.expand(-1, config.num_attention_heads, -1)),
                dim=-1,
            )

            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores * attention.attn.scale
            causal_mask = torch.triu(
                torch.ones(5, 5, dtype=torch.bool), diagonal=1
            )
            probabilities = torch.softmax(
                scores.masked_fill(causal_mask, float("-inf")), dim=-1
            )
            heads = torch.matmul(probabilities, value).transpose(0, 1)
            expected = torch.nn.functional.linear(
                heads.flatten(1), attention.o_proj.weight
            )

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_router_uses_fp32_softmax_without_topk_renormalization(self):
        config = tiny_config(routed_scaling_factor=1.75)
        with single_rank_dist():
            moe = DeepseekV2MoE(config)
        with torch.no_grad():
            moe.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0] + [0.0] * 30,
                        [0.0, 1.0] + [0.0] * 30,
                        [-1.0, 0.0] + [0.0] * 30,
                        [0.0, -1.0] + [0.0] * 30,
                    ]
                )
            )
        hidden_states = torch.zeros(2, 32, dtype=torch.float16)
        hidden_states[0, :2] = torch.tensor([2.0, -1.0])
        hidden_states[1, :2] = torch.tensor([-1.0, -2.0])

        logits, weights, indices = moe.route(hidden_states)
        expected_logits = torch.nn.functional.linear(
            hidden_states.float(), moe.gate.weight.float()
        )
        expected_weights, expected_indices = torch.softmax(
            expected_logits, dim=-1
        ).topk(2, dim=-1, sorted=False)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(weights.dtype, torch.float32)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, expected_weights * 1.75)
        self.assertTrue(torch.all(weights.sum(-1) < 1.75))

    def test_group_limited_router_selects_only_the_best_group(self):
        config = tiny_config(
            n_group=2,
            topk_group=1,
            topk_method="group_limited_greedy",
            routed_scaling_factor=1.5,
        )
        with single_rank_dist():
            moe = DeepseekV2MoE(config)
        with torch.no_grad():
            moe.gate.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0] + [0.0] * 30,
                        [1.0, 0.0] + [0.0] * 30,
                        [0.0, 2.0] + [0.0] * 30,
                        [0.0, 1.0] + [0.0] * 30,
                    ]
                )
            )
        hidden_states = torch.zeros(2, 32, dtype=torch.float16)
        hidden_states[0, :2] = torch.tensor([2.0, 0.25])
        hidden_states[1, :2] = torch.tensor([0.25, 2.0])

        logits, weights, indices = moe.route(hidden_states)
        scores = torch.softmax(logits, dim=-1)
        grouped = scores.view(-1, 2, 2)
        best_group = grouped.max(-1).values.argmax(-1)
        group_mask = torch.zeros_like(grouped, dtype=torch.bool)
        group_mask[
            torch.arange(grouped.shape[0]), best_group
        ] = True
        eligible = scores.masked_fill(~group_mask.reshape_as(scores), 0.0)
        expected_weights, expected_indices = eligible.topk(
            2, dim=-1, sorted=False
        )

        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, expected_weights * 1.5)
        self.assertTrue(
            torch.all(indices.div(2, rounding_mode="floor") == best_group[:, None])
        )
        self.assertTrue(torch.all(weights.sum(-1) < 1.5))

    def test_packed_checkpoint_paths_load_dense_routed_and_shared_mlps(self):
        config = tiny_config()
        with single_rank_dist():
            model = DeepseekV2ForCausalLM(config)

        tensors = {
            "model.layers.0.mlp.gate_proj.weight": torch.full((64, 32), 1.0),
            "model.layers.0.mlp.up_proj.weight": torch.full((64, 32), 2.0),
            "model.layers.1.mlp.experts.2.gate_proj.weight": torch.full(
                (16, 32), 3.0
            ),
            "model.layers.1.mlp.experts.2.up_proj.weight": torch.full(
                (16, 32), 4.0
            ),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.full(
                (16, 32), 5.0
            ),
            "model.layers.1.mlp.shared_experts.up_proj.weight": torch.full(
                (16, 32), 6.0
            ),
        }
        with tempfile.TemporaryDirectory() as model_dir:
            save_file(tensors, f"{model_dir}/model.safetensors")
            load_model(model, model_dir, strict=False)

        dense = model.get_parameter(
            "model.layers.0.mlp.gate_up_proj.weight"
        )
        routed = model.get_parameter(
            "model.layers.1.mlp.experts.2.gate_up_proj.weight"
        )
        shared = model.get_parameter(
            "model.layers.1.mlp.shared_experts.gate_up_proj.weight"
        )
        torch.testing.assert_close(dense[:64], tensors[next(iter(tensors))])
        torch.testing.assert_close(
            dense[64:], tensors["model.layers.0.mlp.up_proj.weight"]
        )
        torch.testing.assert_close(
            routed[:16],
            tensors["model.layers.1.mlp.experts.2.gate_proj.weight"],
        )
        torch.testing.assert_close(
            routed[16:],
            tensors["model.layers.1.mlp.experts.2.up_proj.weight"],
        )
        torch.testing.assert_close(
            shared[:16],
            tensors["model.layers.1.mlp.shared_experts.gate_proj.weight"],
        )
        torch.testing.assert_close(
            shared[16:],
            tensors["model.layers.1.mlp.shared_experts.up_proj.weight"],
        )


if __name__ == "__main__":
    unittest.main()
