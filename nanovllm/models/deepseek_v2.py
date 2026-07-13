"""DeepSeek-V2 inference model with a correctness-first MoE backend.

The attention path expands MLA keys and values before handing them to nano-vLLM's
paged attention implementation.  This is intentionally simpler than a latent-KV
kernel and provides a reference path for DeepSeek-V2/Lite checkpoints.
"""

import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.moe import SparseMoE


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return dim * math.log(
        max_position_embeddings / (num_rotations * 2 * math.pi)
    ) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(
            low_rot, dim, base, max_position_embeddings
        )
    )
    high = math.ceil(
        _yarn_find_correction_dim(
            high_rot, dim, base, max_position_embeddings
        )
    )
    return max(low, 0), min(high, dim - 1)


class DeepseekV2RotaryEmbedding(nn.Module):
    """Partial RoPE used by DeepSeek-V2, including its YaRN variant."""

    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int,
        base: float = 10000.0,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError(f"rotary dimension must be positive and even, got {dim}")

        self.dim = dim
        rope_scaling = rope_scaling or {}
        rope_type = rope_scaling.get(
            "rope_type", rope_scaling.get("type", "default")
        )
        factor = float(rope_scaling.get("factor", 1.0))
        frequencies = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        rotary_mscale = 1.0

        if rope_type in (None, "default"):
            pass
        elif rope_type == "linear":
            frequencies = frequencies / factor
        elif rope_type == "yarn":
            original_max_position = int(
                rope_scaling.get(
                    "original_max_position_embeddings",
                    max_position_embeddings,
                )
            )
            beta_fast = float(rope_scaling.get("beta_fast", 32.0))
            beta_slow = float(rope_scaling.get("beta_slow", 1.0))
            low, high = _yarn_find_correction_range(
                beta_fast,
                beta_slow,
                dim,
                base,
                original_max_position,
            )
            ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (
                high - low if high != low else 0.001
            )
            ramp = ramp.clamp(0, 1)
            extrapolation_mask = 1.0 - ramp
            interpolated = frequencies / factor
            frequencies = (
                interpolated * (1.0 - extrapolation_mask)
                + frequencies * extrapolation_mask
            )
            rotary_mscale = yarn_get_mscale(
                factor, float(rope_scaling.get("mscale", 1.0))
            ) / yarn_get_mscale(
                factor, float(rope_scaling.get("mscale_all_dim", 0.0))
            )
        else:
            raise ValueError(f"unsupported DeepSeek RoPE scaling type: {rope_type}")

        self.register_buffer("inv_freq", frequencies, persistent=False)
        self.rotary_mscale = rotary_mscale

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        first, second = hidden_states.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _apply_rotary(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # DeepSeek checkpoints store the rotary part in adjacent complex pairs.
        # Rearrange it to the half/half layout used by rotate_half first.
        shape = hidden_states.shape
        hidden_states = (
            hidden_states.reshape(*shape[:-1], self.dim // 2, 2)
            .transpose(-1, -2)
            .reshape(shape)
        )
        return hidden_states * cos + self._rotate_half(hidden_states) * sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(
            positions.to(dtype=torch.float32), self.inv_freq.float()
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        cos = (embeddings.cos() * self.rotary_mscale).to(query.dtype).unsqueeze(1)
        sin = (embeddings.sin() * self.rotary_mscale).to(query.dtype).unsqueeze(1)
        return self._apply_rotary(query, cos, sin), self._apply_rotary(key, cos, sin)


def _rope_settings(config) -> tuple[float, dict | None]:
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = getattr(config, "rope_parameters", None)
    rope_scaling = dict(rope_scaling) if rope_scaling else None
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None and rope_scaling is not None:
        rope_theta = rope_scaling.get("rope_theta")
    return float(rope_theta or 10000.0), rope_scaling


class DeepseekV2MLP(nn.Module):

    def __init__(self, config, intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        bias = bool(getattr(config, "mlp_bias", False))
        if config.hidden_act != "silu":
            raise ValueError(
                f"DeepSeek-V2 currently requires silu, got {config.hidden_act}"
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=bias
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=bias
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class DeepseekV2MoE(SparseMoE):
    """DeepSeek-V2 softmax router plus routed and shared experts."""

    def __init__(self, config) -> None:
        experts = [
            DeepseekV2MLP(config, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        ]
        shared_experts = None
        if getattr(config, "n_shared_experts", None):
            shared_experts = DeepseekV2MLP(
                config,
                config.moe_intermediate_size * config.n_shared_experts,
            )
        gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            experts=experts,
            gate=gate,
            shared_experts=shared_experts,
            norm_topk_prob=bool(getattr(config, "norm_topk_prob", False)),
            routed_scaling_factor=float(config.routed_scaling_factor),
        )
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.topk_method = getattr(config, "topk_method", "greedy")
        self.num_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bias = self.gate.bias
        router_logits = F.linear(
            hidden_states.float(),
            self.gate.weight.float(),
            bias.float() if bias is not None else None,
        )
        if self.scoring_func != "softmax":
            raise ValueError(
                "DeepSeek-V2 supports only softmax routing; "
                f"got {self.scoring_func}"
            )
        scores = router_logits.softmax(dim=-1, dtype=torch.float32)

        if self.topk_method == "greedy":
            routing_weights, selected_experts = torch.topk(
                scores, self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            if not self.num_group or not self.topk_group:
                raise ValueError(
                    "group-limited routing requires n_group and topk_group"
                )
            if self.num_experts % self.num_group:
                raise ValueError("n_routed_experts must be divisible by n_group")
            grouped_scores = scores.view(
                -1, self.num_group, self.num_experts // self.num_group
            )
            group_scores = grouped_scores.max(dim=-1).values
            selected_groups = group_scores.topk(
                self.topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, selected_groups, True)
            expert_mask = (
                group_mask.unsqueeze(-1)
                .expand_as(grouped_scores)
                .reshape_as(scores)
            )
            eligible_scores = scores.masked_fill(~expert_mask, 0.0)
            routing_weights, selected_experts = eligible_scores.topk(
                self.top_k, dim=-1, sorted=False
            )
        else:
            raise ValueError(
                f"unsupported DeepSeek-V2 top-k method: {self.topk_method}"
            )

        if self.normalize_topk_weights and self.top_k > 1:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights * self.routed_scaling_factor
        return router_logits, routing_weights, selected_experts


class DeepseekV2Attention(nn.Module):
    """Expanded-KV implementation of DeepSeek-V2 MLA."""

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by TP size")

        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        if self.v_head_dim > self.qk_head_dim:
            raise ValueError("v_head_dim cannot exceed the expanded Q/K head size")

        attention_bias = bool(getattr(config, "attention_bias", False))
        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = ReplicatedLinear(
                config.hidden_size, self.q_lora_rank, bias=attention_bias
            )
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.total_num_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank, eps=config.rms_norm_eps
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            config.hidden_size,
            bias=attention_bias,
        )

        rope_theta, rope_scaling = _rope_settings(config)
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        scaling = self.qk_head_dim**-0.5
        if rope_scaling and rope_scaling.get("mscale_all_dim", 0):
            scaling_factor = float(rope_scaling.get("factor", 1.0))
            mscale = yarn_get_mscale(
                scaling_factor,
                float(rope_scaling["mscale_all_dim"]),
            )
            scaling *= mscale * mscale
        self.attn = Attention(
            self.num_heads,
            self.qk_head_dim,
            scaling,
            self.num_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is None:
            query = self.q_proj(hidden_states)
        else:
            query = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        query = query.view(-1, self.num_heads, self.qk_head_dim)
        query_nope, query_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, key_rope = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        key_rope = key_rope.view(-1, 1, self.qk_rope_head_dim)
        expanded_kv = self.kv_b_proj(
            self.kv_a_layernorm(compressed_kv)
        ).view(
            -1,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = expanded_kv.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        query_rope, key_rope = self.rotary_emb(
            positions, query_rope, key_rope
        )
        key_rope = key_rope.expand(-1, self.num_heads, -1)
        query = torch.cat((query_nope, query_rope), dim=-1)
        key = torch.cat((key_nope, key_rope), dim=-1)
        if self.v_head_dim != self.qk_head_dim:
            value = F.pad(value, (0, self.qk_head_dim - self.v_head_dim))

        output = self.attn(query, key, value)
        output = output[..., : self.v_head_dim]
        return self.o_proj(output.flatten(1, -1))


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        moe_frequency = int(getattr(config, "moe_layer_freq", 1))
        use_moe = (
            getattr(config, "n_routed_experts", None) is not None
            and layer_idx >= int(getattr(config, "first_k_dense_replace", 0))
            and layer_idx % moe_frequency == 0
        )
        self.mlp = DeepseekV2MoE(config) if use_moe else DeepseekV2MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DeepseekV2Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):
    supports_cuda_graph = False
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, quant_config=None) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("AWQ quantization is not supported for DeepSeek-V2")
        self.model = DeepseekV2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = [
    "DeepseekV2Attention",
    "DeepseekV2DecoderLayer",
    "DeepseekV2ForCausalLM",
    "DeepseekV2MLP",
    "DeepseekV2MoE",
    "DeepseekV2Model",
    "DeepseekV2RotaryEmbedding",
]
