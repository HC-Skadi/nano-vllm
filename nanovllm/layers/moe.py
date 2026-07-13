"""Correctness-first PyTorch mixture-of-experts building blocks.

The implementation intentionally dispatches tokens one expert at a time.  It is
not as fast as a fused MoE kernel, but it keeps the routing semantics explicit,
works on CPU, and provides a useful reference implementation for optimized
backends.
"""

from collections.abc import Callable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


LinearFactory = Callable[[int, int, bool], nn.Module]
Activation = Callable[[torch.Tensor], torch.Tensor]


def topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k expert weights and indices for each token.

    Softmax is evaluated before selecting the experts, matching the routing used
    by common DeepSeek and Qwen MoE models.  Computing it in float32 by default
    avoids avoidable precision loss when model activations use fp16/bf16.

    Args:
        router_logits: Tensor whose last dimension is the expert dimension.
        top_k: Number of experts selected per token.
        normalize: Renormalize the selected weights to sum to one.  When false,
            the selected weights retain their probabilities from the full
            expert softmax.
        dtype: Dtype used for softmax and returned routing weights.
    """
    if router_logits.ndim == 0:
        raise ValueError("router_logits must have an expert dimension")
    num_experts = router_logits.shape[-1]
    if not 0 < top_k <= num_experts:
        raise ValueError(
            f"top_k must be in [1, {num_experts}], but got {top_k}"
        )

    routing_weights = F.softmax(router_logits, dim=-1, dtype=dtype)
    routing_weights, selected_experts = torch.topk(
        routing_weights, top_k, dim=-1
    )
    if normalize:
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
    return routing_weights, selected_experts


def _make_linear(
    linear_factory: LinearFactory,
    input_size: int,
    output_size: int,
    bias: bool,
) -> nn.Module:
    # Pass bias positionally so both nn.Linear and nano-vllm's linear classes
    # can be injected without adapters.
    return linear_factory(input_size, output_size, bias)


class GatedMLP(nn.Module):
    """SwiGLU-style expert with injectable projection implementations.

    Projection names intentionally follow Hugging Face checkpoints
    (``gate_proj``, ``up_proj``, and ``down_proj``), which makes this module
    convenient for both DeepSeek and Qwen expert weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Activation | nn.Module = F.silu,
        linear_factory: LinearFactory = nn.Linear,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, but got {hidden_size}")
        if intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive, "
                f"but got {intermediate_size}"
            )

        self.gate_proj = _make_linear(
            linear_factory, hidden_size, intermediate_size, bias
        )
        self.up_proj = _make_linear(
            linear_factory, hidden_size, intermediate_size, bias
        )
        self.down_proj = _make_linear(
            linear_factory, intermediate_size, hidden_size, bias
        )
        self.act_fn = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class TopKRouter(nn.Module):
    """Learned linear router with top-k softmax selection."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        gate: nn.Module | None = None,
        normalize_topk_weights: bool = True,
        routed_scaling_factor: float = 1.0,
        router_dtype: torch.dtype = torch.float32,
        linear_factory: LinearFactory = nn.Linear,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, but got {hidden_size}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, but got {num_experts}")
        if not 0 < top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, {num_experts}], but got {top_k}"
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk_weights = normalize_topk_weights
        self.routed_scaling_factor = routed_scaling_factor
        self.router_dtype = router_dtype
        self.gate = (
            gate
            if gate is not None
            else _make_linear(linear_factory, hidden_size, num_experts, bias)
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(router_logits, routing_weights, selected_experts)``."""
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.hidden_size:
            actual = hidden_states.shape[-1] if hidden_states.ndim else None
            raise ValueError(
                f"expected hidden size {self.hidden_size}, but got {actual}"
            )

        router_logits = self.gate(hidden_states)
        expected_shape = (*hidden_states.shape[:-1], self.num_experts)
        if router_logits.shape != expected_shape:
            raise ValueError(
                "gate must return one logit per expert; expected shape "
                f"{expected_shape}, but got {tuple(router_logits.shape)}"
            )
        routing_weights, selected_experts = topk_softmax(
            router_logits,
            self.top_k,
            normalize=self.normalize_topk_weights,
            dtype=self.router_dtype,
        )
        routing_weights = routing_weights * self.routed_scaling_factor
        return router_logits, routing_weights, selected_experts


class SparseMoE(nn.Module):
    """Generic sparse MoE layer for DeepSeek/Qwen-style decoder blocks.

    ``experts`` may be supplied directly, or default :class:`GatedMLP`
    instances are built from ``intermediate_size`` and ``linear_factory``.
    ``shared_experts`` can be either one module (the common DeepSeek layout) or
    a sequence of modules; shared outputs are summed with the routed output.
    When ``shared_expert_gate`` is present its sigmoid output gates the summed
    shared output, matching the Qwen MoE layout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int | None = None,
        intermediate_size: int | None = None,
        *,
        num_experts_per_tok: int | None = None,
        experts: Sequence[nn.Module] | None = None,
        gate: nn.Module | None = None,
        shared_experts: nn.Module | Sequence[nn.Module] | None = None,
        shared_expert_gate: nn.Module | None = None,
        normalize_topk_weights: bool | None = None,
        norm_topk_prob: bool | None = None,
        routed_scaling_factor: float = 1.0,
        router_dtype: torch.dtype = torch.float32,
        activation: Activation | nn.Module = F.silu,
        linear_factory: LinearFactory = nn.Linear,
        bias: bool = False,
        return_router_logits: bool = False,
    ) -> None:
        super().__init__()
        top_k = self._resolve_alias(
            "top_k", top_k, "num_experts_per_tok", num_experts_per_tok
        )
        if top_k is None:
            raise ValueError("top_k or num_experts_per_tok must be provided")

        normalize_topk_weights = self._resolve_alias(
            "normalize_topk_weights",
            normalize_topk_weights,
            "norm_topk_prob",
            norm_topk_prob,
        )
        if normalize_topk_weights is None:
            normalize_topk_weights = True

        if experts is None:
            if intermediate_size is None:
                raise ValueError(
                    "intermediate_size is required when experts are not supplied"
                )
            experts = [
                GatedMLP(
                    hidden_size,
                    intermediate_size,
                    activation=activation,
                    linear_factory=linear_factory,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        else:
            experts = list(experts)
            if len(experts) != num_experts:
                raise ValueError(
                    f"expected {num_experts} experts, but got {len(experts)}"
                )
            if not all(isinstance(expert, nn.Module) for expert in experts):
                raise TypeError("every expert must be an nn.Module")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(experts)
        # Keep the gate at the top level so standard checkpoint paths such as
        # ``mlp.gate.weight`` load without an extra name translation.
        self.gate = (
            gate
            if gate is not None
            else _make_linear(linear_factory, hidden_size, num_experts, bias)
        )
        self.normalize_topk_weights = normalize_topk_weights
        self.routed_scaling_factor = routed_scaling_factor
        self.router_dtype = router_dtype
        self.shared_experts = self._register_shared_experts(shared_experts)
        self.shared_expert_gate = shared_expert_gate
        self.return_router_logits = return_router_logits

    @staticmethod
    def _resolve_alias(
        name: str,
        value: int | bool | None,
        alias_name: str,
        alias_value: int | bool | None,
    ) -> int | bool | None:
        if value is not None and alias_value is not None and value != alias_value:
            raise ValueError(
                f"{name} ({value}) and {alias_name} ({alias_value}) disagree"
            )
        return value if value is not None else alias_value

    @staticmethod
    def _register_shared_experts(
        shared_experts: nn.Module | Sequence[nn.Module] | None,
    ) -> nn.Module | None:
        if shared_experts is None:
            return None
        if isinstance(shared_experts, nn.ModuleList):
            return shared_experts
        if isinstance(shared_experts, nn.Module):
            return shared_experts
        modules = list(shared_experts)
        if not all(isinstance(expert, nn.Module) for expert in modules):
            raise TypeError("every shared expert must be an nn.Module")
        return nn.ModuleList(modules)

    def _shared_forward(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        if isinstance(self.shared_experts, nn.ModuleList):
            shared_output = torch.zeros_like(hidden_states)
            for expert in self.shared_experts:
                shared_output = shared_output + expert(hidden_states)
        else:
            shared_output = self.shared_experts(hidden_states)

        if shared_output.shape != hidden_states.shape:
            raise ValueError(
                "shared experts must preserve the hidden-state shape; expected "
                f"{tuple(hidden_states.shape)}, got {tuple(shared_output.shape)}"
            )
        if self.shared_expert_gate is not None:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            try:
                shared_output = shared_output * shared_gate
            except RuntimeError as error:
                raise ValueError(
                    "shared_expert_gate output must broadcast to hidden states; "
                    f"got {tuple(shared_gate.shape)} and "
                    f"{tuple(hidden_states.shape)}"
                ) from error
        return shared_output

    def route(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return router logits, selected weights, and selected expert ids."""
        router_logits = self.gate(hidden_states)
        expected_shape = (*hidden_states.shape[:-1], self.num_experts)
        if router_logits.shape != expected_shape:
            raise ValueError(
                "gate must return one logit per expert; expected shape "
                f"{expected_shape}, but got {tuple(router_logits.shape)}"
            )
        routing_weights, selected_experts = topk_softmax(
            router_logits,
            self.top_k,
            normalize=self.normalize_topk_weights,
            dtype=self.router_dtype,
        )
        routing_weights = routing_weights * self.routed_scaling_factor
        return router_logits, routing_weights, selected_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_router_logits: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.hidden_size:
            actual = hidden_states.shape[-1] if hidden_states.ndim else None
            raise ValueError(
                f"expected hidden size {self.hidden_size}, but got {actual}"
            )

        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits, routing_weights, selected_experts = self.route(flat_states)

        # Flatten the top-k choices so all tokens assigned to one expert are
        # evaluated in a single module call.  index_add handles the case where a
        # token contributes through more than one selected expert.
        token_indices = torch.arange(
            flat_states.shape[0], device=flat_states.device
        ).repeat_interleave(self.top_k)
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        output = torch.zeros_like(flat_states)

        for expert_index, expert in enumerate(self.experts):
            assignment_mask = flat_experts == expert_index
            expert_tokens = token_indices[assignment_mask]
            if expert_tokens.numel() == 0:
                continue
            expert_input = flat_states.index_select(0, expert_tokens)
            expert_output = expert(expert_input)
            if expert_output.shape != expert_input.shape:
                raise ValueError(
                    f"expert {expert_index} must preserve hidden shape; expected "
                    f"{tuple(expert_input.shape)}, got {tuple(expert_output.shape)}"
                )
            # Keep routing weights in their FP32 router dtype for the multiply,
            # then cast once before accumulation into the model-dtype output.
            expert_weights = flat_weights[assignment_mask]
            weighted_output = (
                expert_output * expert_weights.unsqueeze(-1)
            ).to(output.dtype)
            output = output.index_add(0, expert_tokens, weighted_output)

        shared_output = self._shared_forward(flat_states)
        if shared_output is not None:
            output = output + shared_output.to(output.dtype)
        output = output.reshape(original_shape)

        should_return_logits = (
            self.return_router_logits
            if return_router_logits is None
            else return_router_logits
        )
        if should_return_logits:
            logits_shape = (*original_shape[:-1], self.num_experts)
            return output, router_logits.reshape(logits_shape)
        return output


# A concise name for model implementations that do not need to distinguish
# this reference implementation from a future fused backend.
MoE = SparseMoE


__all__ = [
    "GatedMLP",
    "MoE",
    "SparseMoE",
    "TopKRouter",
    "topk_softmax",
]
