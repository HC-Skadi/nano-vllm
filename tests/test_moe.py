import unittest

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.moe import GatedMLP, SparseMoE, TopKRouter, topk_softmax


class CountingExpert(nn.Module):

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.batch_sizes = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(hidden_states.shape[0])
        return hidden_states * self.scale


class MoETest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_topk_softmax_normalization_is_optional(self):
        logits = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 0.0, 1.0]])
        probabilities = torch.softmax(logits, dim=-1)
        expected_weights, expected_indices = probabilities.topk(2, dim=-1)

        weights, indices = topk_softmax(logits, 2, normalize=False)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(indices, expected_indices)

        normalized, normalized_indices = topk_softmax(
            logits, 2, normalize=True
        )
        torch.testing.assert_close(normalized_indices, expected_indices)
        torch.testing.assert_close(
            normalized, expected_weights / expected_weights.sum(-1, keepdim=True)
        )
        torch.testing.assert_close(normalized.sum(-1), torch.ones(2))

    def test_router_scaling_and_logits(self):
        gate = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            gate.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            )
        router = TopKRouter(
            2,
            3,
            2,
            gate=gate,
            normalize_topk_weights=False,
            routed_scaling_factor=1.75,
        )
        hidden_states = torch.tensor([[2.0, -1.0], [-1.0, -2.0]])

        logits, weights, indices = router(hidden_states)
        expected_logits = F.linear(hidden_states, gate.weight)
        expected_weights, expected_indices = torch.softmax(
            expected_logits, dim=-1
        ).topk(2, dim=-1)
        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(indices, expected_indices)
        torch.testing.assert_close(weights, expected_weights * 1.75)

    def test_sparse_moe_batches_each_expert_and_restores_shape(self):
        gate = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            gate.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            )
        experts = [CountingExpert(1.0), CountingExpert(2.0), CountingExpert(4.0)]
        moe = SparseMoE(
            hidden_size=2,
            num_experts=3,
            num_experts_per_tok=2,
            experts=experts,
            gate=gate,
            norm_topk_prob=True,
        )
        self.assertIn("gate.weight", moe.state_dict())
        self.assertNotIn("router.gate.weight", moe.state_dict())
        hidden_states = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0]], [[-2.0, -2.0], [1.0, 1.0]]]
        )

        actual, router_logits = moe(
            hidden_states, return_router_logits=True
        )
        flat_states = hidden_states.flatten(0, -2)
        expected_logits = F.linear(flat_states, gate.weight)
        probabilities = torch.softmax(expected_logits, dim=-1)
        weights, selected = probabilities.topk(2, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        scales = torch.tensor([1.0, 2.0, 4.0])
        token_scales = (weights * scales[selected]).sum(-1)
        expected = (flat_states * token_scales.unsqueeze(-1)).view_as(hidden_states)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            router_logits, expected_logits.view(2, 2, 3)
        )
        assignment_counts = torch.bincount(selected.flatten(), minlength=3)
        for expert_index, expert in enumerate(experts):
            self.assertEqual(
                expert.batch_sizes, [assignment_counts[expert_index].item()]
            )

    def test_shared_expert_with_sigmoid_gate(self):
        routed_expert = CountingExpert(2.0)
        shared_expert = CountingExpert(4.0)
        router_gate = nn.Linear(3, 1, bias=False)
        shared_gate = nn.Linear(3, 1, bias=False)
        nn.init.zeros_(router_gate.weight)
        nn.init.zeros_(shared_gate.weight)
        moe = SparseMoE(
            hidden_size=3,
            num_experts=1,
            top_k=1,
            experts=[routed_expert],
            gate=router_gate,
            shared_experts=shared_expert,
            shared_expert_gate=shared_gate,
        )
        hidden_states = torch.randn(2, 3, 3)

        # Routed output is 2*x.  sigmoid(0) gates shared output to 0.5*4*x.
        torch.testing.assert_close(moe(hidden_states), 4.0 * hidden_states)
        self.assertEqual(routed_expert.batch_sizes, [6])
        self.assertEqual(shared_expert.batch_sizes, [6])

    def test_gated_mlp_accepts_a_linear_factory(self):
        calls = []

        def linear_factory(input_size, output_size, bias):
            calls.append((input_size, output_size, bias))
            return nn.Linear(input_size, output_size, bias=bias)

        expert = GatedMLP(
            2, 3, activation=torch.sigmoid, linear_factory=linear_factory
        )
        self.assertEqual(
            calls, [(2, 3, False), (2, 3, False), (3, 2, False)]
        )
        hidden_states = torch.randn(5, 2, requires_grad=True)
        expected = F.linear(
            torch.sigmoid(F.linear(hidden_states, expert.gate_proj.weight))
            * F.linear(hidden_states, expert.up_proj.weight),
            expert.down_proj.weight,
        )
        actual = expert(hidden_states)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(expert.gate_proj.weight.grad)

    def test_default_experts_support_backward_and_empty_tokens(self):
        moe = SparseMoE(
            hidden_size=4,
            num_experts=3,
            top_k=2,
            intermediate_size=8,
            return_router_logits=True,
        )
        hidden_states = torch.randn(2, 3, 4, requires_grad=True)
        output, router_logits = moe(hidden_states)
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(router_logits.shape, (2, 3, 3))
        output.square().mean().backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(moe.gate.weight.grad)

        empty_output, empty_logits = moe(torch.empty(0, 4))
        self.assertEqual(empty_output.shape, (0, 4))
        self.assertEqual(empty_logits.shape, (0, 3))

    def test_constructor_rejects_conflicting_config_aliases(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            SparseMoE(
                hidden_size=2,
                num_experts=2,
                top_k=1,
                num_experts_per_tok=2,
                experts=[CountingExpert(1.0), CountingExpert(1.0)],
            )


if __name__ == "__main__":
    unittest.main()
