import unittest

import torch

from nanovllm.layers.layernorm import RMSNorm


class RMSNormTest(unittest.TestCase):

    def test_fp32_forward_does_not_overwrite_input(self):
        norm = RMSNorm(4, eps=1e-6)
        hidden_states = torch.tensor(
            [[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32
        )
        original = hidden_states.clone()

        with torch.inference_mode():
            actual = norm(hidden_states)

        variance = original.square().mean(dim=-1, keepdim=True)
        expected = original * torch.rsqrt(variance + norm.eps)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hidden_states, original)

    def test_fp32_fused_add_preserves_unnormalized_residual(self):
        norm = RMSNorm(4, eps=1e-6)
        update = torch.tensor([[0.5, -1.0, 2.0, 1.0]], dtype=torch.float32)
        residual = torch.tensor([[1.0, 2.0, -3.0, 4.0]], dtype=torch.float32)
        expected_residual = update + residual

        with torch.inference_mode():
            actual, saved_residual = norm(update, residual)

        variance = expected_residual.square().mean(dim=-1, keepdim=True)
        expected = expected_residual * torch.rsqrt(variance + norm.eps)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(saved_residual, expected_residual)


if __name__ == "__main__":
    unittest.main()
