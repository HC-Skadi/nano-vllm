import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file


# Importing any nanovllm submodule executes nanovllm/__init__.py, which imports
# the attention module.  The AWQ unit tests are CPU-only and never invoke
# attention, so provide a test-local placeholder when FlashAttention is absent.
if importlib.util.find_spec("flash_attn") is None:
    flash_attn = types.ModuleType("flash_attn")

    def _unavailable_flash_attention(*args, **kwargs):
        raise RuntimeError("FlashAttention is unavailable in this CPU-only test")

    flash_attn.flash_attn_varlen_func = _unavailable_flash_attention
    flash_attn.flash_attn_with_kvcache = _unavailable_flash_attention
    sys.modules["flash_attn"] = flash_attn


from transformers import Qwen2Config, Qwen3Config

from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from nanovllm.config import Config
from nanovllm.layers.quantization.awq import (
    AWQConfig,
    apply_awq_linear,
    dequantize_awq_reference,
)
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model


@contextmanager
def default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextmanager
def mock_tensor_parallel(world_size=1, rank=0):
    with patch("torch.distributed.get_world_size", return_value=world_size), patch(
        "torch.distributed.get_rank", return_value=rank
    ):
        yield


def pack_awq_reference(values):
    """Pack logical AWQ nibbles into the standard GEMM int32 layout."""

    values = torch.as_tensor(values, dtype=torch.int64)
    if values.ndim != 2 or values.shape[1] % 8:
        raise ValueError("logical AWQ values must be [rows, columns divisible by 8]")
    if torch.any(values < 0) or torch.any(values > 15):
        raise ValueError("AWQ nibbles must be in [0, 15]")

    logical = values.reshape(values.shape[0], -1, 8)
    # dequantize_awq_reference reads packed nibbles in the order
    # [0, 4, 1, 5, 2, 6, 3, 7], so this is the inverse permutation.
    inverse_order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    packed_order = logical.index_select(-1, inverse_order)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return torch.sum(packed_order << shifts, dim=-1).to(torch.int32)


def make_reference_awq_tensors():
    logical_weight = (torch.arange(4 * 16).reshape(4, 16) % 8).to(torch.int32)
    logical_zeros = torch.stack(
        [
            torch.arange(16, dtype=torch.int32) % 3,
            (torch.arange(16, dtype=torch.int32) + 1) % 3,
        ]
    )
    scales = torch.linspace(0.125, 1.0, 32, dtype=torch.float32).reshape(2, 16)
    qweight = pack_awq_reference(logical_weight)
    qzeros = pack_awq_reference(logical_zeros)
    expected_weight = (
        logical_weight.float() - logical_zeros.repeat_interleave(2, dim=0).float()
    ) * scales.repeat_interleave(2, dim=0)
    return qweight, qzeros, scales, expected_weight


class AWQConfigTest(unittest.TestCase):
    def test_from_dict_accepts_checkpoint_aliases_and_normalizes_values(self):
        config = AWQConfig.from_dict(
            {
                "quantization_method": "AWQ",
                "w_bit": "4",
                "q_group_size": "128",
                "zero_point": True,
                "version": "GEMM",
                "modules_to_not_convert": ["lm_head", "model.embed_tokens"],
            },
            backend="fused",
        )

        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)
        self.assertEqual(config.version, "gemm")
        self.assertEqual(config.backend, "gemm")
        self.assertEqual(config.pack_factor, 8)
        self.assertEqual(
            config.modules_to_not_convert,
            ("lm_head", "model.embed_tokens"),
        )

        per_channel = AWQConfig(4, -1, True, backend="dequantize")
        self.assertEqual(per_channel.backend, "dequant")
        self.assertEqual(per_channel.normalized_group_size(96), 96)

    def test_rejects_invalid_awq_configurations(self):
        invalid_cases = (
            ("bits", lambda: AWQConfig(8, 128, True)),
            ("group_size zero", lambda: AWQConfig(4, 0, True)),
            ("group_size below minus one", lambda: AWQConfig(4, -2, True)),
            ("zero point", lambda: AWQConfig(4, 128, False)),
            ("layout", lambda: AWQConfig(4, 128, True, version="gemv")),
            ("backend", lambda: AWQConfig(4, 128, True, backend="unknown")),
            (
                "quantization method",
                lambda: AWQConfig.from_dict(
                    {
                        "quant_method": "gptq",
                        "bits": 4,
                        "group_size": 128,
                        "zero_point": True,
                    }
                ),
            ),
            (
                "missing required field",
                lambda: AWQConfig.from_dict(
                    {"quant_method": "awq", "group_size": 128, "zero_point": True}
                ),
            ),
            (
                "non-boolean zero point",
                lambda: AWQConfig.from_dict(
                    {
                        "quant_method": "awq",
                        "bits": 4,
                        "group_size": 128,
                        "zero_point": "false",
                    }
                ),
            ),
        )
        for name, constructor in invalid_cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                constructor()

    def test_runtime_config_detects_awq_and_forces_fp16(self):
        model_config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "bfloat16",
            "vocab_size": 32,
            "quantization_config": {
                "bits": 4,
                "group_size": 8,
                "quant_method": "awq",
                "version": "gemm",
                "zero_point": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(model_config),
                encoding="utf-8",
            )
            config = Config(directory, max_model_len=64, awq_backend="dequant")

        self.assertEqual(config.quantization, "awq")
        self.assertIsNotNone(config.quant_config)
        self.assertEqual(config.quant_config.backend, "dequant")
        self.assertEqual(config.runtime_dtype, torch.float16)

    def test_legacy_external_awq_config_without_quant_method_is_detected(self):
        model_config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "float16",
            "vocab_size": 32,
        }
        legacy_quant_config = {
            "w_bit": 4,
            "q_group_size": 8,
            "zero_point": True,
            "version": "GEMM",
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(model_config),
                encoding="utf-8",
            )
            Path(directory, "quant_config.json").write_text(
                json.dumps(legacy_quant_config),
                encoding="utf-8",
            )
            config = Config(directory, max_model_len=64)

        self.assertEqual(config.quantization, "awq")
        self.assertEqual(config.quant_config.bits, 4)
        self.assertEqual(config.quant_config.group_size, 8)


class AWQReferenceTest(unittest.TestCase):
    def test_pack_then_dequantize_matches_logical_awq_formula(self):
        qweight, qzeros, scales, expected_weight = make_reference_awq_tensors()

        actual_weight = dequantize_awq_reference(qweight, qzeros, scales)

        self.assertEqual(actual_weight.shape, (4, 16))
        torch.testing.assert_close(actual_weight, expected_weight)

    def test_apply_dequant_backend_supports_2d_and_3d_inputs(self):
        qweight, qzeros, scales, weight = make_reference_awq_tensors()
        bias = torch.linspace(-0.5, 0.5, 16)
        inputs = (
            torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 7,
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 11,
        )

        for x in inputs:
            with self.subTest(shape=tuple(x.shape)):
                actual = apply_awq_linear(
                    x,
                    qweight,
                    qzeros,
                    scales,
                    backend="dequant",
                    bias=bias,
                )
                expected = torch.matmul(x, weight) + bias
                self.assertEqual(actual.shape, x.shape[:-1] + (16,))
                self.assertEqual(actual.dtype, x.dtype)
                torch.testing.assert_close(actual, expected)


class AWQModelShapeTest(unittest.TestCase):
    def test_tiny_qwen2_builds_all_awq_projection_shapes_in_fp16(self):
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_bias=True,
            tie_word_embeddings=False,
        )
        quant_config = AWQConfig(4, 4, True, backend="dequant")

        with mock_tensor_parallel(), default_dtype(torch.float16):
            model = Qwen2ForCausalLM(config, quant_config=quant_config)

        layer = model.model.layers[0]
        qkv = layer.self_attn.qkv_proj
        o_proj = layer.self_attn.o_proj
        gate_up = layer.mlp.gate_up_proj
        down = layer.mlp.down_proj

        expected_shapes = {
            qkv: ((16, 4), (4, 4), (4, 32)),
            o_proj: ((16, 2), (4, 2), (4, 16)),
            gate_up: ((16, 6), (4, 6), (4, 48)),
            down: ((24, 2), (6, 2), (6, 16)),
        }
        for projection, (qweight_shape, qzeros_shape, scales_shape) in expected_shapes.items():
            with self.subTest(projection=type(projection).__name__):
                self.assertIsNone(projection.weight)
                self.assertEqual(projection.qweight.shape, qweight_shape)
                self.assertEqual(projection.qzeros.shape, qzeros_shape)
                self.assertEqual(projection.scales.shape, scales_shape)
                self.assertEqual(projection.qweight.dtype, torch.int32)
                self.assertEqual(projection.qzeros.dtype, torch.int32)
                self.assertEqual(projection.scales.dtype, torch.float16)
                self.assertFalse(projection.qweight.requires_grad)
                self.assertFalse(projection.qzeros.requires_grad)
                self.assertFalse(projection.scales.requires_grad)

        self.assertFalse(layer.self_attn.use_qk_norm)
        self.assertFalse(hasattr(layer.self_attn, "q_norm"))

    def test_tiny_qwen3_keeps_qk_norms_with_awq_projections(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            attention_bias=False,
            tie_word_embeddings=False,
        )
        quant_config = AWQConfig(4, 4, True, backend="dequant")

        with mock_tensor_parallel(), default_dtype(torch.float16):
            model = Qwen3ForCausalLM(config, quant_config=quant_config)

        attention = model.model.layers[0].self_attn
        self.assertTrue(attention.use_qk_norm)
        self.assertTrue(hasattr(attention, "q_norm"))
        self.assertTrue(hasattr(attention, "k_norm"))


class AWQWeightLoaderTest(unittest.TestCase):
    quant_config = AWQConfig(4, 4, True, backend="dequant")

    def test_qkv_loader_places_q_k_v_for_every_awq_parameter(self):
        with mock_tensor_parallel(), default_dtype(torch.float16):
            layer = QKVParallelLinear(
                hidden_size=8,
                head_size=8,
                total_num_heads=2,
                total_num_kv_heads=1,
                quant_config=self.quant_config,
            )

        logical_output_sizes = (16, 8, 8)
        for kind in ("qweight", "qzeros", "scales"):
            param = getattr(layer, kind)
            param.data.zero_()
            rows = param.shape[0]
            divisor = 8 if kind in ("qweight", "qzeros") else 1
            loaded = [
                torch.full(
                    (rows, output_size // divisor),
                    fill_value=value,
                    dtype=param.dtype,
                )
                for output_size, value in zip(logical_output_sizes, (11, 22, 33))
            ]
            for tensor, shard_id in zip(loaded, ("q", "k", "v")):
                layer.weight_loader(param, tensor, shard_id)

            with self.subTest(kind=kind):
                torch.testing.assert_close(param, torch.cat(loaded, dim=1))

    def test_merged_loader_places_each_source_for_every_awq_parameter(self):
        with mock_tensor_parallel(), default_dtype(torch.float16):
            layer = MergedColumnParallelLinear(
                input_size=8,
                output_sizes=[16, 24],
                quant_config=self.quant_config,
            )

        for kind in ("qweight", "qzeros", "scales"):
            param = getattr(layer, kind)
            param.data.zero_()
            rows = param.shape[0]
            divisor = 8 if kind in ("qweight", "qzeros") else 1
            loaded = [
                torch.full((rows, 16 // divisor), 5, dtype=param.dtype),
                torch.full((rows, 24 // divisor), 9, dtype=param.dtype),
            ]
            for shard_id, tensor in enumerate(loaded):
                layer.weight_loader(param, tensor, shard_id)

            with self.subTest(kind=kind):
                torch.testing.assert_close(param, torch.cat(loaded, dim=1))

    def test_column_parallel_loader_selects_rank_output_partition(self):
        with mock_tensor_parallel(world_size=2, rank=1), default_dtype(torch.float16):
            layer = ColumnParallelLinear(
                input_size=8,
                output_size=32,
                quant_config=self.quant_config,
            )

        full_shapes = {
            "qweight": (8, 4),
            "qzeros": (2, 4),
            "scales": (2, 32),
        }
        for kind, shape in full_shapes.items():
            param = getattr(layer, kind)
            loaded = torch.arange(
                shape[0] * shape[1], dtype=param.dtype
            ).reshape(shape)
            layer.weight_loader(param, loaded)
            expected = loaded.narrow(1, param.shape[1], param.shape[1])
            with self.subTest(kind=kind):
                torch.testing.assert_close(param, expected)

    def test_row_parallel_loader_selects_rank_input_groups(self):
        with mock_tensor_parallel(world_size=2, rank=1), default_dtype(torch.float16):
            layer = RowParallelLinear(
                input_size=16,
                output_size=16,
                quant_config=self.quant_config,
            )

        full_shapes = {
            "qweight": (16, 2),
            "qzeros": (4, 2),
            "scales": (4, 16),
        }
        for kind, shape in full_shapes.items():
            param = getattr(layer, kind)
            loaded = torch.arange(
                shape[0] * shape[1], dtype=param.dtype
            ).reshape(shape)
            layer.weight_loader(param, loaded)
            expected = loaded.narrow(0, param.shape[0], param.shape[0])
            with self.subTest(kind=kind):
                torch.testing.assert_close(param, expected)

    def test_qkv_loader_combines_rank_one_packed_partitions(self):
        with mock_tensor_parallel(world_size=2, rank=1), default_dtype(torch.float16):
            layer = QKVParallelLinear(
                hidden_size=8,
                head_size=8,
                total_num_heads=4,
                total_num_kv_heads=2,
                quant_config=self.quant_config,
            )

        logical_output_sizes = (32, 16, 16)
        for kind in ("qweight", "qzeros", "scales"):
            param = getattr(layer, kind)
            rows = param.shape[0]
            divisor = 8 if kind in ("qweight", "qzeros") else 1
            loaded = []
            expected = []
            for shard_id, (output_size, offset) in zip(
                ("q", "k", "v"),
                zip(logical_output_sizes, (100, 200, 300)),
            ):
                width = output_size // divisor
                tensor = torch.arange(rows * width, dtype=param.dtype).reshape(
                    rows, width
                ) + offset
                loaded.append((shard_id, tensor))
                expected.append(tensor.chunk(2, dim=1)[1])
            for shard_id, tensor in loaded:
                layer.weight_loader(param, tensor, shard_id)
            with self.subTest(kind=kind):
                torch.testing.assert_close(param, torch.cat(expected, dim=1))

    def test_merged_loader_combines_rank_one_packed_partitions(self):
        with mock_tensor_parallel(world_size=2, rank=1), default_dtype(torch.float16):
            layer = MergedColumnParallelLinear(
                input_size=8,
                output_sizes=[32, 48],
                quant_config=self.quant_config,
            )

        for kind in ("qweight", "qzeros", "scales"):
            param = getattr(layer, kind)
            rows = param.shape[0]
            divisor = 8 if kind in ("qweight", "qzeros") else 1
            expected = []
            for shard_id, (output_size, offset) in enumerate(((32, 100), (48, 200))):
                width = output_size // divisor
                tensor = torch.arange(rows * width, dtype=param.dtype).reshape(
                    rows, width
                ) + offset
                layer.weight_loader(param, tensor, shard_id)
                expected.append(tensor.chunk(2, dim=1)[1])
            with self.subTest(kind=kind):
                torch.testing.assert_close(param, torch.cat(expected, dim=1))

    def test_model_loader_maps_all_shards_and_rejects_incomplete_checkpoint(self):
        class PackedModel(nn.Module):
            packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
            }

            def __init__(self):
                super().__init__()
                self.qkv_proj = QKVParallelLinear(
                    hidden_size=8,
                    head_size=8,
                    total_num_heads=2,
                    total_num_kv_heads=1,
                    quant_config=AWQWeightLoaderTest.quant_config,
                )

        checkpoint = {}
        expected = {"qweight": [], "qzeros": [], "scales": []}
        logical_output_sizes = {"q": 16, "k": 8, "v": 8}
        for kind in expected:
            rows = 8 if kind == "qweight" else 2
            divisor = 8 if kind in ("qweight", "qzeros") else 1
            for index, (shard, output_size) in enumerate(logical_output_sizes.items()):
                tensor = torch.full(
                    (rows, output_size // divisor),
                    index + 1,
                    dtype=torch.int32 if kind != "scales" else torch.float16,
                )
                checkpoint[f"{shard}_proj.{kind}"] = tensor
                expected[kind].append(tensor)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock_tensor_parallel(),
            default_dtype(torch.float16),
        ):
            save_file(checkpoint, str(Path(directory, "model.safetensors")))
            model = PackedModel()
            load_model(model, directory)
            for kind, tensors in expected.items():
                torch.testing.assert_close(
                    getattr(model.qkv_proj, kind),
                    torch.cat(tensors, dim=1),
                )

        incomplete = checkpoint.copy()
        del incomplete["v_proj.scales"]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock_tensor_parallel(),
            default_dtype(torch.float16),
        ):
            save_file(incomplete, str(Path(directory, "model.safetensors")))
            model = PackedModel()
            with self.assertRaisesRegex(RuntimeError, "qkv_proj.scales.*shard='v'"):
                load_model(model, directory)


if __name__ == "__main__":
    unittest.main()
