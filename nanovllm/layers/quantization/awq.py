"""AWQ W4A16 linear support.

The packed parameter layout and vLLM operator dispatch follow vLLM's AWQ
implementation.  vLLM itself is an optional runtime dependency: its fused
``awq_gemm`` kernel is used when available, while a small PyTorch reference
dequantizer keeps loading and correctness tests independent of the extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import warnings
from typing import Any

import torch
from torch import nn


_AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_BACKEND_ALIASES = {
    "auto": "auto",
    "gemm": "gemm",
    "fused": "gemm",
    "dequant": "dequant",
    "dequantize": "dequant",
    "unfused": "dequant",
}
_warned_about_reference_backend = False


@dataclass(frozen=True, slots=True)
class AWQConfig:
    """Configuration for standard AutoAWQ GEMM W4A16 checkpoints."""

    bits: int
    group_size: int
    zero_point: bool
    version: str = "gemm"
    backend: str = "auto"
    modules_to_not_convert: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        version = self.version.lower()
        backend = _BACKEND_ALIASES.get(self.backend.lower())
        object.__setattr__(self, "version", version)
        if backend is None:
            choices = ", ".join(sorted(_BACKEND_ALIASES))
            raise ValueError(f"Unsupported AWQ backend {self.backend!r}; choose one of: {choices}")
        object.__setattr__(self, "backend", backend)
        if self.bits != 4:
            raise ValueError(f"nano-vllm AWQ only supports 4-bit weights, got {self.bits}")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError(f"AWQ group_size must be positive or -1, got {self.group_size}")
        if not self.zero_point:
            raise ValueError("standard AWQ GEMM requires zero_point=true")
        if version != "gemm":
            raise ValueError(
                f"nano-vllm supports the standard AWQ GEMM layout, not {self.version!r}"
            )
        supported_unquantized = ("lm_head", "embed_tokens")
        unsupported = [
            name
            for name in self.modules_to_not_convert
            if not name.endswith(supported_unquantized)
        ]
        if unsupported:
            raise ValueError(
                "Selectively unquantized transformer projections are not supported: "
                f"{unsupported}"
            )

    @property
    def pack_factor(self) -> int:
        return 32 // self.bits

    @classmethod
    def from_dict(cls, config: dict[str, Any], backend: str = "auto") -> "AWQConfig":
        quant_method = str(
            config.get("quant_method", config.get("quantization_method", "awq"))
        ).lower()
        if quant_method != "awq":
            raise ValueError(f"Expected an AWQ checkpoint, got quant_method={quant_method!r}")

        def required(*keys: str) -> Any:
            for key in keys:
                if key in config:
                    return config[key]
            raise ValueError(f"AWQ configuration is missing one of {keys}")

        modules = config.get("modules_to_not_convert") or ()
        zero_point = required("zero_point")
        if not isinstance(zero_point, bool):
            raise ValueError(f"AWQ zero_point must be a boolean, got {zero_point!r}")
        return cls(
            bits=int(required("bits", "w_bit")),
            group_size=int(required("group_size", "q_group_size")),
            zero_point=zero_point,
            version=str(config.get("version", "gemm")),
            backend=backend,
            modules_to_not_convert=tuple(modules),
        )

    def normalized_group_size(self, input_size: int) -> int:
        return input_size if self.group_size == -1 else self.group_size


@lru_cache(maxsize=1)
def _get_vllm_ops():
    try:
        from vllm import _custom_ops as ops
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "The fused AWQ backend requires a vLLM installation built for the "
            "current PyTorch/CUDA ABI. Install nano-vllm[awq] or select "
            "awq_backend='dequant' for the reference fallback."
        ) from exc
    if not callable(getattr(ops, "awq_gemm", None)) or not callable(
        getattr(ops, "awq_dequantize", None)
    ):
        raise RuntimeError("This vLLM build does not expose awq_gemm and awq_dequantize")
    return ops


@lru_cache(maxsize=1)
def _try_get_vllm_ops():
    try:
        return _get_vllm_ops()
    except RuntimeError:
        return None


def vllm_awq_ops_available() -> bool:
    return _try_get_vllm_ops() is not None


@lru_cache(maxsize=None)
def _get_device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _unpack_awq(values: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=values.device)
    unpacked = torch.bitwise_right_shift(values.unsqueeze(-1), shifts)
    unpacked = torch.bitwise_and(unpacked, 0xF).reshape(values.shape[0], -1)
    order = torch.tensor(_AWQ_REVERSE_ORDER, dtype=torch.long, device=values.device)
    return unpacked.reshape(values.shape[0], -1, 8).index_select(-1, order).reshape(
        values.shape[0], -1
    )


def dequantize_awq_reference(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize standard AWQ GEMM tensors using only PyTorch.

    This is intentionally a correctness/fallback path, not a high-performance
    kernel.  Its output layout is ``[input_features, output_features]``.
    """

    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AWQ qweight and qzeros must be int32")
    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("AWQ qweight, qzeros, and scales must all be rank-2")
    if qzeros.shape[0] == 0 or qweight.shape[0] % qzeros.shape[0] != 0:
        raise ValueError("AWQ qzeros do not describe an integral input group size")
    if qzeros.shape[1] != qweight.shape[1] or scales.shape[1] != qweight.shape[1] * 8:
        raise ValueError("AWQ packed output dimensions are inconsistent")
    if scales.shape[0] != qzeros.shape[0]:
        raise ValueError("AWQ scales and qzeros must have the same number of groups")

    group_size = qweight.shape[0] // qzeros.shape[0]
    int_weight = _unpack_awq(qweight).to(scales.dtype)
    zeros = _unpack_awq(qzeros).to(scales.dtype).repeat_interleave(group_size, dim=0)
    expanded_scales = scales.repeat_interleave(group_size, dim=0)
    return (int_weight - zeros) * expanded_scales


def apply_awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    backend: str = "auto",
    bias: torch.Tensor | None = None,
    pack_factor: int = 8,
) -> torch.Tensor:
    """Apply an AWQ linear projection through fused or dequantized dispatch."""

    canonical_backend = _BACKEND_ALIASES.get(backend.lower())
    if canonical_backend is None:
        raise ValueError(f"Unsupported AWQ backend: {backend!r}")
    if pack_factor != 8:
        raise ValueError(f"Only the AWQ W4 pack factor 8 is supported, got {pack_factor}")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError("AWQ qweight and qzeros must be int32")
    if (
        not qweight.is_contiguous()
        or not qzeros.is_contiguous()
        or not scales.is_contiguous()
    ):
        raise ValueError("AWQ qweight, qzeros, and scales must be contiguous")
    if x.shape[-1] != qweight.shape[0]:
        raise ValueError(
            f"AWQ input dimension mismatch: x={x.shape[-1]}, qweight={qweight.shape[0]}"
        )

    output_shape = x.shape[:-1] + (qweight.shape[1] * pack_factor,)
    x_2d = x.reshape(-1, x.shape[-1])
    selected_backend = canonical_backend
    if selected_backend == "auto":
        selected_backend = "gemm" if x_2d.shape[0] < 256 else "dequant"

    ops = _try_get_vllm_ops() if x.is_cuda else None
    if ops is not None:
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if _get_device_capability(device_index) < (7, 5):
            raise RuntimeError("vLLM AWQ operators require compute capability 7.5 or newer")
        if x.dtype != torch.float16 or scales.dtype != torch.float16:
            raise TypeError(
                "vLLM AWQ operators require FP16 activations and scales, got "
                f"x={x.dtype}, scales={scales.dtype}"
            )
    if selected_backend == "gemm":
        if not x.is_cuda:
            if canonical_backend == "gemm":
                raise RuntimeError("vLLM AWQ GEMM requires a CUDA tensor")
            selected_backend = "dequant"
        elif ops is None:
            # An explicit fused request must never silently become the baseline.
            if canonical_backend == "gemm":
                _get_vllm_ops()
            selected_backend = "dequant"

    if selected_backend == "gemm":
        out = ops.awq_gemm(x_2d, qweight, scales, qzeros, pack_factor)
    else:
        if ops is not None:
            weight = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
        else:
            global _warned_about_reference_backend
            if x.is_cuda and not _warned_about_reference_backend:
                warnings.warn(
                    "vLLM AWQ operators are unavailable; using the slow PyTorch "
                    "dequantization fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warned_about_reference_backend = True
            weight = dequantize_awq_reference(qweight, qzeros, scales)
        out = torch.matmul(x_2d, weight.to(x.dtype))

    if bias is not None:
        out.add_(bias)
    return out.reshape(output_shape)


class AWQLinearMethod:
    """Creates packed AWQ parameters and applies the selected vLLM operator."""

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        group_size = self.quant_config.normalized_group_size(input_size)
        pack_factor = self.quant_config.pack_factor
        if input_size % group_size != 0:
            raise ValueError(
                f"AWQ input partition {input_size} is not divisible by group_size {group_size}"
            )
        if output_size % pack_factor != 0:
            raise ValueError(
                f"AWQ output partition {output_size} is not divisible by pack_factor {pack_factor}"
            )
        params_dtype = params_dtype or torch.get_default_dtype()
        tensors = {
            "qweight": torch.empty(
                input_size,
                output_size // pack_factor,
                dtype=torch.int32,
            ),
            "qzeros": torch.empty(
                input_size // group_size, output_size // pack_factor, dtype=torch.int32
            ),
            "scales": torch.empty(input_size // group_size, output_size, dtype=params_dtype),
        }
        for name, tensor in tensors.items():
            param = nn.Parameter(tensor, requires_grad=False)
            param.weight_loader = layer.weight_loader
            param.awq_kind = name
            layer.register_parameter(name, param)
        layer.awq_group_size = group_size

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_awq_linear(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            backend=self.quant_config.backend,
            bias=bias,
            pack_factor=self.quant_config.pack_factor,
        )
