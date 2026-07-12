import os
import json
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoConfig

from nanovllm.layers.quantization.awq import AWQConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    quantization: str | None = None
    awq_backend: str = "auto"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    quant_config: AWQConfig | None = field(init=False, default=None, repr=False)
    runtime_dtype: torch.dtype = field(init=False, default=torch.float16)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if getattr(self.hf_config, "use_sliding_window", False):
            raise ValueError("nano-vllm does not support Qwen sliding-window attention")
        rope_scaling = getattr(self.hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type not in (None, "default"):
                raise ValueError(
                    f"nano-vllm does not support Qwen rope_type={rope_type!r}"
                )
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        raw_quant_config = self._load_quant_config()
        inferred_quantization = None
        if raw_quant_config:
            raw_method = raw_quant_config.get(
                "quant_method", raw_quant_config.get("quantization_method")
            )
            if raw_method is not None:
                inferred_quantization = str(raw_method).lower() or None
            if inferred_quantization is None:
                has_bits = "bits" in raw_quant_config or "w_bit" in raw_quant_config
                has_group = (
                    "group_size" in raw_quant_config
                    or "q_group_size" in raw_quant_config
                )
                if has_bits and has_group and "zero_point" in raw_quant_config:
                    inferred_quantization = "awq"
        if self.quantization is None:
            self.quantization = inferred_quantization
        elif inferred_quantization and self.quantization.lower() != inferred_quantization:
            raise ValueError(
                f"Requested quantization={self.quantization!r}, but the checkpoint uses "
                f"{inferred_quantization!r}"
            )
        if self.quantization is not None:
            self.quantization = self.quantization.lower()
            if self.quantization != "awq":
                raise ValueError(f"Unsupported quantization method: {self.quantization!r}")
            if not raw_quant_config:
                raise ValueError("AWQ was requested, but the model has no AWQ quantization config")
            self.quant_config = AWQConfig.from_dict(
                raw_quant_config,
                backend=self.awq_backend,
            )
            self.runtime_dtype = torch.float16
        else:
            self.runtime_dtype = self._get_hf_dtype()

    def _load_quant_config(self) -> dict[str, Any] | None:
        embedded = None
        raw = getattr(self.hf_config, "quantization_config", None)
        if raw:
            if hasattr(raw, "to_dict"):
                raw = raw.to_dict()
            if not isinstance(raw, dict):
                raise ValueError("Embedded quantization_config must be a JSON object")
            embedded = dict(raw)
        sidecar = None
        for filename in ("quant_config.json", "quantize_config.json"):
            path = os.path.join(self.model, filename)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    sidecar = json.load(f)
                if not isinstance(sidecar, dict):
                    raise ValueError(f"{filename} must contain a JSON object")
                break
        if embedded is not None and sidecar is not None:
            return {**sidecar, **embedded}
        return embedded if embedded is not None else sidecar

    def _get_hf_dtype(self) -> torch.dtype:
        dtype = getattr(self.hf_config, "dtype", None)
        if dtype is None:
            dtype = getattr(self.hf_config, "torch_dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            dtype = dtype.removeprefix("torch.")
            resolved = getattr(torch, dtype, None)
            if isinstance(resolved, torch.dtype):
                return resolved
        return torch.float16
