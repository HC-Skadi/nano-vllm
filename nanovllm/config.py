import os
import json
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoConfig

from nanovllm.layers.quantization.awq import AWQConfig


@dataclass(slots=True)
class Config:
    # 模型权重目录，必须是本地已存在的 Hugging Face 格式目录。
    model: str
    # 单次调度最多处理的 token 数，用于限制 prefill/decode 的批处理规模。
    max_num_batched_tokens: int = 16384
    # 单次调度最多同时运行的序列数量。
    max_num_seqs: int = 512
    # 模型允许的最大上下文长度，初始化时会被裁剪到模型配置上限以内。
    max_model_len: int = 4096
    # 用于 KV cache 的 GPU 显存比例，剩余显存会预留给运行时开销。
    gpu_memory_utilization: float = 0.9
    # 张量并行进程数，每个进程负责一部分注意力头和权重。
    tensor_parallel_size: int = 1
    # 是否强制使用 eager 模式；False 时 decode 阶段会尝试使用 CUDA Graph。
    enforce_eager: bool = False
    # 是否允许 tokenizer 加载模型目录中的自定义代码；默认关闭。
    trust_remote_code: bool = False
    # 量化方式和 AWQ kernel 后端；默认从 checkpoint 自动识别。
    quantization: str | None = None
    awq_backend: str = "auto"
    # Hugging Face 模型配置，初始化时根据 model 自动加载。
    hf_config: AutoConfig | None = None
    # 结束符 token id，调度后处理阶段用它判断序列是否完成。
    eos: int = -1
    # KV cache 的块大小，注意力缓存按块申请和复用。
    kvcache_block_size: int = 256
    # 可用 KV cache 块数，模型预热并统计显存后自动计算。
    num_kvcache_blocks: int = -1
    quant_config: AWQConfig | None = field(init=False, default=None, repr=False)
    runtime_dtype: torch.dtype = field(init=False, default=torch.float16)

    def __post_init__(self):
        # 只支持加载本地模型目录，避免运行时隐式下载权重。
        assert os.path.isdir(self.model)
        # 块大小需要和底层 kernel/cache 布局对齐。
        assert self.kvcache_block_size % 256 == 0
        # 当前实现限制张量并行规模，通常对应单机 GPU 数量。
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        )
        # 不能超过模型自身支持的最大位置编码长度。
        architectures = set(getattr(self.hf_config, "architectures", []) or [])
        is_qwen = getattr(self.hf_config, "model_type", None) in {"qwen2", "qwen3"}
        is_qwen = is_qwen or bool(
            architectures & {"Qwen2ForCausalLM", "Qwen3ForCausalLM"}
        )
        if is_qwen:
            if getattr(self.hf_config, "use_sliding_window", False):
                raise ValueError(
                    "nano-vllm does not support Qwen sliding-window attention"
                )
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
