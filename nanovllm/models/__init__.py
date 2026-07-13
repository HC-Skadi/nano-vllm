"""Model architecture registry.

Model implementations are imported lazily so loading nano-vLLM does not import
every optional architecture and its dependencies up front.
"""

from importlib import import_module
from typing import Any

from torch import nn


ModelEntry = tuple[str, str]

_QWEN2: ModelEntry = ("nanovllm.models.qwen2", "Qwen2ForCausalLM")
_QWEN3: ModelEntry = ("nanovllm.models.qwen3", "Qwen3ForCausalLM")
_LLAMA: ModelEntry = ("nanovllm.models.llama", "LlamaForCausalLM")
_DEEPSEEK_V2: ModelEntry = (
    "nanovllm.models.deepseek_v2",
    "DeepseekV2ForCausalLM",
)

MODEL_ARCHITECTURES: dict[str, ModelEntry] = {
    "Qwen2ForCausalLM": _QWEN2,
    "Qwen3ForCausalLM": _QWEN3,
    "LlamaForCausalLM": _LLAMA,
    "DeepseekV2ForCausalLM": _DEEPSEEK_V2,
}

MODEL_TYPES: dict[str, ModelEntry] = {
    "qwen2": _QWEN2,
    "qwen3": _QWEN3,
    "llama": _LLAMA,
    "deepseek_v2": _DEEPSEEK_V2,
}


def _resolve_model_class(entry: ModelEntry) -> type[nn.Module]:
    module_name, class_name = entry
    module = import_module(module_name)
    model_class = getattr(module, class_name)
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise TypeError(
            f"Registered model {module_name}.{class_name} is not an nn.Module class"
        )
    return model_class


def _config_architectures(config: Any) -> list[str]:
    architectures = getattr(config, "architectures", None)
    if architectures is None:
        return []
    if isinstance(architectures, str):
        return [architectures]
    return list(architectures)


def get_model_class(config: Any) -> type[nn.Module]:
    """Select the nano-vLLM model implementation for a HF config.

    Hugging Face's explicit ``architectures`` declaration takes precedence.
    ``model_type`` is retained as a fallback for local or synthetic configs that
    omit architectures, or for checkpoints whose architecture alias is not
    registered.
    """

    architectures = _config_architectures(config)
    for architecture in architectures:
        entry = MODEL_ARCHITECTURES.get(architecture)
        if entry is not None:
            return _resolve_model_class(entry)

    model_type = getattr(config, "model_type", None)
    entry = MODEL_TYPES.get(model_type)
    if entry is not None:
        return _resolve_model_class(entry)

    requested = f"architectures={architectures!r}, model_type={model_type!r}"
    supported_architectures = ", ".join(sorted(MODEL_ARCHITECTURES))
    supported_model_types = ", ".join(sorted(MODEL_TYPES))
    raise ValueError(
        "Unsupported model configuration "
        f"({requested}). Supported architectures: {supported_architectures}. "
        f"Supported model types: {supported_model_types}."
    )


__all__ = [
    "MODEL_ARCHITECTURES",
    "MODEL_TYPES",
    "get_model_class",
]
