import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded: set[tuple[str, str | int | None]] = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                name_parts = weight_name.split(".")
                for source_name, (target_name, shard_id) in packed_modules_mapping.items():
                    if source_name in name_parts:
                        mapped_parts = name_parts.copy()
                        mapped_parts[mapped_parts.index(source_name)] = target_name
                        param_name = ".".join(mapped_parts)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded.add((param_name, shard_id))
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded.add((weight_name, None))

    target_shards: dict[str, set[str | int]] = {}
    for target_name, shard_id in packed_modules_mapping.values():
        target_shards.setdefault(target_name, set()).add(shard_id)

    parameters = dict(model.named_parameters())
    loaded_regular_ptrs = {
        parameters[name].data_ptr()
        for name, shard_id in loaded
        if shard_id is None and name in parameters
    }
    missing = []
    for param_name, param in parameters.items():
        expected_shards: set[str | int | None] = {None}
        name_parts = param_name.split(".")
        for target_name, shard_ids in target_shards.items():
            if target_name in name_parts:
                expected_shards = shard_ids
                break
        for shard_id in expected_shards:
            if (param_name, shard_id) in loaded:
                continue
            if shard_id is None and param.data_ptr() in loaded_regular_ptrs:
                # Tied embedding/LM-head parameters can share storage while only
                # one name is present in the checkpoint.
                continue
            missing.append(
                param_name if shard_id is None else f"{param_name}[shard={shard_id!r}]"
            )
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise RuntimeError(
            f"Checkpoint did not initialize {len(missing)} model parameter shards: "
            f"{preview}{suffix}"
        )
