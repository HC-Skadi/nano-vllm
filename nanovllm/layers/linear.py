import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.quantization.awq import AWQConfig, AWQLinearMethod


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _awq_kind(param: nn.Parameter) -> str | None:
    return getattr(param, "awq_kind", None)


def _awq_output_divisor(param: nn.Parameter, pack_factor: int) -> int:
    return pack_factor if _awq_kind(param) in ("qweight", "qzeros") else 1


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: AWQConfig | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_method = AWQLinearMethod(quant_config) if quant_config else None
        if self.quant_method is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.weight.weight_loader = self.weight_loader
        else:
            self.register_parameter("weight", None)
            self.quant_method.create_weights(self, input_size, output_size)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, bias)
        return F.linear(x, self.weight, bias)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_linear(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(
            input_size,
            divide(output_size, tp_size),
            bias,
            0,
            quant_config=quant_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_dim = 1 if _awq_kind(param) else self.tp_dim
        shard_size = param_data.size(shard_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        self.output_sizes = output_sizes
        if quant_config is not None:
            tp_size = dist.get_world_size()
            for output_size in output_sizes:
                alignment = tp_size * quant_config.pack_factor
                if output_size % alignment:
                    raise ValueError(
                        f"AWQ merged output {output_size} is not divisible by {alignment}"
                    )
        super().__init__(input_size, sum(output_sizes), bias, quant_config=quant_config)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        awq_kind = _awq_kind(param)
        if awq_kind:
            pack_factor = self.quant_method.quant_config.pack_factor
            divisor = _awq_output_divisor(param, pack_factor)
            shard_dim = 1
        else:
            divisor = 1
            shard_dim = self.tp_dim
        shard_offset = divide(sum(self.output_sizes[:loaded_shard_id]), self.tp_size * divisor)
        shard_size = divide(self.output_sizes[loaded_shard_id], self.tp_size * divisor)
        param_data = param.data.narrow(shard_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, shard_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        if quant_config is not None:
            q_size = self.num_heads * head_size
            kv_size = self.num_kv_heads * head_size
            if q_size % quant_config.pack_factor or kv_size % quant_config.pack_factor:
                raise ValueError(
                    "Every AWQ Q/K/V tensor-parallel output shard must be "
                    f"divisible by pack_factor {quant_config.pack_factor}"
                )
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config=quant_config)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            logical_shard_size = self.num_heads * self.head_size
            logical_shard_offset = 0
        elif loaded_shard_id == "k":
            logical_shard_size = self.num_kv_heads * self.head_size
            logical_shard_offset = self.num_heads * self.head_size
        else:
            logical_shard_size = self.num_kv_heads * self.head_size
            logical_shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )

        awq_kind = _awq_kind(param)
        if awq_kind:
            pack_factor = self.quant_method.quant_config.pack_factor
            divisor = _awq_output_divisor(param, pack_factor)
            shard_dim = 1
        else:
            divisor = 1
            shard_dim = self.tp_dim
        shard_size = divide(logical_shard_size, divisor)
        shard_offset = divide(logical_shard_offset, divisor)
        param_data = param.data.narrow(shard_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, shard_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        if quant_config is not None and quant_config.group_size == -1 and tp_size > 1:
            raise ValueError("AWQ group_size=-1 is not supported by row tensor parallelism")
        super().__init__(
            divide(input_size, tp_size),
            output_size,
            bias,
            1,
            quant_config=quant_config,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_dim = 0 if _awq_kind(param) else self.tp_dim
        shard_size = param_data.size(shard_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.tp_rank == 0 else None
        y = self._apply_linear(x, bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
