import os
from dataclasses import dataclass
from transformers import AutoConfig


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
    # Hugging Face 模型配置，初始化时根据 model 自动加载。
    hf_config: AutoConfig | None = None
    # 结束符 token id，调度后处理阶段用它判断序列是否完成。
    eos: int = -1
    # KV cache 的块大小，注意力缓存按块申请和复用。
    kvcache_block_size: int = 256
    # 可用 KV cache 块数，模型预热并统计显存后自动计算。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # 只支持加载本地模型目录，避免运行时隐式下载权重。
        assert os.path.isdir(self.model)
        # 块大小需要和底层 kernel/cache 布局对齐。
        assert self.kvcache_block_size % 256 == 0
        # 当前实现限制张量并行规模，通常对应单机 GPU 数量。
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 不能超过模型自身支持的最大位置编码长度。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
