<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.
* 🤖 **MoE model support** - Correctness-first BF16 inference for DeepSeek-V2-Lite and DeepSeek-V2-Lite-Chat
* 📦 **Qwen AWQ** - W4A16 inference for standard GEMM-format Qwen2/Qwen2.5/Qwen3 checkpoints

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

Install the optional vLLM kernels when running AWQ models:

```bash
pip install -e '.[awq]'
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## MoE / DeepSeek

The first MoE model path targets the BF16 DeepSeek-V2-Lite and
DeepSeek-V2-Lite-Chat checkpoints. It supports top-k and shared experts, MLA
with expanded K/V caching, YaRN, tensor parallelism, and eager execution. Use
`enforce_eager=True` and start with a smaller `max_num_batched_tokens` (for
example, `2048`) to limit warm-up and prefill memory.

See [MoE and DeepSeek support](docs/moe.md) for the support matrix, current
limitations, model download command, and an `LLM` example.

## Qwen AWQ

Standard AutoAWQ W4/GEMM checkpoints are detected from their Hugging Face
`quantization_config`. The default `auto` backend follows vLLM's dispatch:
fused `awq_gemm` below 256 tokens and `awq_dequantize + torch.matmul` for larger
token batches.

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/YOUR/Qwen2.5-0.5B-Instruct-AWQ",
    awq_backend="auto",       # auto | gemm | dequant
    enforce_eager=True,
)
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))
```

See [docs/awq.md](docs/awq.md) for the supported format, correctness checks,
limitations, and fused-vs-unfused benchmark results.

## Benchmark

See `bench.py` for benchmark.

For an operator-level AWQ comparison using identical packed weights, run:

```bash
python bench_awq.py --dense
```

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
