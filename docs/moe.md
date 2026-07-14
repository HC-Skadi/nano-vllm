# MoE and DeepSeek support

Nano-vLLM's first MoE milestone targets **correctness-first offline inference**
with the BF16 checkpoints of:

- `deepseek-ai/DeepSeek-V2-Lite`
- `deepseek-ai/DeepSeek-V2-Lite-Chat`

The implementation deliberately reuses the readable eager execution and paged
KV-cache paths already present in Nano-vLLM. It is intended as a reference
implementation first; it is not a performance-optimized MoE backend.

## Supported in the first version

- Model-configured top-k routed experts and shared experts.
- DeepSeek Multi-head Latent Attention (MLA) with weight absorption. The paged
  cache stores the normalized KV latent and decoupled RoPE key once per token;
  `W_UK` is applied on the query side and `W_UV` after latent attention, so
  cached tokens are not expanded into per-head keys and values.
- DeepSeek's YaRN rotary-position scaling configuration.
- Tensor parallelism through `tensor_parallel_size`, subject to the usual
  divisibility requirements of the model dimensions. This is tensor
  parallelism, not expert parallelism.
- Eager prefill and decode. Pass `enforce_eager=True` when constructing `LLM`.

## Not supported yet

- DeepSeek-V3 FP8 checkpoints or an FP8 inference path.
- Expert parallelism.
- A fused MoE dispatch/expert kernel.
- CUDA Graph execution for the DeepSeek/MoE model path.
- A fused Triton/CUDA kernel for latent MLA attention. The current
  correctness-first implementation uses PyTorch tensor operations over the
  native paged latent cache.

MoE weights need to be resident even though only a subset of routed experts is
active for each token. Plan memory from the model's total parameter count, not
only its active parameter count.

## Compare expanded and weight-absorbed MLA

The pre-optimization expanded-KV path remains available as a benchmark
baseline. Run both backends in isolated processes with a fixed cache capacity:

```bash
python bench_deepseek_mla.py \
  --model ~/huggingface/DeepSeek-V2-Lite-Chat \
  --requests 8 \
  --input-tokens 512 \
  --output-tokens 128 \
  --repeats 3 \
  --json benchmarks/deepseek_mla.json
```

The report includes TTFT, TPOT, prefill/decode throughput, cache allocation,
and sampled-token agreement. The two backends use the same checkpoint, prompt
token IDs, fixed cache-block count, eager mode, and BF16 runtime. Use
`deepseek_mla_backend="expanded"` or `"latent"` on `LLM` to select one path
directly.

For a GPU that cannot hold the complete 16B BF16 checkpoint, run the real Lite
attention dimensions as a one-layer decode benchmark:

```bash
python bench_deepseek_mla_ops.py \
  --cases 1:128 1:1024 8:1024 \
  --warmup 10 \
  --repeats 50 \
  --json benchmarks/deepseek_mla_rtx3060.json
```

Measured on an NVIDIA GeForce RTX 3060 Laptop GPU in BF16:

| Batch | Context | Expanded | Latent | Relative throughput | Expanded cache | Latent cache |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.384 ms | 2.260 ms | 0.612x | 3.00 MiB | 0.28 MiB |
| 1 | 1024 | 1.396 ms | 2.271 ms | 0.615x | 12.00 MiB | 1.12 MiB |
| 8 | 1024 | 2.160 ms | 2.739 ms | 0.788x | 96.00 MiB | 9.00 MiB |

The latent cache is 90.62% smaller and output cosine similarity is above
0.99998 in these cases. The current PyTorch latent implementation is not yet a
latency optimization: it remains slower than the fused Triton expanded-KV
baseline. A fused latent MLA decode kernel is required before claiming a decode
throughput improvement. Raw results are stored in
`benchmarks/deepseek_mla_rtx3060.json`.

### One-command A100 run

For a single A100 40GB or 80GB with a local checkpoint, run:

```bash
./run_deepseek_mla_a100.sh /models/DeepSeek-V2-Lite-Chat
```

The script checks GPU memory, PyTorch CUDA/BF16 support, and checkpoint files.
It then runs a real-dimension operator benchmark, a full-model smoke test, and
an end-to-end matrix. A100 80GB uses batch 8 and a 4096-token long-context
case; A100 40GB uses batch 4 and 2048 tokens. Results, logs, and environment
metadata are written to a timestamped `benchmarks/a100_*` directory.

Useful overrides:

```bash
PROFILE=smoke REPEATS=1 \
  ./run_deepseek_mla_a100.sh /models/DeepSeek-V2-Lite-Chat

PROFILE=full REPEATS=5 OUTPUT_DIR=/results/mla-a100 \
  ./run_deepseek_mla_a100.sh /models/DeepSeek-V2-Lite-Chat
```

## Download a checkpoint

Install the Hugging Face Hub CLI if it is not already available, then download
one of the BF16 checkpoints to a local directory:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download deepseek-ai/DeepSeek-V2-Lite-Chat \
  --local-dir ~/huggingface/DeepSeek-V2-Lite-Chat
```

For the base checkpoint, replace both occurrences of
`DeepSeek-V2-Lite-Chat` with `DeepSeek-V2-Lite`.

## Run chat completion

The following example uses two-way tensor parallelism as an illustration. Set
`tensor_parallel_size` to the number of local GPUs assigned to the process and
ensure that the model dimensions are divisible by that value.

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "/home/user/huggingface/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,  # required by the checkpoint's custom tokenizer
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain mixture-of-experts routing briefly."}],
    tokenize=False,
    add_generation_prompt=True,
)

llm = LLM(
    model_path,
    tensor_parallel_size=2,
    enforce_eager=True,
    trust_remote_code=True,  # also used by Nano-vLLM's internal tokenizer
    max_model_len=4096,
    max_num_batched_tokens=2048,
    max_num_seqs=16,
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
output = llm.generate([prompt], sampling_params)
print(output[0]["text"])
```

Only enable `trust_remote_code` for a checkpoint repository whose tokenizer code
you trust and have reviewed. Nano-vLLM's model configuration and inference
implementation do not otherwise depend on the checkpoint's remote model code.

`max_num_batched_tokens` controls both scheduler batching and the warm-up
prefill shape. The general default (`16384`) can cause a large temporary memory
peak for this model path. Start with `2048` or lower, then increase it only
after measuring available memory. Reducing `max_model_len`, `max_num_seqs`, or
`gpu_memory_utilization` may also help when memory is tight.

This support statement describes the implemented model path and its intended
checkpoint format; it is not a claim of end-to-end validation of the complete
16B-class checkpoint on a particular GPU setup.
