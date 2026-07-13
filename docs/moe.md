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
- DeepSeek Multi-head Latent Attention (MLA). The latent projections are
  expanded into ordinary per-head keys and values before attention, so the
  existing paged-attention implementation can be reused.
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
- A native latent MLA KV cache. Nano-vLLM currently stores expanded K/V tensors
  in its regular paged KV cache.

Expanded K/V is simpler to inspect and validate, but uses more KV-cache memory
than a native latent-cache implementation. MoE weights also need to be resident
even though only a subset of routed experts is active for each token. Plan
memory from the model's total parameter count, not only its active parameter
count.

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
