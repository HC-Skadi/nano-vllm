# Qwen AWQ support

nano-vLLM supports text-only `Qwen2ForCausalLM` and `Qwen3ForCausalLM`
checkpoints that use the standard AutoAWQ GEMM layout:

- 4-bit weights and FP16 activations (W4A16)
- group size 128, or another group size aligned with every TP input shard
- zero point enabled
- `version: "gemm"`
- NVIDIA compute capability 7.5 or newer for vLLM's fused CUDA kernel

The `awq` extra pins the tested vLLM 0.8.5 / Transformers 4.51.3 pair so pip
does not combine the binary extension with an unverified Torch/Transformers ABI.

GEMV, AWQ-Marlin, Qwen-VL, and Qwen MoE checkpoints are not included in this
implementation. Embeddings, normalization, and the LM head remain FP16.

## Execution modes

Pass `awq_backend` to `LLM`:

| Value | Execution |
|---|---|
| `auto` | vLLM `awq_gemm` for M < 256; dequantize + matmul for M >= 256 |
| `gemm` | Always force fused vLLM `awq_gemm` |
| `dequant` | Always dequantize the packed weight, then use `torch.matmul` |

The `fused` and `dequantize` spellings are accepted as aliases. If the vLLM
extension is unavailable, `auto`/`dequant` can use a slow PyTorch reference
dequantizer for correctness; an explicit `gemm` request fails with an actionable
error instead of silently changing the benchmark path. The reference fallback
must be run with `enforce_eager=True`; CUDA Graph mode requires the vLLM
operators.

## Model smoke test

The implementation was checked with the official
`Qwen/Qwen2.5-0.5B-Instruct-AWQ` checkpoint. All 24 layers loaded successfully,
including packed Q/K/V and gate/up concatenation. Both eager and CUDA Graph
generation completed on an RTX 3060 Laptop GPU.

For a model-level correctness check, the checkpoint was independently
dequantized into a Transformers 4.51.3 `Qwen2ForCausalLM` reference model. On a
fixed four-token input, nano-vLLM matched the reference argmax and all top-5
token IDs. The full-vocabulary logits had max absolute error 0.02344 and mean
absolute error 0.00427.

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-AWQ \
  --local-dir ~/huggingface/Qwen2.5-0.5B-Instruct-AWQ
```

```python
from os.path import expanduser

from nanovllm import LLM, SamplingParams

llm = LLM(
    expanduser("~/huggingface/Qwen2.5-0.5B-Instruct-AWQ"),
    awq_backend="auto",
    max_model_len=4096,
)
print(llm.generate(["Hello"], SamplingParams(max_tokens=32))[0]["text"])
```

## Fused versus unfused operator benchmark

`bench_awq.py` times the same FP16 activation and packed
`qweight/qzeros/scales` through both paths. Dequantization is inside every
unfused timed invocation. Timing uses per-call CUDA events, warm-up iterations,
and median/P90 statistics.

```bash
python bench_awq.py --k 896 --n 1152 --group-size 128 --dense
python bench_awq.py --k 896 --n 896 --group-size 128
python bench_awq.py --k 896 --n 9728 --group-size 128
python bench_awq.py --k 4864 --n 896 --group-size 128
python bench_awq.py --json awq-results.json
```

The following median-latency speedups were measured with vLLM 0.8.5,
PyTorch 2.6.0+cu124, FP16, and an RTX 3060 Laptop GPU (SM86). A value above 1
means fused `awq_gemm` was faster. Shapes match Qwen2.5-0.5B projections; input
and packed weights were synthetic so the operator paths receive identical data.

| Projection (K x N) | M=1 | M=8 | M=32 | M=128 | M=256 | M=512 | M=2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| QKV (896 x 1152) | 1.17x | 0.86x | 1.14x | 1.11x | 0.72x | 0.55x | 0.38x |
| O (896 x 896) | 1.25x | 1.30x | 1.05x | 1.01x | 1.07x | 0.55x | 0.38x |
| gate/up (896 x 9728) | 7.44x | 6.88x | 2.76x | 0.86x | 0.62x | 0.46x | 0.33x |
| down (4864 x 896) | 1.60x | 2.08x | 1.83x | 1.79x | 1.09x | 0.96x | 0.81x |

For example, QKV at M=1 took 0.0727 ms fused versus 0.0849 ms unfused;
at M=2048 it took 1.3010 ms fused versus 0.5007 ms unfused. The crossover is
shape-dependent, but the measurements confirm why a decode/prefill dispatch is
necessary rather than forcing one path globally. Re-run the benchmark on the
deployment GPU before tuning the threshold.

The maximum absolute difference between paths was 0.0625 across these runs;
mean absolute errors were between roughly 0.0009 and 0.0031, consistent with
different FP16 accumulation/reduction orders.

The complete JSON output, including median/P90 latency, numerical error,
environment, and methodology, is committed under
[`benchmarks/awq_rtx3060`](../benchmarks/awq_rtx3060/).
