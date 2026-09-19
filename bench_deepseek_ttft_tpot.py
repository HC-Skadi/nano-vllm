"""End-to-end TTFT/TPOT benchmark for the adapted DeepSeek-V2 model.

The real DeepSeek-V2-Lite checkpoint (~16B params, BF16) does not fit on a
single consumer GPU, so this benchmark first materializes a synthetic
DeepSeek-V2 checkpoint whose MLA dimensions match DeepSeek-V2-Lite exactly
(hidden 2048, q_lora 1536, kv_lora 512, 16 heads, 128+64 rope/nope, v 128)
but with fewer layers and experts, and random weights.  The full engine
(tokenizer -> scheduler -> model runner -> DeepseekV2ForCausalLM) then runs
prefill/decode on that checkpoint, so TTFT/TPOT measure the adapted code
path end to end.

TTFT: submit -> first output token, per request (batch requests scheduled
across multiple prefill steps get honest per-request values).
TPOT: (finish - first token) / (output_tokens - 1), per request.

Prompts are re-randomized every repeat so the scheduler's prefix cache
cannot serve prefills from the previous run.

Example:
    python bench_deepseek_ttft_tpot.py --backend latent \
        --scenarios 1:512:128,4:512:128,1:2048:128 --repeats 3 \
        --json benchmarks/deepseek_ttft_tpot_latent.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

DEFAULT_MODEL_DIR = Path.home() / "models" / "synthetic-deepseek-v2-ttft"
TOKENIZER_SOURCE = Path.home() / "models" / "Qwen3-0.6B"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")

# MLA dimensions copied verbatim from DeepSeek-V2-Lite-Chat; depth and expert
# count are reduced so BF16 weights stay under ~2 GiB for consumer GPUs.
SYNTHETIC_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "torch_dtype": "bfloat16",
    "vocab_size": 32768,
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "num_hidden_layers": 6,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "hidden_act": "silu",
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 1024,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "greedy",
    "scoring_func": "softmax",
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "rope_theta": 10000.0,
}


def build_synthetic_checkpoint(model_dir: Path, tokenizer_src: Path, seed: int) -> None:
    from safetensors.torch import save_file
    from transformers import AutoConfig
    from unittest.mock import patch

    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(SYNTHETIC_CONFIG, indent=2), encoding="utf-8")

    for name in TOKENIZER_FILES:
        source = tokenizer_src / name
        if not source.is_file():
            raise FileNotFoundError(f"tokenizer file not found: {source}")
        shutil.copy2(source, model_dir / name)

    import nanovllm.models.deepseek_v2 as deepseek_v2

    generator = torch.Generator().manual_seed(seed)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        config = AutoConfig.from_pretrained(model_dir)
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            model = deepseek_v2.DeepseekV2ForCausalLM(config)
        tensors = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.normal_(0.0, 0.02, generator=generator)
                weight = param.detach().cpu().contiguous()
                if name.endswith("gate_up_proj.weight"):
                    # Store the packed module the way an HF DeepSeek-V2
                    # checkpoint does so the loader's shard mapping applies.
                    prefix = name[: -len("gate_up_proj.weight")]
                    gate, up = weight.chunk(2, dim=0)
                    tensors[f"{prefix}gate_proj.weight"] = gate.contiguous()
                    tensors[f"{prefix}up_proj.weight"] = up.contiguous()
                else:
                    tensors[name] = weight
    finally:
        torch.set_default_dtype(default_dtype)

    save_file(tensors, str(model_dir / "model.safetensors"))
    total_gib = sum(t.numel() * t.element_size() for t in tensors.values()) / 2**30
    print(
        f"built synthetic checkpoint at {model_dir} "
        f"({len(tensors)} tensors, {total_gib:.2f} GiB)"
    )


def ensure_checkpoint(model_dir: Path, tokenizer_src: Path, seed: int) -> None:
    # A directory with config + weights is a real checkpoint (e.g. a rented-GPU
    # run against DeepSeek-V2-Lite-Chat); never rebuild or touch it.  Only a
    # missing/incomplete directory gets the synthetic checkpoint written.
    has_weights = any(model_dir.glob("*.safetensors"))
    if (model_dir / "config.json").is_file() and has_weights:
        print(f"using existing checkpoint at {model_dir}")
        return
    build_synthetic_checkpoint(model_dir, tokenizer_src, seed)


def make_prompts(requests: int, input_tokens: int, vocab_size: int, seed: int):
    generator = random.Random(seed)
    return [
        [generator.randrange(10, vocab_size - 10) for _ in range(input_tokens)]
        for _ in range(requests)
    ]


def run_batch(llm, prompts, sampling_params):
    """Mirror LLMEngine.step() while timestamping each request's tokens."""
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first_token_at: dict[int, float] = {}
    finish_at: dict[int, float] = {}
    output_lengths: dict[int, int] = {}
    prefill_seconds = 0.0
    prefill_tokens = 0
    decode_seconds = 0.0
    decode_tokens = 0
    decode_steps = 0

    while not llm.is_finished():
        seqs, is_prefill = llm.scheduler.schedule()
        # postprocess() resets num_scheduled_tokens, so snapshot it first.
        scheduled_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        )
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        torch.cuda.synchronize()
        step_ended = time.perf_counter()
        if scheduled_tokens > 0:
            prefill_seconds += step_ended - step_started
            prefill_tokens += scheduled_tokens
        else:
            decode_seconds += step_ended - step_started
            decode_tokens += -scheduled_tokens
            decode_steps += 1
        for seq in seqs:
            if seq.seq_id not in first_token_at and seq.num_completion_tokens >= 1:
                first_token_at[seq.seq_id] = step_ended
            if seq.is_finished and seq.seq_id not in finish_at:
                finish_at[seq.seq_id] = step_ended
                output_lengths[seq.seq_id] = seq.num_completion_tokens

    elapsed = time.perf_counter() - started
    ttfts_ms = [
        (first_token_at[seq_id] - started) * 1000.0
        for seq_id in sorted(first_token_at)
    ]
    tpots_ms = []
    for seq_id in sorted(finish_at):
        produced = output_lengths[seq_id]
        if produced >= 2:
            tpots_ms.append(
                (finish_at[seq_id] - first_token_at[seq_id]) * 1000.0 / (produced - 1)
            )
    total_output_tokens = sum(output_lengths.values())
    return {
        "ttft_ms_mean": statistics.fmean(ttfts_ms),
        "ttft_ms_median": statistics.median(ttfts_ms),
        "ttft_ms_max": max(ttfts_ms),
        "tpot_ms_mean": statistics.fmean(tpots_ms),
        "tpot_ms_median": statistics.median(tpots_ms),
        "prefill_tokens_per_second": prefill_tokens / prefill_seconds,
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "output_tokens_per_second": total_output_tokens / elapsed,
        "requests_per_second": len(prompts) / elapsed,
        "elapsed_seconds": elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def parse_scenarios(raw: str):
    scenarios = []
    for chunk in raw.split(","):
        requests, input_tokens, output_tokens = (int(part) for part in chunk.split(":"))
        if min(requests, input_tokens, output_tokens) <= 0:
            raise ValueError(f"invalid scenario: {chunk}")
        if output_tokens < 2:
            raise ValueError(f"TPOT needs >= 2 output tokens: {chunk}")
        scenarios.append((requests, input_tokens, output_tokens))
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("latent", "expanded"), default="latent")
    parser.add_argument(
        "--scenarios",
        default="1:512:128,4:512:128,8:512:128,1:2048:128",
        help="comma list of requests:input_tokens:output_tokens",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer-src", type=Path, default=TOKENIZER_SOURCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; TTFT/TPOT need a GPU")

    scenarios = parse_scenarios(args.scenarios)
    max_requests = max(requests for requests, _, _ in scenarios)
    max_context = max(inp + out for _, inp, out in scenarios)

    ensure_checkpoint(args.model_dir, args.tokenizer_src, args.seed)

    from nanovllm import LLM, SamplingParams

    initialization_started = time.perf_counter()
    llm = LLM(
        str(args.model_dir),
        enforce_eager=True,  # DeepseekV2ForCausalLM disables CUDA graphs anyway
        tensor_parallel_size=1,
        max_model_len=max_context,
        max_num_seqs=max_requests,
        max_num_batched_tokens=max(max_requests * max(inp for _, inp, _ in scenarios), max_context),
        gpu_memory_utilization=args.gpu_memory_utilization,
        deepseek_mla_backend=args.backend,
        trust_remote_code=True,
    )
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialization_started

    vocab_size = int(llm.model_runner.config.hf_config.vocab_size)
    results = []
    for requests, input_tokens, output_tokens in scenarios:
        params = SamplingParams(
            temperature=args.temperature,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        # Warmup with distinct ids so repeats below cannot hit prefix cache.
        warmup = make_prompts(requests, input_tokens, vocab_size, args.seed + 100_000)
        run_batch(llm, warmup, params)

        runs = []
        for repeat in range(args.repeats):
            prompts = make_prompts(
                requests, input_tokens, vocab_size, args.seed + repeat
            )
            runs.append(run_batch(llm, prompts, params))
        median = {
            field: statistics.median(run[field] for run in runs) for field in runs[0]
        }
        results.append(
            {
                "requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "median": median,
                "runs": runs,
            }
        )
        print(
            f"[{args.backend}] req={requests} in={input_tokens} out={output_tokens}: "
            f"TTFT {median['ttft_ms_median']:.1f} ms (mean {median['ttft_ms_mean']:.1f}), "
            f"TPOT {median['tpot_ms_median']:.1f} ms (mean {median['tpot_ms_mean']:.1f}), "
            f"prefill {median['prefill_tokens_per_second']:.0f} tok/s, "
            f"decode {median['decode_tokens_per_second']:.0f} tok/s"
        )

    header = (
        f"{'scenario':<16} {'TTFT med ms':>12} {'TTFT mean':>10} "
        f"{'TPOT med ms':>12} {'TPOT mean':>10} {'prefill t/s':>11} {'decode t/s':>10}"
    )
    print(f"\nDeepSeek-V2 TTFT/TPOT  backend={args.backend}  "
          f"gpu={torch.cuda.get_device_name(0)}  dtype=bf16")
    print(header)
    for result in results:
        median = result["median"]
        label = f"{result['requests']}x{result['input_tokens']}+{result['output_tokens']}"
        print(
            f"{label:<16} {median['ttft_ms_median']:>12.1f} {median['ttft_ms_mean']:>10.1f} "
            f"{median['tpot_ms_median']:>12.2f} {median['tpot_ms_mean']:>10.2f} "
            f"{median['prefill_tokens_per_second']:>11.0f} "
            f"{median['decode_tokens_per_second']:>10.0f}"
        )

    if args.json:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "model_dir": str(args.model_dir.resolve()),
            "synthetic_config": SYNTHETIC_CONFIG,
            "initialization_seconds": initialization_seconds,
            "num_kvcache_blocks": llm.model_runner.config.num_kvcache_blocks,
            "repeats": args.repeats,
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
