"""TTFT/TPOT benchmark for DeepSeek-V2 on vLLM, matched to the nano-vllm
benchmark methodology (same prompt generator, same scenarios, per-request
TTFT/TPOT definitions).

Runs the vLLM V1 async engine directly (no HTTP server) so the comparison
against nano-vllm measures engine throughput, not serving-stack overhead.

Example:
    python bench_vllm_ttft_tpot.py --model /workspace/DeepSeek-V2-Lite-Chat \
        --scenarios 1:512:128,4:512:128,8:512:128,1:2048:128 --repeats 3 \
        --json /workspace/vllm_ttft_tpot.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def make_prompts(requests: int, input_tokens: int, vocab_size: int, seed: int):
    generator = random.Random(seed)
    return [
        [generator.randrange(10, vocab_size - 10) for _ in range(input_tokens)]
        for _ in range(requests)
    ]


async def run_scenario(engine, prompts, sampling_params, tag):
    submit = time.perf_counter()
    first_at: dict[str, float] = {}
    finish_at: dict[str, float] = {}
    counts: dict[str, int] = {}

    async def one(prompt, rid):
        from vllm.inputs import TokensPrompt
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=prompt), sampling_params, rid
        ):
            token_ids = output.outputs[0].token_ids
            if rid not in first_at and len(token_ids) >= 1:
                first_at[rid] = time.perf_counter()
            counts[rid] = len(token_ids)
            if output.finished:
                finish_at[rid] = time.perf_counter()

    await asyncio.gather(*(one(p, f"{tag}-{i}") for i, p in enumerate(prompts)))
    elapsed = time.perf_counter() - submit

    ttfts = [(first_at[r] - submit) * 1000 for r in sorted(first_at)]
    tpots = []
    for r in sorted(finish_at):
        if counts[r] >= 2:
            tpots.append((finish_at[r] - first_at[r]) * 1000 / (counts[r] - 1))
    total_out = sum(counts.values())
    return {
        "ttft_ms_mean": statistics.fmean(ttfts),
        "ttft_ms_median": statistics.median(ttfts),
        "ttft_ms_max": max(ttfts),
        "tpot_ms_mean": statistics.fmean(tpots),
        "tpot_ms_median": statistics.median(tpots),
        "output_tokens_per_second": total_out / elapsed,
        "requests_per_second": len(prompts) / elapsed,
        "elapsed_seconds": elapsed,
    }


async def amain(args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    try:
        from vllm import AsyncLLM
    except ImportError:
        from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max(
            inp + out for _, inp, out in parse_scenarios(args.scenarios)
        ),
        max_num_seqs=max(r for r, _, _ in parse_scenarios(args.scenarios)),
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
    except (TypeError, AttributeError):
        engine = AsyncLLM(engine_args=engine_args)
    config = json.loads((Path(args.model) / "config.json").read_text())
    vocab_size = int(config["vocab_size"])

    results = []
    for requests, input_tokens, output_tokens in parse_scenarios(args.scenarios):
        sp = SamplingParams(
            temperature=args.temperature,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        warmup = make_prompts(requests, input_tokens, vocab_size, args.seed + 100_000)
        await run_scenario(engine, warmup, sp, "warm")

        runs = []
        for repeat in range(args.repeats):
            prompts = make_prompts(
                requests, input_tokens, vocab_size, args.seed + repeat
            )
            runs.append(await run_scenario(engine, prompts, sp, f"r{repeat}"))
        median = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
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
            f"[vllm] req={requests} in={input_tokens} out={output_tokens}: "
            f"TTFT {median['ttft_ms_median']:.1f} ms, "
            f"TPOT {median['tpot_ms_median']:.1f} ms, "
            f"{median['output_tokens_per_second']:.0f} out tok/s",
            flush=True,
        )

    try:
        engine.shutdown()
    except Exception:
        pass

    print(f"\n{'scenario':<16} {'TTFT med ms':>12} {'TTFT mean':>10} "
          f"{'TPOT med ms':>12} {'TPOT mean':>10} {'out tok/s':>10}")
    for res in results:
        m = res["median"]
        label = f"{res['requests']}x{res['input_tokens']}+{res['output_tokens']}"
        print(
            f"{label:<16} {m['ttft_ms_median']:>12.1f} {m['ttft_ms_mean']:>10.1f} "
            f"{m['tpot_ms_median']:>12.2f} {m['tpot_ms_mean']:>10.2f} "
            f"{m['output_tokens_per_second']:>10.0f}"
        )

    if args.json:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": "vllm",
            "vllm_version": __import__("vllm").__version__,
            "model": str(Path(args.model).resolve()),
            "enforce_eager": args.enforce_eager,
            "repeats": args.repeats,
            "results": results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))


def parse_scenarios(raw: str):
    scenarios = []
    for chunk in raw.split(","):
        r, i, o = (int(x) for x in chunk.split(":"))
        scenarios.append((r, i, o))
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--scenarios", default="1:512:128,4:512:128,8:512:128,1:2048:128"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
