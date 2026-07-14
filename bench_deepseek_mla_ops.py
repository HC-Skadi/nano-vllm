"""Benchmark expanded and weight-absorbed DeepSeek-V2-Lite MLA decode.

This benchmark uses the checkpoint's real attention dimensions but synthetic
BF16 weights/activations, so it fits on GPUs that cannot hold the 16B model.
Both caches are derived from the same latent history and both modules share
identical weights; output error and decode latency are therefore comparable.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import torch

from nanovllm.models.deepseek_v2 import DeepseekV2Attention
from nanovllm.utils.context import reset_context, set_context


BLOCK_SIZE = 256
DTYPE = torch.bfloat16


def lite_config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=163840,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        },
        deepseek_mla_backend=backend,
    )


@contextmanager
def single_rank_dist():
    with (
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        yield


@dataclass
class CaseResult:
    batch: int
    context: int
    expanded_ms: float
    latent_ms: float
    speedup: float
    expanded_tokens_per_second: float
    latent_tokens_per_second: float
    expanded_cache_mib: float
    latent_cache_mib: float
    cache_reduction_percent: float
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float


def make_modules() -> tuple[DeepseekV2Attention, DeepseekV2Attention]:
    with single_rank_dist():
        expanded = DeepseekV2Attention(lite_config("expanded"))
        latent = DeepseekV2Attention(lite_config("latent"))
    with torch.no_grad():
        for parameter in expanded.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    latent.load_state_dict(expanded.state_dict())
    return expanded.cuda().to(DTYPE), latent.cuda().to(DTYPE)


def block_table(batch: int, context: int) -> torch.Tensor:
    blocks_per_sequence = math.ceil(context / BLOCK_SIZE)
    return torch.arange(
        batch * blocks_per_sequence, device="cuda", dtype=torch.int32
    ).view(batch, blocks_per_sequence)


def fill_caches(
    expanded: DeepseekV2Attention,
    latent: DeepseekV2Attention,
    batch: int,
    context: int,
) -> tuple[torch.Tensor, int, int]:
    table = block_table(batch, context)
    num_blocks = table.numel()
    latent_dim = latent.kv_lora_rank
    rope_dim = latent.qk_rope_head_dim
    qk_dim = expanded.qk_head_dim

    history_latent = torch.randn(
        batch, context, latent_dim, device="cuda", dtype=DTYPE
    )
    history_rope = torch.randn(
        batch, context, rope_dim, device="cuda", dtype=DTYPE
    )
    latent.attn.k_cache = torch.zeros(
        num_blocks,
        BLOCK_SIZE,
        1,
        latent_dim + rope_dim,
        device="cuda",
        dtype=DTYPE,
    )
    latent.attn.v_cache = torch.empty(0, device="cuda", dtype=DTYPE)
    expanded.attn.k_cache = torch.zeros(
        num_blocks,
        BLOCK_SIZE,
        expanded.num_heads,
        qk_dim,
        device="cuda",
        dtype=DTYPE,
    )
    expanded.attn.v_cache = torch.zeros_like(expanded.attn.k_cache)

    up = expanded.kv_b_proj.weight.view(
        expanded.num_heads,
        expanded.qk_nope_head_dim + expanded.v_head_dim,
        latent_dim,
    )
    key_weight, value_weight = up.split(
        [expanded.qk_nope_head_dim, expanded.v_head_dim], dim=1
    )
    with torch.no_grad():
        key_nope = torch.einsum("bsl,hnl->bshn", history_latent, key_weight)
        values = torch.einsum("bsl,hvl->bshv", history_latent, value_weight)
        keys = torch.cat(
            (
                key_nope,
                history_rope.unsqueeze(2).expand(-1, -1, expanded.num_heads, -1),
            ),
            dim=-1,
        )
        if values.shape[-1] != qk_dim:
            values = torch.nn.functional.pad(
                values, (0, qk_dim - values.shape[-1])
            )

        for batch_idx in range(batch):
            for logical_block, physical_block in enumerate(table[batch_idx].tolist()):
                start = logical_block * BLOCK_SIZE
                end = min(start + BLOCK_SIZE, context)
                length = end - start
                expanded.attn.k_cache[physical_block, :length] = keys[
                    batch_idx, start:end
                ]
                expanded.attn.v_cache[physical_block, :length] = values[
                    batch_idx, start:end
                ]
                latent.attn.k_cache[physical_block, :length, 0] = torch.cat(
                    (
                        history_latent[batch_idx, start:end],
                        history_rope[batch_idx, start:end],
                    ),
                    dim=-1,
                )

    expanded_bytes = (
        expanded.attn.k_cache.numel() + expanded.attn.v_cache.numel()
    ) * expanded.attn.k_cache.element_size()
    latent_bytes = latent.attn.k_cache.numel() * latent.attn.k_cache.element_size()
    del keys, values, key_nope, history_latent, history_rope
    torch.cuda.empty_cache()
    return table, expanded_bytes, latent_bytes


def compare_latency(
    functions: dict[str, Callable[[], torch.Tensor]],
    warmup: int,
    repeats: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Interleave backend order to reduce clock and thermal bias."""
    names = list(functions)
    outputs = {}
    for iteration in range(warmup):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            outputs[name] = functions[name]()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    for iteration in range(repeats):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[name] = functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return outputs, {
        name: statistics.median(values) for name, values in samples.items()
    }


def benchmark_case(
    expanded: DeepseekV2Attention,
    latent: DeepseekV2Attention,
    batch: int,
    context: int,
    warmup: int,
    repeats: int,
) -> CaseResult:
    table, expanded_bytes, latent_bytes = fill_caches(
        expanded, latent, batch, context
    )
    hidden = torch.randn(batch, 2048, device="cuda", dtype=DTYPE)
    positions = torch.full(
        (batch,), context - 1, device="cuda", dtype=torch.int64
    )
    slots = table[:, -1] * BLOCK_SIZE + (context - 1) % BLOCK_SIZE
    context_lens = torch.full(
        (batch,), context, device="cuda", dtype=torch.int32
    )
    set_context(
        False,
        slot_mapping=slots,
        context_lens=context_lens,
        block_tables=table,
        host_context_lens=(context,) * batch,
    )

    with torch.inference_mode():
        outputs, timings = compare_latency(
            {
                "expanded": lambda: expanded(positions, hidden),
                "latent": lambda: latent(positions, hidden),
            },
            warmup,
            repeats,
        )
    expanded_output = outputs["expanded"]
    latent_output = outputs["latent"]
    expanded_ms = timings["expanded"]
    latent_ms = timings["latent"]
    error = (latent_output.float() - expanded_output.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        latent_output.float().flatten(),
        expanded_output.float().flatten(),
        dim=0,
    ).item()
    reset_context()
    return CaseResult(
        batch=batch,
        context=context,
        expanded_ms=expanded_ms,
        latent_ms=latent_ms,
        speedup=expanded_ms / latent_ms,
        expanded_tokens_per_second=batch / (expanded_ms / 1000.0),
        latent_tokens_per_second=batch / (latent_ms / 1000.0),
        expanded_cache_mib=expanded_bytes / 2**20,
        latent_cache_mib=latent_bytes / 2**20,
        cache_reduction_percent=(1.0 - latent_bytes / expanded_bytes) * 100.0,
        max_abs_error=error.max().item(),
        mean_abs_error=error.mean().item(),
        cosine_similarity=cosine,
    )


def parse_cases(values: list[str]) -> list[tuple[int, int]]:
    cases = []
    for value in values:
        try:
            batch_text, context_text = value.split(":", 1)
            case = (int(batch_text), int(context_text))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid case {value!r}; expected BATCH:CONTEXT"
            ) from exc
        if min(case) <= 0:
            raise argparse.ArgumentTypeError("case values must be positive")
        cases.append(case)
    return cases


def print_results(results: list[CaseResult]) -> None:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("DeepSeek-V2-Lite dimensions, BF16, one MLA layer, decode")
    print(
        f"{'B':>3} {'context':>7} {'expanded ms':>12} {'latent ms':>10} "
        f"{'speedup':>9} {'expanded MiB':>13} {'latent MiB':>11} "
        f"{'cache saved':>12} {'cosine':>10}"
    )
    for result in results:
        print(
            f"{result.batch:>3} {result.context:>7} "
            f"{result.expanded_ms:>12.3f} {result.latent_ms:>10.3f} "
            f"{result.speedup:>8.3f}x {result.expanded_cache_mib:>13.2f} "
            f"{result.latent_cache_mib:>11.2f} "
            f"{result.cache_reduction_percent:>11.2f}% "
            f"{result.cosine_similarity:>10.7f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", nargs="+", default=["1:128", "1:1024", "8:1024"]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    cases = parse_cases(args.cases)
    torch.manual_seed(0)
    expanded, latent = make_modules()
    results = [
        benchmark_case(expanded, latent, batch, context, args.warmup, args.repeats)
        for batch, context in cases
    ]
    print_results(results)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "dtype": str(DTYPE),
                    "scope": "one DeepSeek-V2-Lite MLA layer decode",
                    "results": [result.__dict__ for result in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
