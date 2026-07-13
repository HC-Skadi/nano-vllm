"""End-to-end nano-vLLM benchmark for Triton and FlashAttention backends."""

import argparse
import time

import torch


def select_backend(name: str) -> None:
    if name == "triton":
        return
    import nanovllm.layers.attention as attention
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    attention.flash_attn_varlen_func = flash_attn_varlen_func

    def decode(q, k, v, cache_seqlens, block_table, softmax_scale, causal=True):
        return flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    attention.flash_attn_with_kvcache = decode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("triton", "flash"), required=True)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    args = parser.parse_args()

    select_backend(args.backend)
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Request {i}: explain one benefit of GPU computing."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(args.requests)
    ]
    params = SamplingParams(temperature=0.6, max_tokens=args.output_tokens, ignore_eos=True)
    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    print(f"backend={args.backend}")
    print(f"requests={args.requests}")
    print(f"output_tokens={output_tokens}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"throughput_tokens_per_second={output_tokens / elapsed:.3f}")


if __name__ == "__main__":
    main()
