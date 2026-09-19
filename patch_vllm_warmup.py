"""Stub vLLM 0.25.1's MiniMax-M3 warmup import, whose triton kernel trips a
triton>=3.6 JITFunction parse bug ("NoneType has no attribute start") on
startup. The MiniMax warmup is irrelevant to DeepSeek-V2."""

from pathlib import Path

TARGET = Path(
    "/opt/vv/lib/python3.11/site-packages/vllm/model_executor/warmup/kernel_warmup.py"
)

OLD = """    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )"""
NEW = """    try:  # ZZPATCH: minimax warmup breaks triton>=3.6 JIT parse at startup
        from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
            minimax_m3_msa_warmup,
        )
    except Exception:
        def minimax_m3_msa_warmup(*args, **kwargs):
            return None"""


def main():
    src = TARGET.read_text()
    if "ZZPATCH" in src:
        print("already patched")
        return
    if OLD not in src:
        raise SystemExit("import pattern not found; patch manually")
    TARGET.write_text(src.replace(OLD, NEW))
    print("patched OK")


if __name__ == "__main__":
    main()
