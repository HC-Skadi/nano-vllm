#!/usr/bin/env bash
# Full nano-vllm vs vLLM TTFT/TPOT comparison on one GPU (real DeepSeek-V2-Lite).
set -x
cd /workspace/nano-vllm
M=/workspace/DeepSeek-V2-Lite-Chat
SCEN=1:512:128,4:512:128,8:512:128,1:2048:128

echo "=== STEP1 nano latent ==="
/opt/vn/bin/python bench_deepseek_ttft_tpot.py \
  --model-dir "$M" --backend latent --scenarios "$SCEN" \
  --repeats 3 --json /workspace/nano_latent.json || echo STEP1_FAILED

echo "=== STEP2 nano expanded ==="
/opt/vn/bin/python bench_deepseek_ttft_tpot.py \
  --model-dir "$M" --backend expanded --scenarios "$SCEN" \
  --repeats 3 --json /workspace/nano_expanded.json || echo STEP2_FAILED

echo "=== STEP3 vllm default ==="
/opt/vv/bin/python bench_vllm_ttft_tpot.py \
  --model "$M" --scenarios "$SCEN" \
  --repeats 3 --json /workspace/vllm_default.json || echo STEP3_FAILED

echo "=== STEP4 vllm eager ==="
/opt/vv/bin/python bench_vllm_ttft_tpot.py \
  --model "$M" --scenarios "$SCEN" --enforce-eager \
  --repeats 3 --json /workspace/vllm_eager.json || echo STEP4_FAILED

echo "=== ALL_DONE ==="
