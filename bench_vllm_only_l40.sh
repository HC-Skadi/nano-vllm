#!/usr/bin/env bash
# vLLM-only steps of the L40 comparison (rerun after env fix).
set -x
# torch 2.11 wheel is cu130; pin the toolkit so flashinfer JIT matches torch.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
cd /workspace/nano-vllm
M=/workspace/DeepSeek-V2-Lite-Chat
SCEN=1:512:128,4:512:128,8:512:128,1:2048:128

echo "=== STEP3 vllm default ==="
/opt/vv/bin/python bench_vllm_ttft_tpot.py \
  --model "$M" --scenarios "$SCEN" \
  --repeats 3 --json /workspace/vllm_default.json || echo STEP3_FAILED

echo "=== STEP4 vllm eager ==="
/opt/vv/bin/python bench_vllm_ttft_tpot.py \
  --model "$M" --scenarios "$SCEN" --enforce-eager \
  --repeats 3 --json /workspace/vllm_eager.json || echo STEP4_FAILED

echo "=== VLLM_DONE ==="
