#!/usr/bin/env bash
set -euo pipefail

# One-command DeepSeek-V2-Lite BF16 MLA benchmark for A100 40GB/80GB.
# Usage: ./run_deepseek_mla_a100.sh /models/DeepSeek-V2-Lite-Chat
# Optional: PROFILE=smoke|standard|full REPEATS=3 TP=1 OUTPUT_DIR=/results

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-${MODEL:-}}"
PYTHON_BIN="${PYTHON:-python}"
PROFILE="${PROFILE:-standard}"
REPEATS="${REPEATS:-3}"
TP="${TP:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmarks/a100_${TIMESTAMP}}"

usage() {
  cat <<'EOF'
Usage:
  ./run_deepseek_mla_a100.sh /models/DeepSeek-V2-Lite-Chat
  MODEL=/models/DeepSeek-V2-Lite-Chat ./run_deepseek_mla_a100.sh

Environment:
  PROFILE=smoke|standard|full  REPEATS=3  TP=1
  OUTPUT_DIR=/results          PYTHON=python
  CUDA_VISIBLE_DEVICES=0       TRUST_REMOTE_CODE=1
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

run_logged() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "$@" 2>&1 | tee "${OUTPUT_DIR}/${name}.log"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -n "${MODEL_PATH}" ]] || fail "provide model path as argument 1 or MODEL"
[[ -d "${MODEL_PATH}" ]] || fail "model directory does not exist: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/config.json" ]] || fail "missing ${MODEL_PATH}/config.json"
compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null \
  || fail "no safetensors files found in ${MODEL_PATH}"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
command -v "${PYTHON_BIN}" >/dev/null || fail "Python is unavailable: ${PYTHON_BIN}"
[[ "${PROFILE}" == "smoke" || "${PROFILE}" == "standard" || "${PROFILE}" == "full" ]] \
  || fail "PROFILE must be smoke, standard, or full"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || fail "REPEATS must be positive"
[[ "${TP}" =~ ^[1-8]$ ]] || fail "TP must be from 1 to 8"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p' | xargs)"
GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p' | xargs)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)"
[[ "${GPU_NAME}" == *A100* ]] || fail "expected A100, detected: ${GPU_NAME}"
[[ "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]] || fail "could not read GPU memory"
(( GPU_COUNT >= TP )) || fail "TP=${TP} requested, but only ${GPU_COUNT} GPUs are visible"
(( GPU_MEMORY_MIB >= 38000 )) \
  || fail "single-GPU BF16 needs A100 40GB/80GB; detected ${GPU_MEMORY_MIB} MiB"

if (( GPU_MEMORY_MIB >= 70000 )); then
  A100_CLASS="80GB"
  BATCHED_REQUESTS=8
  LONG_CONTEXT=4096
  OP_BATCH=16
else
  A100_CLASS="40GB"
  BATCHED_REQUESTS=4
  LONG_CONTEXT=2048
  OP_BATCH=8
fi

TRUST_ARGS=()
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  TRUST_ARGS+=(--trust-remote-code)
fi

{
  echo "created_at=${TIMESTAMP}"
  echo "model=${MODEL_PATH}"
  echo "profile=${PROFILE}"
  echo "repeats=${REPEATS}"
  echo "tensor_parallel_size=${TP}"
  echo "gpu_name=${GPU_NAME}"
  echo "gpu_memory_mib=${GPU_MEMORY_MIB}"
  echo "a100_class=${A100_CLASS}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=driver_version,pstate,power.limit --format=csv,noheader
  "${PYTHON_BIN}" -c "import torch; print(f'torch={torch.__version__}'); print(f'torch_cuda={torch.version.cuda}'); print(f'cuda_available={torch.cuda.is_available()}'); print(f'bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}')"
} > "${OUTPUT_DIR}/environment.txt"

"${PYTHON_BIN}" -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'; assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'" \
  || fail "CUDA/BF16 preflight failed; inspect ${OUTPUT_DIR}/environment.txt"

echo "DeepSeek MLA A100 benchmark"
echo "GPU: ${GPU_NAME} (${GPU_MEMORY_MIB} MiB, ${A100_CLASS})"
echo "Model: ${MODEL_PATH}"
echo "Profile: ${PROFILE}"
echo "Results: ${OUTPUT_DIR}"

run_logged operator_mla \
  "${PYTHON_BIN}" bench_deepseek_mla_ops.py \
  --cases "1:128" "1:1024" "${OP_BATCH}:1024" "${OP_BATCH}:${LONG_CONTEXT}" \
  --warmup 10 --repeats 50 \
  --json "${OUTPUT_DIR}/operator_mla.json"

run_logged e2e_smoke \
  "${PYTHON_BIN}" bench_deepseek_mla.py \
  --model "${MODEL_PATH}" --requests 1 \
  --input-tokens 128 --output-tokens 32 --repeats 1 \
  --tensor-parallel-size "${TP}" "${TRUST_ARGS[@]}" \
  --json "${OUTPUT_DIR}/e2e_smoke.json"

if [[ "${PROFILE}" != "smoke" ]]; then
  run_logged e2e_batch1_ctx512 \
    "${PYTHON_BIN}" bench_deepseek_mla.py \
    --model "${MODEL_PATH}" --requests 1 \
    --input-tokens 512 --output-tokens 128 --repeats "${REPEATS}" \
    --tensor-parallel-size "${TP}" "${TRUST_ARGS[@]}" \
    --json "${OUTPUT_DIR}/e2e_batch1_ctx512.json"

  run_logged e2e_batched_ctx512 \
    "${PYTHON_BIN}" bench_deepseek_mla.py \
    --model "${MODEL_PATH}" --requests "${BATCHED_REQUESTS}" \
    --input-tokens 512 --output-tokens 128 --repeats "${REPEATS}" \
    --tensor-parallel-size "${TP}" "${TRUST_ARGS[@]}" \
    --json "${OUTPUT_DIR}/e2e_batched_ctx512.json"

  run_logged e2e_batch1_long_context \
    "${PYTHON_BIN}" bench_deepseek_mla.py \
    --model "${MODEL_PATH}" --requests 1 \
    --input-tokens "${LONG_CONTEXT}" --output-tokens 128 --repeats "${REPEATS}" \
    --tensor-parallel-size "${TP}" "${TRUST_ARGS[@]}" \
    --json "${OUTPUT_DIR}/e2e_batch1_long_context.json"
fi

if [[ "${PROFILE}" == "full" ]]; then
  run_logged e2e_batched_long_context \
    "${PYTHON_BIN}" bench_deepseek_mla.py \
    --model "${MODEL_PATH}" --requests "${BATCHED_REQUESTS}" \
    --input-tokens "${LONG_CONTEXT}" --output-tokens 256 --repeats "${REPEATS}" \
    --tensor-parallel-size "${TP}" "${TRUST_ARGS[@]}" \
    --json "${OUTPUT_DIR}/e2e_batched_long_context.json"
fi

echo
echo "Completed. Results: ${OUTPUT_DIR}"
echo "JSON files contain metrics; log files preserve console output."
