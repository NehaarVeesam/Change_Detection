#!/usr/bin/env bash
# Start vLLM for Qwen VL / Qwen3.5 models (default :7100 to avoid clashing with :8000).
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-${MODEL_ID:-Qwen/Qwen3.5-9B}}"
PORT="${PORT:-${VLLM_PORT:-7100}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-32768}}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

export MODEL_ID="${MODEL_ID:-$MODEL_NAME}"

ARGS=(
  vllm serve "$MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --trust-remote-code
)

if [[ "$MODEL_NAME" == *"Qwen3.5"* ]] || [[ "$MODEL_NAME" == *"Qwen3_5"* ]]; then
  ARGS+=(--reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 2}')
elif [[ "$MODEL_NAME" == *"Qwen3-VL"* ]] || [[ "$MODEL_NAME" == *"Qwen2.5-VL"* ]] || [[ "$MODEL_NAME" == *"Qwen2_5-VL"* ]]; then
  ARGS+=(--limit-mm-per-prompt '{"image": 10}')
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=(${VLLM_EXTRA_ARGS})
  ARGS+=("${EXTRA[@]}")
fi

echo "[serve_vllm] ${ARGS[*]}"
exec "${ARGS[@]}"
