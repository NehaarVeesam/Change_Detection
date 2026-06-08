#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-${MODEL_ID:-Qwen/Qwen3.5-9B}}"
PORT="${PORT:-${VLLM_PORT:-7100}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-16384}}"
GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:- --max-num-seqs 24}"

export MODEL_ID="${MODEL_ID:-$MODEL_NAME}"

ARGS=(
  vllm serve "$MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM"
)

if [[ "$MODEL_NAME" == *"Qwen3.5"* ]] || [[ "$MODEL_NAME" == *"Qwen3_5"* ]]; then
  ARGS+=(--reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 2}')
fi

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=(${VLLM_EXTRA_ARGS})
  ARGS+=("${EXTRA[@]}")
fi

echo "[cd-vllm] ${ARGS[*]}"
exec "${ARGS[@]}"
