#!/usr/bin/env bash
# =============================================================================
# Launch vLLM on a single H200 GPU for swarm testing.
#
# Usage:
#   ./launch_vllm.sh [MODEL_NAME] [PORT]
#
# Defaults:
#   MODEL_NAME = huihui-ai/Qwen3.5-32B-abliterated
#   PORT       = 8000
#
# Environment variables:
#   VLLM_MODEL        — override model name
#   VLLM_PORT         — override port
#   VLLM_MAX_MODEL_LEN — max context length (default: 32768)
#   VLLM_GPU_UTIL     — GPU memory utilization (default: 0.92)
#   VLLM_DTYPE        — data type: auto, float16, bfloat16, float8 (default: auto)
#   HF_TOKEN          — HuggingFace token for gated models
#
# The script serves an OpenAI-compatible API at http://localhost:PORT/v1
# which the swarm bridge consumes via SWARM_API_BASE.
# =============================================================================
set -euo pipefail

MODEL="${VLLM_MODEL:-${1:-huihui-ai/Qwen3.5-32B-abliterated}}"
PORT="${VLLM_PORT:-${2:-8000}}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"
DTYPE="${VLLM_DTYPE:-auto}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MiroThinker — vLLM Swarm Test Launcher (1×H200)           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Model:     ${MODEL}"
echo "║  Port:      ${PORT}"
echo "║  Max Ctx:   ${MAX_MODEL_LEN}"
echo "║  GPU Util:  ${GPU_UTIL}"
echo "║  Dtype:     ${DTYPE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Verify GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Is this an H200 instance?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "Detected GPU: ${GPU_NAME} (${GPU_MEM} MiB)"
echo ""

# Install vLLM if not present
if ! command -v vllm &> /dev/null; then
    echo "Installing vLLM..."
    pip install vllm --quiet
fi

# Launch vLLM with OpenAI-compatible API
exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --dtype "${DTYPE}" \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-seqs 32 \
    --api-key "local-test" \
    --served-model-name "${MODEL}"
