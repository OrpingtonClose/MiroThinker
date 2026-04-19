#!/bin/bash
# =============================================================================
# MiroThinker — Download local swarm models from B2 and import into Ollama
# =============================================================================
set -euo pipefail

B2_BUCKET="swarm-local-models"
MODEL_DIR="${MODEL_DIR:-/workspace/swarm-models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure B2 credentials
if [ -z "${B2_APPLICATION_KEY_ID:-}" ] || [ -z "${B2_APPLICATION_KEY:-}" ]; then
    echo "ERROR: B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set"
    echo "  export B2_APPLICATION_KEY_ID=<your-key-id>"
    echo "  export B2_APPLICATION_KEY=<your-key>"
    exit 1
fi

# Ensure b2 CLI
if ! command -v b2 &>/dev/null; then
    echo "Installing b2 CLI..."
    pip install --break-system-packages b2 2>/dev/null || pip install b2
fi

# Ensure Ollama
if ! command -v ollama &>/dev/null; then
    echo "ERROR: Ollama is not installed. Install from https://ollama.com"
    exit 1
fi

mkdir -p "$MODEL_DIR"

echo "=== Downloading models from B2 bucket: $B2_BUCKET ==="

# Qwen3.5-27B-Claude-Opus (Queen — 15.4 GB)
QWEN_FILE="$MODEL_DIR/qwen3.5-27b-claude-opus-Q4_K_M.gguf"
if [ ! -f "$QWEN_FILE" ]; then
    echo "Downloading Qwen3.5-27B-Claude-Opus (15.4 GB)..."
    b2 file download "b2://$B2_BUCKET/qwen3.5-27b-claude-opus-abliterated/Q4_K_M.gguf" "$QWEN_FILE"
else
    echo "Qwen3.5-27B-Claude-Opus already downloaded"
fi

# Gemma-4-26B-A4B (Worker — 15.6 GB)
GEMMA_FILE="$MODEL_DIR/gemma-4-26b-a4b-uncensored-Q4_K_M.gguf"
if [ ! -f "$GEMMA_FILE" ]; then
    echo "Downloading Gemma-4-26B-A4B (15.6 GB)..."
    b2 file download "b2://$B2_BUCKET/gemma-4-26b-a4b-uncensored/Q4_K_M.gguf" "$GEMMA_FILE"
else
    echo "Gemma-4-26B-A4B already downloaded"
fi

echo ""
echo "=== Importing models into Ollama ==="

# Import Qwen-Claude-Opus
echo "Creating Ollama model: qwen-claude-opus..."
ollama create qwen-claude-opus -f "$SCRIPT_DIR/Modelfile.qwen-claude"

# Import Gemma-4-Uncensored
echo "Creating Ollama model: gemma-4-uncensored..."
ollama create gemma-4-uncensored -f "$SCRIPT_DIR/Modelfile.gemma"

echo ""
echo "=== Done ==="
ollama list
echo ""
echo "Test with: ollama run qwen-claude-opus 'Hello, what can you help with?'"
