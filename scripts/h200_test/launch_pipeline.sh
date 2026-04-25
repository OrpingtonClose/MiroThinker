#!/bin/bash
# Full H200 multi-model pipeline launcher.
#
# Handles: model download → multi-instance vLLM → bees + gossip →
#          model swap → Flock evaluation → diffusion queen → results
#
# Usage:
#   # On 8×H200 after git pull + data transfer:
#   ./launch_pipeline.sh
#
#   # With custom models:
#   LOCAL_MODEL=other/model FLOCK_MODEL=other/flock ./launch_pipeline.sh
#
# Environment:
#   OPENROUTER_API_KEY  — required for remote workers (Ling, DeepSeek)
#   HF_HOME             — HuggingFace cache dir (default: /workspace/models)
#   LOCAL_MODEL          — local vLLM model (default: Kimi-Linear-48B-A3B)
#   FLOCK_MODEL          — Flock evaluation model (default: Qwen3-235B-A22B)
#   NUM_GPUS             — GPU count (default: 8)
#   CORPUS_PATH          — path to corpus file
#   DB_PATH              — path to enriched DuckDB
#   OUTPUT_DIR           — results directory (default: /workspace/results)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────

LOCAL_MODEL="${LOCAL_MODEL:-huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated}"
FLOCK_MODEL="${FLOCK_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
NUM_GPUS="${NUM_GPUS:-8}"
HF_HOME="${HF_HOME:-/workspace/models}"
CORPUS_PATH="${CORPUS_PATH:-/workspace/data/full_corpus.txt}"
DB_PATH="${DB_PATH:-/workspace/data/enriched.duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results}"

export HF_HOME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo "  H200 MULTI-MODEL PIPELINE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Local model:   $LOCAL_MODEL"
echo "  Flock model:   $FLOCK_MODEL"
echo "  GPUs:          $NUM_GPUS"
echo "  Corpus:        $CORPUS_PATH"
echo "  DB:            $DB_PATH"
echo "  Output:        $OUTPUT_DIR"
echo "  HF cache:      $HF_HOME"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Download models in parallel ───────────────────────────────

echo ""
echo "▶ Step 1: Downloading models (parallel)..."

download_model() {
    local model="$1"
    local label="$2"
    echo "  Downloading $label: $model"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model', local_dir=None)
print('  ✓ $label downloaded')
" 2>&1 | tail -1
}

# Download both models in parallel
download_model "$LOCAL_MODEL" "local" &
PID_LOCAL=$!
download_model "$FLOCK_MODEL" "flock" &
PID_FLOCK=$!

# Wait for local model (needed first)
if ! wait $PID_LOCAL; then
    echo "  ✗ Local model download failed — aborting"
    exit 1
fi
echo "  Local model ready"

# Wait for Flock model too — Phase 2 (swap_to_flock_model) needs it
# before starting vLLM.  Both downloads run in parallel so total time
# is max(local, flock), not local + flock.
if ! wait $PID_FLOCK; then
    echo "  ✗ Flock model download failed — Phase 2 will be skipped"
fi
echo "  Flock model ready"

# ── Step 2: Verify prerequisites ──────────────────────────────────────

echo ""
echo "▶ Step 2: Verifying prerequisites..."

if [ ! -f "$CORPUS_PATH" ] && [ ! -f "$DB_PATH" ]; then
    echo "  ✗ Neither corpus ($CORPUS_PATH) nor DB ($DB_PATH) found"
    echo "    Transfer data first: scp local_corpus.txt H200:$CORPUS_PATH"
    exit 1
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "  ⚠ OPENROUTER_API_KEY not set — remote workers disabled"
    echo "    All workers will use local model only"
fi

echo "  Prerequisites OK"

# ── Step 3: Run multi-model pipeline ──────────────────────────────────

echo ""
echo "▶ Step 3: Running multi-model pipeline..."
echo "  This starts vLLM instances, runs bees+gossip+queen,"
echo "  swaps to Flock model, runs evaluation."

mkdir -p "$OUTPUT_DIR"

PIPELINE_ARGS=(
    --multi-model
    --engine gossip
    --local-model "$LOCAL_MODEL"
    --flock-model "$FLOCK_MODEL"
    --num-gpus "$NUM_GPUS"
    --output-dir "$OUTPUT_DIR"
)

if [ -f "$CORPUS_PATH" ]; then
    PIPELINE_ARGS+=(--corpus "$CORPUS_PATH")
fi

if [ -f "$DB_PATH" ]; then
    PIPELINE_ARGS+=(--db "$DB_PATH")
fi

cd "$REPO_ROOT"
python3 scripts/h200_test/run_swarm_test.py "${PIPELINE_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/pipeline.log"

# ── Step 4: Post-pipeline ─────────────────────────────────────────────

# ── Step 5: Results ───────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Results:  $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"*.md "$OUTPUT_DIR/"*.json 2>/dev/null || echo "  (no output files found)"
echo ""
echo "  Copy results back:"
echo "    scp -r H200:$OUTPUT_DIR/ ./results/"
echo "═══════════════════════════════════════════════════════════════"
