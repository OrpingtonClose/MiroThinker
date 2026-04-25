#!/bin/bash
# Full H200 multi-model pipeline launcher.
#
# Architecture:
#   Phase 1 (bees+gossip): DeepSeek V4 Flash via API (unlimited concurrency)
#   Phase 2 (Flock):       DeepSeek V4 Pro on 8×H200 via vLLM Docker
#                          (vllm/vllm-openai:deepseekv4-cu130)
#
# Tracing: OpenTelemetry spans for every phase, worker, query, and LLM call.
#          Exported to JSONL files + optional B2 upload.
#
# Usage:
#   # On 8×H200 after git pull + data transfer:
#   ./launch_pipeline.sh
#
#   # With B2 upload of traces:
#   UPLOAD_B2=1 ./launch_pipeline.sh
#
# Environment:
#   DEEPSEEK_API_KEY    — required for V4 Flash worker bees
#   OPENROUTER_API_KEY  — fallback for workers if no DEEPSEEK_API_KEY
#   HF_HOME             — HuggingFace cache dir (default: /workspace/models)
#   LOCAL_MODEL          — local vLLM model (fallback for workers)
#   FLOCK_MODEL          — Flock evaluation model (default: DeepSeek V4 Pro)
#   NUM_GPUS             — GPU count (default: 8)
#   CORPUS_PATH          — path to corpus file
#   DB_PATH              — path to enriched DuckDB
#   OUTPUT_DIR           — results directory (default: /workspace/results)
#   TRACE_DIR            — trace file directory (default: /workspace/traces)
#   UPLOAD_B2            — set to 1 to upload traces+outputs to B2
#   B2_APPLICATION_KEY_ID — Backblaze B2 key ID (for B2 upload)
#   B2_APPLICATION_KEY   — Backblaze B2 key (for B2 upload)
#   B2_BUCKET            — B2 bucket name (default: mirothinker-traces)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────

LOCAL_MODEL="${LOCAL_MODEL:-huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated}"
FLOCK_MODEL="${FLOCK_MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
NUM_GPUS="${NUM_GPUS:-8}"
HF_HOME="${HF_HOME:-/workspace/models}"
CORPUS_PATH="${CORPUS_PATH:-/workspace/data/full_corpus.txt}"
DB_PATH="${DB_PATH:-/workspace/data/enriched.duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results}"
TRACE_DIR="${TRACE_DIR:-/workspace/traces}"
UPLOAD_B2="${UPLOAD_B2:-0}"
B2_BUCKET="${B2_BUCKET:-mirothinker-traces}"

export HF_HOME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "═══════════════════════════════════════════════════════════════"
echo "  H200 MULTI-MODEL PIPELINE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Worker bees:   DeepSeek V4 Flash (API)"
echo "  Flock model:   $FLOCK_MODEL (local vLLM Docker)"
echo "  Local fallback: $LOCAL_MODEL"
echo "  GPUs:          $NUM_GPUS"
echo "  Corpus:        $CORPUS_PATH"
echo "  DB:            $DB_PATH"
echo "  Output:        $OUTPUT_DIR"
echo "  Traces:        $TRACE_DIR"
echo "  B2 upload:     $UPLOAD_B2"
echo "  HF cache:      $HF_HOME"
echo "═══════════════════════════════════════════════════════════════"

# ── Step 1: Verify prerequisites ──────────────────────────────────────

echo ""
echo "▶ Step 1: Verifying prerequisites..."

if [ ! -f "$CORPUS_PATH" ] && [ ! -f "$DB_PATH" ]; then
    echo "  ✗ Neither corpus ($CORPUS_PATH) nor DB ($DB_PATH) found"
    echo "    Transfer data first: scp local_corpus.txt H200:$CORPUS_PATH"
    exit 1
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "  ⚠ DEEPSEEK_API_KEY not set — worker bees will need OPENROUTER_API_KEY or local vLLM"
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "  ⚠ OPENROUTER_API_KEY also not set — workers will use local vLLM only"
    fi
else
    echo "  DeepSeek API key present — workers will use V4 Flash"
fi

# Verify Docker is available (needed for V4 Pro)
if ! command -v docker &>/dev/null; then
    echo "  ⚠ Docker not found — Flock will fall back to standard vLLM"
fi

# Pull the vLLM deepseekv4 Docker image if not present
if command -v docker &>/dev/null; then
    if ! docker image inspect vllm/vllm-openai:deepseekv4-cu130 &>/dev/null; then
        echo "  Pulling vLLM deepseekv4-cu130 Docker image..."
        docker pull vllm/vllm-openai:deepseekv4-cu130
    else
        echo "  vLLM deepseekv4-cu130 Docker image present"
    fi
fi

echo "  Prerequisites OK"

# ── Step 2: Download Flock model weights ──────────────────────────────

echo ""
echo "▶ Step 2: Downloading Flock model weights..."
echo "  (Worker bees use DeepSeek API — no model download needed)"

download_model() {
    local model="$1"
    local label="$2"
    echo "  Downloading $label: $model"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model', local_dir=None)
print('  Done: $label downloaded')
" 2>&1 | tail -1
}

# Only need the Flock model for local serving (V4 Pro)
download_model "$FLOCK_MODEL" "flock"

# ── Step 3: Install tracing dependencies ──────────────────────────────

echo ""
echo "▶ Step 3: Installing tracing dependencies..."

pip install -q opentelemetry-sdk opentelemetry-exporter-otlp-proto-http b2sdk httpx 2>/dev/null || true

echo "  Tracing dependencies OK"

# ── Step 4: Run multi-model pipeline ──────────────────────────────────

echo ""
echo "▶ Step 4: Running multi-model pipeline..."
echo "  Phase 1: Worker bees via DeepSeek V4 Flash API"
echo "  Phase 2: Flock evaluation via DeepSeek V4 Pro (local Docker)"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR"

PIPELINE_ARGS=(
    --multi-model
    --engine gossip
    --local-model "$LOCAL_MODEL"
    --flock-model "$FLOCK_MODEL"
    --num-gpus "$NUM_GPUS"
    --output-dir "$OUTPUT_DIR"
    --trace-dir "$TRACE_DIR"
)

if [ -f "$CORPUS_PATH" ]; then
    PIPELINE_ARGS+=(--corpus "$CORPUS_PATH")
fi

if [ -f "$DB_PATH" ]; then
    PIPELINE_ARGS+=(--db "$DB_PATH")
fi

if [ "$UPLOAD_B2" = "1" ]; then
    PIPELINE_ARGS+=(--upload-b2 --b2-bucket "$B2_BUCKET")
fi

cd "$REPO_ROOT"
python3 scripts/h200_test/run_swarm_test.py "${PIPELINE_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/pipeline.log"

# ── Step 5: Results ───────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo "  Results:  $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"*.md "$OUTPUT_DIR/"*.json 2>/dev/null || echo "  (no output files found)"
echo ""
echo "  Traces:   $TRACE_DIR/"
ls -lh "$TRACE_DIR/"*.jsonl 2>/dev/null || echo "  (no trace files found)"
echo ""
if [ "$UPLOAD_B2" = "1" ]; then
    echo "  B2 upload: enabled (check pipeline log for URLs)"
else
    echo "  B2 upload: disabled (set UPLOAD_B2=1 to enable)"
fi
echo ""
echo "  Copy results back:"
echo "    scp -r H200:$OUTPUT_DIR/ ./results/"
echo "    scp -r H200:$TRACE_DIR/ ./traces/"
echo "═══════════════════════════════════════════════════════════════"
