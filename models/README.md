# MiroThinker — Local Model Serving

MiroThinker's gossip swarm **only** accepts localhost URLs (a guard rejects remote APIs at startup and before every LLM call). Workers and queen merge run on local uncensored models via Ollama.

## Architecture

```
┌─────────────────────────┐      HTTP (OpenAI-compatible)      ┌──────────────────┐
│  CPU Instance            │  ──────────────────────────────→  │  GPU Instance      │
│  - Proxies (ports 9100+) │                                    │  - Ollama (:11434) │
│  - LibreChat (:3080)     │  ←──────────────────────────────  │  - qwen-claude-opus│
│  - strands-agent (:8100) │      Streaming responses           │  - gemma-4-uncen.  │
│  - MongoDB, Neo4j        │                                    │                    │
└─────────────────────────┘                                    └──────────────────┘
```

Proxies and agents are **CPU-only** — they never touch the GPU. The model serving layer communicates via HTTP and can run on a completely separate machine.

## Models

| Model | Role | Size | VRAM | Context | Refusals |
|-------|------|------|------|---------|----------|
| **qwen-claude-opus** | Queen merge | 15.4 GB | ~20 GB | 262K | 0% |
| **gemma-4-uncensored** | Swarm worker | 15.6 GB | ~18 GB | 256K | 0.9% |

Both fit on a single RTX 4090 (24 GB). For larger models, use remote APIs (Venice, OpenRouter).

## Quick Start

### 1. Download models from B2

```bash
export B2_APPLICATION_KEY_ID=<your-key-id>
export B2_APPLICATION_KEY=<your-key>
bash models/download.sh
```

### 2. Or pull from Ollama Hub

```bash
ollama run huihui_ai/qwen3.5-abliterated:27b-Claude    # Queen
ollama run trevorjs/gemma-4-26b-a4b-uncensored          # Workers
```

### 3. Configure proxies to use local models

Set in your `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434    # or http://<gpu-host>:11434
```

## B2 Bucket Structure

```
swarm-local-models/
├── manifest.json
├── qwen3.5-27b-claude-opus-abliterated/
│   └── Q4_K_M.gguf                      (15.4 GB)
└── gemma-4-26b-a4b-uncensored/
    └── Q4_K_M.gguf                      (15.6 GB)
```

## GPU Tier Guide

| GPU | VRAM | Can Run |
|-----|------|---------|
| RTX 4090 | 24 GB | Both models (one at a time with full context) |
| A6000 | 48 GB | Both models simultaneously |
| H100/H200 | 80-144 GB | All models + room for larger future additions |

## API-Only Models (too large for local)

For tasks requiring stronger models, use remote APIs:
- **GLM 5.1** (334 GB) — Venice API: `zai-org-glm-5-1`
- **Mistral Small 4 119B** — HuggingFace GGUF, needs H100 80GB
- **Kimi Linear 48B** — Linear attention, constant memory, needs H100+

See `manifest.json` for full metadata.
