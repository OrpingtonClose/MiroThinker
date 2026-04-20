# MiroThinker Gossip Swarm Engine

Parallel corpus synthesis with full-corpus gossip, serendipity bridge, and adaptive convergence.

## Architecture

```
    ┌──────────────────────────────────────────────────────────┐
    │                    QUEEN (merge)                          │
    │   Reads all gossip-refined summaries + serendipity        │
    │   insights → produces final unified synthesis             │
    └──────────┬──────────┬──────────┬──────────┬─────────────┘
               │          │          │          │
    ┌──────────▼──┐ ┌─────▼─────┐ ┌──▼────────┐ │
    │  Worker A   │ │ Worker B  │ │ Worker C  │ ...
    │  Angle 1    │ │ Angle 2   │ │ Angle 3   │
    │  + raw data │ │ + raw data│ │ + raw data│
    └──────┬──────┘ └──────┬────┘ └──────┬────┘
           │               │             │
           └───────────────┼─────────────┘
                    Gossip Round(s)
              (each worker reads peers'
               summaries + own raw section,
               refines with cross-references)
                           │
               ┌───────────▼───────────┐
               │  Serendipity Bridge   │
               │  Polymath connector   │
               │  finds cross-angle    │
               │  surprises            │
               └───────────────────────┘
```

## Phases

| Phase | Name | Parallel? | LLM Calls | Purpose |
|-------|------|-----------|-----------|---------|
| 0 | Corpus Analysis | N/A | 0-1 | Detect sections, extract angles, assign workers |
| 1 | Map | Yes | N workers | Each worker synthesizes its assigned section |
| 2 | Gossip | Yes | N × rounds | Workers read peers' summaries + own raw data, refine |
| 3 | Serendipity | No | 1 | Polymath finds cross-angle surprises |
| 4 | Queen Merge | No | 1 | Combines everything into final synthesis |

## Key Innovation: Full-Corpus Gossip

In standard gossip protocols, workers only see compressed summaries from peers. Our benchmark showed this loses granular detail before cross-referencing happens.

**Full-corpus gossip** (enabled by default): during the gossip round, each worker retains access to its FULL original corpus section alongside peer summaries. When Worker 1 reads Worker 5's forum findings, it can go back to its own raw data and pull out specific details that become relevant in light of the peer's insight.

Benchmark result: 9/10 cross-referencing (full-corpus) vs 8/10 (summary-only).

## Usage

```python
import asyncio
from swarm import GossipSwarm, SwarmConfig

# Any async callable that takes a prompt and returns a string
async def my_llm(prompt: str) -> str:
    # Call Venice, Ollama, OpenAI, etc.
    ...

swarm = GossipSwarm(
    complete=my_llm,
    config=SwarmConfig(
        max_workers=6,
        gossip_rounds=2,
        enable_full_corpus_gossip=True,
        enable_serendipity=True,
        enable_adaptive_rounds=True,
    ),
)

result = await swarm.synthesize(
    corpus="your large corpus text...",
    query="what are the mechanisms of X?",
)

print(result.synthesis)        # Final merged output
print(result.metrics)          # Telemetry (LLM calls, timing, etc.)
print(result.worker_summaries) # Per-angle summaries
print(result.serendipity_insights)  # Cross-angle connections
```

## Configuration

All settings have environment variable overrides (prefixed `SWARM_`):

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `max_workers` | `SWARM_MAX_WORKERS` | 6 | Max parallel specialist workers |
| `gossip_rounds` | `SWARM_GOSSIP_ROUNDS` | 1 | Number of gossip refinement rounds |
| `max_summary_chars` | `SWARM_MAX_SUMMARY_CHARS` | 6000 | Max chars per worker summary |
| `max_section_chars` | `SWARM_MAX_SECTION_CHARS` | 30000 | Max chars per corpus section |
| `convergence_threshold` | `SWARM_CONVERGENCE_THRESHOLD` | 0.85 | Jaccard similarity for adaptive stop |
| `context_budget` | `SWARM_CONTEXT_BUDGET` | 100000 | Target token budget for queen merge |
| `enable_serendipity` | `SWARM_SERENDIPITY` | 1 | Enable cross-angle serendipity bridge |
| `enable_full_corpus_gossip` | `SWARM_FULL_CORPUS_GOSSIP` | 1 | Workers retain raw data during gossip |
| `enable_adaptive_rounds` | `SWARM_ADAPTIVE_ROUNDS` | 1 | Stop gossip early on convergence |

## Model Agnostic

The engine accepts any `async (str) -> str` callable. It does not import or depend on any specific LLM SDK. Example backends:

```python
# Venice API
async def venice_complete(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.venice.ai/api/v1/chat/completions",
            json={"model": "glm-5", "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {VENICE_API_KEY}"},
        )
        return resp.json()["choices"][0]["message"]["content"]

# Local Ollama
async def ollama_complete(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": "qwen-claude-opus", "prompt": prompt},
        )
        return resp.json()["response"]
```

## Benchmark Results (84K char steroid-insulin corpus)

| Approach | Output | Time | Quality | Cross-Ref | Serendipity |
|----------|--------|------|---------|-----------|-------------|
| Single-Pass | 11,843 | 5.6m | 6/10 | 5/10 | 4/10 |
| Ruflo Gossip | 7,260 | 22.1m | 8/10 | 8/10 | 9/10 |
| **Angle Gossip** | **9,453** | **19.6m** | **9/10** | **9/10** | **7/10** |
| **→ Parallel** | est. | **~7.2m** | same | same | same |

## File Structure

```
swarm/
├── __init__.py       # Public API: GossipSwarm, SwarmConfig, SwarmResult
├── config.py         # SwarmConfig dataclass + CompleteFn type alias
├── engine.py         # GossipSwarm orchestrator (phases 0-4)
├── angles.py         # Section detection, angle extraction, worker assignment
├── worker.py         # Worker synthesis + gossip refinement prompts
├── queen.py          # Queen merge prompt + fallback concatenation
├── serendipity.py    # Cross-angle polymath bridge
├── convergence.py    # Adaptive convergence detection (trigram Jaccard)
└── README.md         # This file
```

## Lineage

Built from proven patterns in three existing implementations:

| Feature | Source | Adaptation |
|---------|--------|------------|
| Gossip protocol | `proxies/tools/ruflo_synthesis.py` | Kept CRDT append-only model, added full-corpus access |
| Angle-based workers | `apps/adk-agent/tools/swarm_thinkers.py` | Replaced DuckDB corpus dependency with plain text sections |
| Serendipity bridge | `apps/adk-agent/tools/swarm_thinkers.py` | Generalized from finding-row citations to free-text |
| Convergence detection | `apps/adk-agent/tools/swarm_thinkers.py` | Simplified from multi-signal (LLM+trigram+exhaustion) to trigram Jaccard |
| Parallel execution | `proxies/tools/ruflo_synthesis.py` | Kept semaphore-bounded asyncio.gather pattern |
| Queen merge | `proxies/tools/ruflo_synthesis.py` | Added serendipity injection + fallback concatenation |
