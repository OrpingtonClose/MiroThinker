# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MiroThinker Gossip Swarm Engine — parallel corpus synthesis with full-corpus gossip.

Supports per-phase model selection:
- Workers: fast, uncensored, parallel (e.g. local Gemma-4 on Ollama)
- Queen: best available writer (e.g. Qwen-Claude-Opus or remote Claude)
- Serendipity: best cross-domain reasoner

Lineage tracking:
- Pass ``lineage_store=InMemoryLineageStore()`` in SwarmConfig to record
  every phase output with parent pointers forming a DAG.

Quality manifest:
- Computed provenance footer appended to every report (not LLM-generated).

Usage:
    from swarm import GossipSwarm, SwarmConfig, InMemoryLineageStore

    store = InMemoryLineageStore()
    swarm = GossipSwarm(
        complete=my_worker_llm,
        queen_complete=my_queen_llm,
        config=SwarmConfig(gossip_rounds=3, lineage_store=store),
    )
    result = await swarm.synthesize(corpus="...", query="...")
    print(result.user_report)
    print(result.knowledge_report)
    print(result.quality_manifest.to_markdown())
    for entry in store.entries:
        print(entry.phase, entry.angle, len(entry.content))
"""

from swarm.config import CompleteFn, SwarmConfig
from swarm.engine import GossipSwarm, SwarmMetrics, SwarmResult
from swarm.flock_query_manager import (
    CloneContext,
    FlockQueryManager,
    FlockQueryManagerConfig,
    FlockSwarmResult,
    QueryType,
)
from swarm.lineage import InMemoryLineageStore, LineageEntry, LineageStore
from swarm.mcp_researcher import (
    MCPResearcherConfig,
    MCPResearchRoundMetrics,
    run_mcp_research_round,
)
from swarm.quality_manifest import SwarmQualityManifest

__all__ = [
    "CloneContext",
    "CompleteFn",
    "FlockQueryManager",
    "FlockQueryManagerConfig",
    "FlockSwarmResult",
    "GossipSwarm",
    "InMemoryLineageStore",
    "LineageEntry",
    "LineageStore",
    "MCPResearcherConfig",
    "MCPResearchRoundMetrics",
    "QueryType",
    "SwarmConfig",
    "SwarmMetrics",
    "SwarmQualityManifest",
    "SwarmResult",
    "run_mcp_research_round",
]
