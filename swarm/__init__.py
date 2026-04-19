# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MiroThinker Gossip Swarm Engine — parallel corpus synthesis with full-corpus gossip.

Supports per-phase model selection:
- Workers: fast, uncensored, parallel (e.g. local Gemma-4 on Ollama)
- Queen: best available writer (e.g. Qwen-Claude-Opus or remote Claude)
- Serendipity: best cross-domain reasoner

Usage:
    from swarm import GossipSwarm, SwarmConfig

    swarm = GossipSwarm(
        complete=my_worker_llm,                # used for workers by default
        queen_complete=my_queen_llm,           # optional: better model for queen
        serendipity_complete=my_polymath_llm,  # optional: cross-domain model
        config=SwarmConfig(gossip_rounds=3),
    )
    result = await swarm.synthesize(corpus="...", query="...")
    print(result.user_report)          # concise narrative (3000-6000 words)
    print(result.knowledge_report)     # full structured report (arbitrary length)
"""

from swarm.config import CompleteFn, SwarmConfig
from swarm.engine import GossipSwarm, SwarmResult

__all__ = ["CompleteFn", "GossipSwarm", "SwarmConfig", "SwarmResult"]
