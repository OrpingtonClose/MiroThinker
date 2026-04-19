# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MiroThinker Gossip Swarm Engine — parallel corpus synthesis with full-corpus gossip.

Usage:
    from swarm import GossipSwarm, SwarmConfig

    swarm = GossipSwarm(
        complete=my_llm_callable,  # async (prompt: str) -> str
        config=SwarmConfig(max_workers=6, gossip_rounds=1),
    )
    result = await swarm.synthesize(corpus="...", query="...")
"""

from swarm.config import SwarmConfig
from swarm.engine import GossipSwarm, SwarmResult

__all__ = ["GossipSwarm", "SwarmConfig", "SwarmResult"]
