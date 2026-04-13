# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Concrete pipeline blocks — one per phase.

Each block is a fenced ``PipelineBlock`` subclass containing ONLY business
logic.  Cross-cutting concerns are handled by aspects.
"""

from blocks.scout_block import ScoutBlock
from blocks.thinker_block import ThinkerBlock
from blocks.search_executor_block import SearchExecutorBlock
from blocks.maestro_block import MaestroBlock
from blocks.swarm_block import SwarmSynthesisBlock
from blocks.synthesiser_block import SynthesiserBlock

__all__ = [
    "ScoutBlock",
    "ThinkerBlock",
    "SearchExecutorBlock",
    "MaestroBlock",
    "SwarmSynthesisBlock",
    "SynthesiserBlock",
]
