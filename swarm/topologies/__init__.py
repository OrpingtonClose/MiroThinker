# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm communication topologies.

Three-tier hierarchy for large-corpus synthesis:

    Leaf Workers (tiny slices, ~15-25 findings each)
        ↕ summaries only
    Angle Coordinators (one per top-level angle)
        ↕ condensed summaries only
    Cross-Angle Bridge Workers (mesh topology, CollisionStatements)
        ↕ collision statements only
    Serendipity Panel (concurrent polymaths, different lenses)
        ↕ top collisions only
    Queen (editor, never sees raw data)

No agent at any tier ever holds the full corpus.
"""

from swarm.topologies.collision import CollisionStatement
from swarm.topologies.coordinator import coordinate_all_angles, coordinate_angle
from swarm.topologies.hierarchy import LeafCluster, cluster_findings
from swarm.topologies.mesh import BridgeWorker, run_mesh_rounds
from swarm.topologies.serendipity_panel import run_serendipity_panel

__all__ = [
    "CollisionStatement",
    "LeafCluster",
    "BridgeWorker",
    "cluster_findings",
    "coordinate_all_angles",
    "coordinate_angle",
    "run_mesh_rounds",
    "run_serendipity_panel",
]
