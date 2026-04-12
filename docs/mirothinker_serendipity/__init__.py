"""
MiroThinker Serendipity Module

This module provides serendipity-aware algorithms and components
for the MiroThinker deep research agent framework.

Core Formula: Serendipity = Relevance × Unexpectedness
"""

from .algorithms.base import BaseSerendipityAlgorithm, SerendipityItem, SerendipityScore
from .algorithms.sog import SOGAlgorithm
from .reranker.hybrid import HybridSerendipityReranker, RerankerConfig
from .curiosity_engine.profiler import CuriosityProfiler, CuriosityProfile

__version__ = "1.0.0"

__all__ = [
    # Base classes
    'BaseSerendipityAlgorithm',
    'SerendipityItem',
    'SerendipityScore',
    # Algorithms
    'SOGAlgorithm',
    # Rerankers
    'HybridSerendipityReranker',
    'RerankerConfig',
    # Curiosity
    'CuriosityProfiler',
    'CuriosityProfile',
]
