"""
Hybrid Serendipity Reranker.

Combines multiple serendipity algorithms and strategies to provide
configurable balance between accuracy and discovery.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..algorithms.base import SerendipityItem, SerendipityScore
from ..algorithms.sog import SOGAlgorithm
from ..algorithms.kfn import KFNAlgorithm


@dataclass
class RerankerConfig:
    """
    Configuration for hybrid reranker.
    
    Attributes:
        relevance_weight: Weight for relevance component (0-1)
        serendipity_weight: Weight for serendipity component (0-1)
        diversity_weight: Weight for diversity component (0-1)
        algorithm_name: Name of primary serendipity algorithm
        top_k: Number of items to return
    """
    relevance_weight: float = 0.7
    serendipity_weight: float = 0.3
    diversity_weight: float = 0.0
    algorithm_name: str = "sog"
    top_k: int = 10
    
    def __post_init__(self):
        """Validate weights sum to approximately 1.0"""
        total = self.relevance_weight + self.serendipity_weight
        if abs(total - 1.0) > 0.01:
            # Normalize weights
            self.relevance_weight /= total
            self.serendipity_weight /= total


class HybridSerendipityReranker:
    """
    Hybrid reranker that balances relevance and serendipity.
    
    This reranker combines:
    1. Pre-computed relevance scores
    2. Serendipity scores from configured algorithm
    3. Optional diversity component
    
    The hybrid score is computed as:
        hybrid = w_rel * relevance + w_ser * serendipity + w_div * diversity
    
    Example:
        >>> config = {'reranker': {'relevance_weight': 0.6, 'serendipity_weight': 0.4}}
        >>> reranker = HybridSerendipityReranker(config)
        >>> reranked = reranker.rerank(items, context, top_k=10)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid reranker.
        
        Args:
            config: Configuration dictionary with 'reranker' and 'algorithm' sections
        """
        self.config = RerankerConfig(**config.get('reranker', {}))
        self.algorithm = self._load_algorithm(
            self.config.algorithm_name,
            config.get('algorithm', {})
        )
    
    def _load_algorithm(
        self,
        name: str,
        algo_config: Dict[str, Any]
    ) -> Any:
        """
        Load the specified serendipity algorithm.
        
        Args:
            name: Algorithm name ('sog', 'kfn', etc.)
            algo_config: Algorithm-specific configuration
            
        Returns:
            Initialized algorithm instance
        """
        algorithms = {
            'sog': SOGAlgorithm,
            'kfn': KFNAlgorithm,
            # Add more as implemented
        }
        
        algo_class = algorithms.get(name, SOGAlgorithm)
        return algo_class(algo_config)
    
    def rerank(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[SerendipityItem]:
        """
        Re-rank items balancing relevance and serendipity.
        
        Args:
            items: Items to rerank
            context: Context including user profile, query, etc.
            top_k: Number of items to return (overrides config)
            
        Returns:
            Re-ranked items
        """
        if not items:
            return []
        
        k = top_k or self.config.top_k
        
        if len(items) <= k:
            return items
        
        # Calculate hybrid scores for all items
        scored_items = []
        for item in items:
            # Get serendipity score
            ser_score = self.algorithm.calculate_serendipity(item, context)
            
            # Calculate hybrid score
            hybrid_score = self._calculate_hybrid_score(
                item.relevance_score,
                ser_score.overall
            )
            
            scored_items.append((item, hybrid_score, ser_score))
        
        # Sort by hybrid score (descending)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _, _ in scored_items[:k]]
    
    def _calculate_hybrid_score(
        self,
        relevance: float,
        serendipity: float
    ) -> float:
        """
        Calculate hybrid score combining relevance and serendipity.
        
        Args:
            relevance: Relevance score (0-1)
            serendipity: Serendipity score (0-1)
            
        Returns:
            Hybrid score (0-1)
        """
        return (
            self.config.relevance_weight * relevance +
            self.config.serendipity_weight * serendipity
        )
    
    def get_serendipity_scores(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any]
    ) -> Dict[str, SerendipityScore]:
        """
        Get serendipity scores for all items without reranking.
        
        Args:
            items: Items to score
            context: Context information
            
        Returns:
            Dictionary mapping item IDs to their serendipity scores
        """
        scores = {}
        for item in items:
            scores[item.id] = self.algorithm.calculate_serendipity(item, context)
        return scores
    
    def get_top_serendipitous(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any],
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> List[SerendipityItem]:
        """
        Get items that are specifically serendipitous (high unexpectedness + relevance).
        
        Args:
            items: Items to filter
            context: Context information
            threshold: Minimum serendipity score
            top_k: Maximum number to return
            
        Returns:
            List of serendipitous items
        """
        scores = self.get_serendipity_scores(items, context)
        
        # Filter and sort by serendipity score
        serendipitous = [
            (item, scores[item.id])
            for item in items
            if scores[item.id].is_serendipitous(threshold)
        ]
        
        serendipitous.sort(key=lambda x: x[1].overall, reverse=True)
        
        k = top_k or len(serendipitous)
        return [item for item, _ in serendipitous[:k]]
