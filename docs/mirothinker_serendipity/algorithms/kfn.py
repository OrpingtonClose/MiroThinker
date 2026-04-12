"""
k-Furthest Neighbors Algorithm Implementation.

k-Furthest Neighbors is a collaborative filtering variant that
selects items furthest from the user's typical preferences,
promoting discovery of unexpected but potentially useful items.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseSerendipityAlgorithm, SerendipityItem, SerendipityScore


class KFNAlgorithm(BaseSerendipityAlgorithm):
    """
    k-Furthest Neighbors algorithm for serendipity.
    
    Unlike k-NN which finds similar items, k-FN finds items
    that are most different from the user's historical preferences,
    creating opportunities for serendipitous discovery.
    
    Configuration Options:
        - k: Number of furthest neighbors to consider (default: 5)
        - distance_metric: Distance metric to use (default: 'cosine')
        - min_distance_threshold: Minimum distance to be considered (default: 0.3)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize k-FN algorithm.
        
        Args:
            config: Configuration with optional keys:
                - k: Number of furthest neighbors
                - distance_metric: Distance metric ('cosine', 'euclidean', 'manhattan')
                - min_distance_threshold: Minimum distance threshold
        """
        super().__init__(config)
        self.k = config.get('k', 5)
        self.distance_metric = config.get('distance_metric', 'cosine')
        self.min_distance_threshold = config.get('min_distance_threshold', 0.3)
    
    def calculate_serendipity(
        self,
        item: SerendipityItem,
        context: Dict[str, Any]
    ) -> SerendipityScore:
        """
        Calculate serendipity using k-FN approach.
        
        Unexpectedness is based on distance from user's centroid
        (average of historical item features).
        
        Args:
            item: Item to evaluate
            context: Context with 'history' of items
            
        Returns:
            SerendipityScore with components
        """
        history = context.get('history', [])
        
        # Calculate unexpectedness as distance from user centroid
        unexpectedness = self._calculate_kfn_unexpectedness(item, history)
        
        # Use pre-computed relevance
        relevance = item.relevance_score
        
        # Apply serendipity formula
        overall = self.compute_serendipity_formula(relevance, unexpectedness)
        
        return SerendipityScore(
            overall=overall,
            relevance=relevance,
            unexpectedness=unexpectedness
        )
    
    def _calculate_kfn_unexpectedness(
        self,
        item: SerendipityItem,
        history: List[SerendipityItem]
    ) -> float:
        """
        Calculate unexpectedness using k-FN approach.
        
        1. Compute user centroid from history
        2. Calculate distance from item to centroid
        3. Unexpectedness = normalized distance
        
        Args:
            item: Item to evaluate
            history: User's historical items
            
        Returns:
            Unexpectedness score (0-1)
        """
        if not history or item.features is None:
            return 0.5
        
        # Extract features from history
        hist_features = [
            h.features for h in history
            if h.features is not None
        ]
        
        if not hist_features:
            return 0.5
        
        # Compute user centroid
        centroid = np.mean(hist_features, axis=0)
        
        # Calculate distance to centroid
        distance = self._calculate_distance(item.features, centroid)
        
        # Normalize distance to [0, 1]
        # Assuming max possible distance is sqrt(2) for cosine, 2 for euclidean
        if self.distance_metric == 'cosine':
            max_dist = 2.0  # Cosine distance ranges 0-2
        else:
            max_dist = np.linalg.norm(np.ones_like(item.features))
        
        normalized_distance = min(1.0, distance / max_dist) if max_dist > 0 else 0.5
        
        # Apply minimum threshold
        if normalized_distance < self.min_distance_threshold:
            normalized_distance *= 0.5  # Reduce score for too-similar items
        
        return normalized_distance
    
    def _calculate_distance(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """
        Calculate distance between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Distance value
        """
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            similarity = np.dot(a, b) / (norm_a * norm_b)
            return 1.0 - similarity
        
        elif self.distance_metric == 'euclidean':
            return np.linalg.norm(a - b)
        
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(a - b))
        
        else:
            # Default to cosine
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            similarity = np.dot(a, b) / (norm_a * norm_b)
            return 1.0 - similarity
    
    def rerank(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any],
        top_k: int = 10
    ) -> List[SerendipityItem]:
        """
        Re-rank items using k-FN approach.
        
        Selects items with highest distance from user centroid
        while maintaining relevance threshold.
        
        Args:
            items: Items to rerank
            context: Context information
            top_k: Number of items to return
            
        Returns:
            Re-ranked list of items
        """
        if not items:
            return []
        
        # Calculate serendipity scores
        scored_items = []
        for item in items:
            score = self.calculate_serendipity(item, context)
            scored_items.append((item, score))
        
        # Sort by serendipity score (descending)
        scored_items.sort(key=lambda x: x[1].overall, reverse=True)
        
        # Return top_k
        return [item for item, _ in scored_items[:top_k]]
