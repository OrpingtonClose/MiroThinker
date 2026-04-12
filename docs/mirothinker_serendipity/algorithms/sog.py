"""
SOG (Serendipity-Oriented Greedy) Algorithm Implementation.

SOG is a re-ranking approach that greedily selects items to maximize
serendipity while maintaining diversity in the result set.

Reference: Based on serendipity-oriented greedy selection strategies
for recommendation systems.
"""

from typing import List, Dict, Any
import numpy as np
from .base import BaseSerendipityAlgorithm, SerendipityItem, SerendipityScore


class SOGAlgorithm(BaseSerendipityAlgorithm):
    """
    Serendipity-Oriented Greedy algorithm.
    
    This algorithm re-ranks items by:
    1. Calculating serendipity scores for all items
    2. Greedily selecting items that maximize serendipity
    3. Penalizing items too similar to already selected ones
    
    The diversity penalty ensures the result set contains
    varied serendipitous discoveries rather than similar ones.
    
    Configuration Options:
        - lambda_param: Balance between relevance and unexpectedness (default: 0.5)
        - diversity_weight: Weight for diversity penalty (default: 0.3)
        - similarity_threshold: Threshold for considering items similar (default: 0.7)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SOG algorithm.
        
        Args:
            config: Configuration dictionary with optional keys:
                - lambda_param: Serendipity weight
                - diversity_weight: Diversity penalty weight
                - similarity_threshold: Similarity threshold
        """
        super().__init__(config)
        self.diversity_weight = config.get('diversity_weight', 0.3)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
    
    def calculate_serendipity(
        self,
        item: SerendipityItem,
        context: Dict[str, Any]
    ) -> SerendipityScore:
        """
        Calculate serendipity using SOG approach.
        
        The unexpectedness is calculated as the distance from
        the user's historical items. More different = more unexpected.
        
        Args:
            item: Item to evaluate
            context: Context with 'user_profile' and 'history'
            
        Returns:
            SerendipityScore with components
        """
        # Get user history from context
        history = context.get('history', [])
        
        # Calculate unexpectedness based on deviation from history
        unexpectedness = self._calculate_unexpectedness(item, history)
        
        # Use pre-computed relevance or calculate if needed
        relevance = item.relevance_score
        
        # Apply serendipity formula
        overall = self.compute_serendipity_formula(relevance, unexpectedness)
        
        return SerendipityScore(
            overall=overall,
            relevance=relevance,
            unexpectedness=unexpectedness
        )
    
    def _calculate_unexpectedness(
        self,
        item: SerendipityItem,
        history: List[SerendipityItem]
    ) -> float:
        """
        Calculate unexpectedness as distance from historical items.
        
        Unexpectedness = 1 - average_similarity_to_history
        
        Args:
            item: Item to evaluate
            history: List of historical items
            
        Returns:
            Unexpectedness score (0-1)
        """
        if not history or item.features is None:
            # Neutral unexpectedness if no history or features
            return 0.5
        
        # Calculate similarity to each historical item
        similarities = []
        for hist_item in history:
            if hist_item.features is not None:
                sim = self._cosine_similarity(item.features, hist_item.features)
                similarities.append(sim)
        
        if not similarities:
            return 0.5
        
        # Unexpectedness is inverse of average similarity
        avg_similarity = np.mean(similarities)
        unexpectedness = 1.0 - avg_similarity
        
        return unexpectedness
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def rerank(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any],
        top_k: int = 10
    ) -> List[SerendipityItem]:
        """
        Greedy re-ranking for serendipity with diversity.
        
        Algorithm:
        1. Calculate serendipity scores for all items
        2. Iteratively select the item with highest score
        3. Penalize remaining items similar to selected ones
        4. Repeat until top_k items selected
        
        Args:
            items: Items to rerank
            context: Context information
            top_k: Number of items to return
            
        Returns:
            Re-ranked list of items
        """
        if not items:
            return []
        
        if len(items) <= top_k:
            return items
        
        # Calculate serendipity scores for all items
        scored_items = []
        for item in items:
            score = self.calculate_serendipity(item, context)
            scored_items.append((item, score))
        
        # Greedy selection with diversity
        selected = []
        remaining = scored_items.copy()
        
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -1.0
            
            for idx, (item, ser_score) in enumerate(remaining):
                # Base score from serendipity
                score = ser_score.overall
                
                # Apply diversity penalty if we have selected items
                if selected and item.features is not None:
                    diversity_bonus = self._calculate_diversity_bonus(
                        item, [s[0] for s in selected]
                    )
                    # Combine serendipity and diversity
                    score = (1 - self.diversity_weight) * score + \
                            self.diversity_weight * diversity_bonus
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return [item for item, _ in selected]
    
    def _calculate_diversity_bonus(
        self,
        item: SerendipityItem,
        selected: List[SerendipityItem]
    ) -> float:
        """
        Calculate diversity bonus (higher = more diverse from selected).
        
        Args:
            item: Candidate item
            selected: Already selected items
            
        Returns:
            Diversity score (0-1)
        """
        if item.features is None:
            return 0.5
        
        # Find minimum similarity to selected items
        min_similarity = 1.0
        for sel_item in selected:
            if sel_item.features is not None:
                sim = self._cosine_similarity(item.features, sel_item.features)
                min_similarity = min(min_similarity, sim)
        
        # Diversity bonus is inverse of minimum similarity
        return 1.0 - min_similarity
