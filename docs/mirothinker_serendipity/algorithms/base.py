"""
Base class and data models for serendipity algorithms.

All serendipity algorithms must inherit from BaseSerendipityAlgorithm
and implement the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SerendipityItem:
    """
    Item that can be evaluated for serendipity.
    
    Attributes:
        id: Unique identifier for the item
        content: The actual content (can be any type)
        relevance_score: Pre-computed relevance score (0-1)
        features: Optional feature vector for similarity calculations
        metadata: Optional additional metadata
    """
    id: str
    content: Any
    relevance_score: float
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SerendipityScore:
    """
    Serendipity score breakdown.
    
    Attributes:
        overall: Overall serendipity score (0-1)
        relevance: Relevance component (0-1)
        unexpectedness: Unexpectedness component (0-1)
        usefulness: Optional usefulness score (0-1)
    """
    overall: float
    relevance: float
    unexpectedness: float
    usefulness: Optional[float] = None
    
    @property
    def is_serendipitous(self, threshold: float = 0.5) -> bool:
        """
        Check if this score qualifies as serendipitous.
        
        Args:
            threshold: Minimum score to be considered serendipitous
            
        Returns:
            True if overall score >= threshold
        """
        return self.overall >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'overall': self.overall,
            'relevance': self.relevance,
            'unexpectedness': self.unexpectedness,
            'usefulness': self.usefulness,
            'is_serendipitous': self.is_serendipitous
        }


class BaseSerendipityAlgorithm(ABC):
    """
    Abstract base class for all serendipity algorithms.
    
    All serendipity algorithms must implement:
    - calculate_serendipity: Calculate score for a single item
    - rerank: Re-rank a list of items by serendipity
    
    The core serendipity formula is:
        Serendipity = Relevance × Unexpectedness
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the algorithm with configuration.
        
        Args:
            config: Algorithm-specific configuration dictionary
        """
        self.config = config
        self.lambda_param = config.get('lambda_param', 0.5)
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_serendipity(
        self,
        item: SerendipityItem,
        context: Dict[str, Any]
    ) -> SerendipityScore:
        """
        Calculate serendipity score for an item.
        
        Args:
            item: Item to evaluate
            context: Context including user profile, history, query, etc.
            
        Returns:
            SerendipityScore with breakdown of components
        """
        pass
    
    @abstractmethod
    def rerank(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any],
        top_k: int = 10
    ) -> List[SerendipityItem]:
        """
        Re-rank items by serendipity score.
        
        Args:
            items: Items to rerank
            context: Context information
            top_k: Number of items to return
            
        Returns:
            Re-ranked list of items
        """
        pass
    
    def compute_serendipity_formula(
        self,
        relevance: float,
        unexpectedness: float
    ) -> float:
        """
        Core serendipity formula: Serendipity = Relevance × Unexpectedness
        
        This is the fundamental formula that all algorithms should use
        as the basis for their serendipity calculations.
        
        Args:
            relevance: Relevance score (0-1)
            unexpectedness: Unexpectedness score (0-1)
            
        Returns:
            Serendipity score (0-1)
        """
        return relevance * unexpectedness
    
    def batch_calculate(
        self,
        items: List[SerendipityItem],
        context: Dict[str, Any]
    ) -> Dict[str, SerendipityScore]:
        """
        Calculate serendipity scores for multiple items.
        
        Args:
            items: List of items to evaluate
            context: Context information
            
        Returns:
            Dictionary mapping item IDs to their scores
        """
        return {
            item.id: self.calculate_serendipity(item, context)
            for item in items
        }
