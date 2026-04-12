"""
Serendipity Metrics Models.

Models for tracking and reporting serendipity-related metrics.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SerendipityMetrics:
    """
    Container for serendipity-related metrics.
    
    Attributes:
        timestamp: When metrics were recorded
        total_items_evaluated: Number of items evaluated
        serendipitous_items: Number of items above threshold
        serendipity_rate: Proportion of serendipitous items
        avg_serendipity_score: Average serendipity score
        avg_relevance: Average relevance score
        avg_unexpectedness: Average unexpectedness score
        top_discoveries: List of top serendipitous discoveries
        algorithm_used: Algorithm that produced metrics
    """
    timestamp: datetime = field(default_factory=datetime.now)
    total_items_evaluated: int = 0
    serendipitous_items: int = 0
    serendipity_rate: float = 0.0
    avg_serendipity_score: float = 0.0
    avg_relevance: float = 0.0
    avg_unexpectedness: float = 0.0
    top_discoveries: List[Dict[str, Any]] = field(default_factory=list)
    algorithm_used: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_items_evaluated': self.total_items_evaluated,
            'serendipitous_items': self.serendipitous_items,
            'serendipity_rate': round(self.serendipity_rate, 3),
            'avg_serendipity_score': round(self.avg_serendipity_score, 3),
            'avg_relevance': round(self.avg_relevance, 3),
            'avg_unexpectedness': round(self.avg_unexpectedness, 3),
            'top_discoveries': self.top_discoveries,
            'algorithm_used': self.algorithm_used
        }


class MetricsCollector:
    """Collects and aggregates serendipity metrics over time."""
    
    def __init__(self):
        self.metrics_history: List[SerendipityMetrics] = []
    
    def record_metrics(self, metrics: SerendipityMetrics):
        """Record a metrics snapshot."""
        self.metrics_history.append(metrics)
    
    def get_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Get trend analysis over recent metrics.
        
        Args:
            window_size: Number of recent metrics to analyze
            
        Returns:
            Trend analysis dictionary
        """
        if len(self.metrics_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent = self.metrics_history[-window_size:]
        
        # Calculate trends
        serendipity_scores = [m.avg_serendipity_score for m in recent]
        
        if len(serendipity_scores) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend: compare first half to second half
        mid = len(serendipity_scores) // 2
        first_half = sum(serendipity_scores[:mid]) / mid if mid > 0 else 0
        second_half = sum(serendipity_scores[mid:]) / (len(serendipity_scores) - mid)
        
        if second_half > first_half * 1.1:
            trend = 'improving'
        elif second_half < first_half * 0.9:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_avg': round(second_half, 3),
            'previous_avg': round(first_half, 3),
            'change_pct': round((second_half - first_half) / first_half * 100, 1) if first_half > 0 else 0
        }
