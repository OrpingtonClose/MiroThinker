"""
Adaptive Weight Controller for Serendipity.

Dynamically adjusts serendipity/relevance balance based on:
- User curiosity profile
- Task context
- Historical performance
- Explicit user feedback
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class AdaptiveState:
    """
    Current adaptive state.
    
    Tracks the current weights and performance history
    for adaptive adjustment.
    """
    current_relevance_weight: float = 0.7
    current_serendipity_weight: float = 0.3
    exploration_rate: float = 0.2
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    
    def record_performance(self, metrics: Dict[str, float]):
        """Record performance metrics for future adaptation."""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        # Keep only last 100 records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class AdaptiveWeightController:
    """
    Dynamically adjusts serendipity/relevance balance.
    
    This controller uses multiple signals to determine the optimal
    balance between serendipity and relevance for each context.
    
    Signals used:
    1. User curiosity profile (novelty seeking, diversity appreciation)
    2. Task context (urgency, type, progress)
    3. Historical performance (satisfaction, completion rates)
    4. User feedback (explicit ratings, implicit signals)
    
    Example:
        >>> controller = AdaptiveWeightController(config)
        >>> weights = controller.adapt_weights(curiosity_profile, task_context, feedback)
        >>> print(weights)  # {'relevance_weight': 0.6, 'serendipity_weight': 0.4}
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adaptive controller.
        
        Args:
            config: Configuration with keys:
                - min_serendipity: Minimum serendipity weight
                - max_serendipity: Maximum serendipity weight
                - adaptation_rate: How quickly to adapt (0-1)
                - strategy: Adaptation strategy
        """
        self.config = config
        self.min_serendipity = config.get('min_serendipity', 0.1)
        self.max_serendipity = config.get('max_serendipity', 0.7)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.strategy = config.get('strategy', 'curiosity_driven')
        self.state = AdaptiveState()
    
    def adapt_weights(
        self,
        curiosity_profile: Dict[str, float],
        task_context: Dict[str, Any],
        recent_feedback: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Adapt weights based on multiple factors.
        
        Args:
            curiosity_profile: User's curiosity dimensions
            task_context: Current task information
            recent_feedback: Recent user feedback
            
        Returns:
            Updated weights dict with 'relevance_weight' and 'serendipity_weight'
        """
        # Calculate base serendipity from curiosity profile
        base_serendipity = self._curiosity_to_serendipity(curiosity_profile)
        
        # Calculate adjustments from different signals
        context_adjustment = self._context_adjustment(task_context)
        feedback_adjustment = self._feedback_adjustment(recent_feedback)
        performance_adjustment = self._performance_adjustment()
        
        # Combine adjustments based on strategy
        if self.strategy == 'curiosity_driven':
            new_serendipity = base_serendipity + 0.3 * context_adjustment + 0.2 * feedback_adjustment
        elif self.strategy == 'performance_driven':
            new_serendipity = 0.4 + 0.6 * performance_adjustment
        else:  # hybrid
            new_serendipity = (
                0.5 * base_serendipity +
                0.2 * context_adjustment +
                0.2 * feedback_adjustment +
                0.1 * performance_adjustment
            )
        
        # Clamp to valid range
        new_serendipity = max(
            self.min_serendipity,
            min(self.max_serendipity, new_serendipity)
        )
        
        # Apply adaptation rate (smooth transitions)
        current = self.state.current_serendipity_weight
        smoothed = current + self.adaptation_rate * (new_serendipity - current)
        
        # Update state
        self.state.current_serendipity_weight = smoothed
        self.state.current_relevance_weight = 1.0 - smoothed
        self.state.last_update = datetime.now()
        
        return {
            'relevance_weight': self.state.current_relevance_weight,
            'serendipity_weight': self.state.current_serendipity_weight
        }
    
    def _curiosity_to_serendipity(
        self,
        profile: Dict[str, float]
    ) -> float:
        """
        Convert curiosity profile to serendipity weight.
        
        Uses weighted combination of curiosity dimensions:
        - novelty_seeking: 35%
        - diversity_appreciation: 25%
        - surprise_tolerance: 25%
        - exploration_depth: 15%
        
        Args:
            profile: Curiosity profile with dimension scores
            
        Returns:
            Base serendipity weight (0-1)
        """
        weights = {
            'novelty_seeking': 0.35,
            'diversity_appreciation': 0.25,
            'surprise_tolerance': 0.25,
            'exploration_depth': 0.15
        }
        
        serendipity = sum(
            profile.get(dim, 0.5) * weight
            for dim, weight in weights.items()
        )
        
        return serendipity
    
    def _context_adjustment(self, context: Dict[str, Any]) -> float:
        """
        Calculate adjustment based on task context.
        
        Adjustments:
        - High urgency: -0.2 (reduce serendipity)
        - Exploration task: +0.15 (increase serendipity)
        - Early progress: +0.1 (more exploration)
        - Late progress: -0.1 (more focus)
        
        Args:
            context: Task context dictionary
            
        Returns:
            Context adjustment value
        """
        adjustment = 0.0
        
        # Reduce serendipity for time-critical tasks
        if context.get('urgency') == 'high':
            adjustment -= 0.2
        elif context.get('urgency') == 'low':
            adjustment += 0.1
        
        # Increase serendipity for exploratory tasks
        task_type = context.get('task_type', 'general')
        if task_type == 'exploration':
            adjustment += 0.15
        elif task_type == 'focused_research':
            adjustment -= 0.1
        
        # Adjust based on task progress
        progress = context.get('progress', 0.5)
        if progress < 0.3:
            adjustment += 0.1  # Early stage: more exploration
        elif progress > 0.8:
            adjustment -= 0.1  # Late stage: more focus
        
        # Adjust based on complexity
        complexity = context.get('complexity', 'medium')
        if complexity == 'high':
            adjustment -= 0.05  # Reduce slightly for complex tasks
        
        return adjustment
    
    def _feedback_adjustment(
        self,
        feedback: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate adjustment based on user feedback.
        
        Positive signals:
        - High satisfaction + appreciated discovery: +0.1
        
        Negative signals:
        - Low satisfaction: -0.15
        - High skip rate: -0.1
        
        Args:
            feedback: List of feedback entries
            
        Returns:
            Feedback adjustment value
        """
        if not feedback:
            return 0.0
        
        adjustment = 0.0
        
        # Calculate average satisfaction
        satisfactions = [f.get('satisfaction', 0.5) for f in feedback]
        avg_satisfaction = np.mean(satisfactions)
        
        # Calculate discovery appreciation rate
        discoveries = [f.get('appreciated_discovery', False) for f in feedback]
        discovery_rate = sum(discoveries) / len(discoveries) if discoveries else 0
        
        # Calculate skip rate
        skips = [f.get('skipped', False) for f in feedback]
        skip_rate = sum(skips) / len(skips) if skips else 0
        
        # Apply adjustments
        if avg_satisfaction > 0.7 and discovery_rate > 0.5:
            adjustment += 0.1
        elif avg_satisfaction < 0.4:
            adjustment -= 0.15
        
        if skip_rate > 0.5:
            adjustment -= 0.1
        
        return adjustment
    
    def _performance_adjustment(self) -> float:
        """
        Calculate adjustment based on historical performance.
        
        Looks at recent performance trends and adjusts accordingly.
        
        Returns:
            Performance adjustment value
        """
        if len(self.state.performance_history) < 5:
            return 0.0
        
        # Get recent performance
        recent = self.state.performance_history[-10:]
        
        # Calculate trend in task completion
        completions = [p.get('task_completed', 0) for p in recent]
        avg_completion = np.mean(completions)
        
        # Calculate trend in user satisfaction
        satisfactions = [p.get('user_satisfaction', 0.5) for p in recent]
        avg_satisfaction = np.mean(satisfactions)
        
        # Adjust based on performance
        if avg_completion > 0.8 and avg_satisfaction > 0.7:
            return 0.05  # Slight increase if performing well
        elif avg_completion < 0.5 or avg_satisfaction < 0.4:
            return -0.1  # Decrease if performing poorly
        
        return 0.0
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current adaptive weights.
        
        Returns:
            Dictionary with 'relevance_weight' and 'serendipity_weight'
        """
        return {
            'relevance_weight': self.state.current_relevance_weight,
            'serendipity_weight': self.state.current_serendipity_weight
        }
    
    def record_performance(self, metrics: Dict[str, float]):
        """
        Record performance metrics for future adaptation.
        
        Args:
            metrics: Performance metrics dictionary
        """
        self.state.record_performance(metrics)
