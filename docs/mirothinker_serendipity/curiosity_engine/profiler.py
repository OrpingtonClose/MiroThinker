"""
Curiosity Profiler for User Modeling.

Tracks and updates user curiosity profiles based on interactions,
enabling personalized serendipity experiences.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class CuriosityProfile:
    """
    User curiosity profile with multiple dimensions.
    
    Dimensions:
        - novelty_seeking: Preference for new/unfamiliar items (0-1)
        - diversity_appreciation: Preference for variety (0-1)
        - surprise_tolerance: Comfort with unexpected results (0-1)
        - exploration_depth: Tendency to explore deeply (0-1)
    
    Attributes:
        user_id: Unique user identifier
        novelty_seeking: Novelty seeking score
        diversity_appreciation: Diversity appreciation score
        surprise_tolerance: Surprise tolerance score
        exploration_depth: Exploration depth score
        created_at: Profile creation timestamp
        updated_at: Last update timestamp
        interaction_count: Number of interactions recorded
    """
    user_id: str
    novelty_seeking: float = 0.5
    diversity_appreciation: float = 0.5
    surprise_tolerance: float = 0.5
    exploration_depth: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            'user_id': self.user_id,
            'novelty_seeking': self.novelty_seeking,
            'diversity_appreciation': self.diversity_appreciation,
            'surprise_tolerance': self.surprise_tolerance,
            'exploration_depth': self.exploration_depth,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'interaction_count': self.interaction_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CuriosityProfile':
        """Create profile from dictionary."""
        profile = cls(
            user_id=data['user_id'],
            novelty_seeking=data.get('novelty_seeking', 0.5),
            diversity_appreciation=data.get('diversity_appreciation', 0.5),
            surprise_tolerance=data.get('surprise_tolerance', 0.5),
            exploration_depth=data.get('exploration_depth', 0.5),
            interaction_count=data.get('interaction_count', 0)
        )
        
        # Parse timestamps if present
        if 'created_at' in data:
            profile.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            profile.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return profile
    
    def get_exploration_type(self) -> str:
        """
        Classify user exploration type based on profile.
        
        Returns:
            One of: 'explorer', 'specialist', 'dabbler', 'focused'
        """
        high_novelty = self.novelty_seeking > 0.6
        high_diversity = self.diversity_appreciation > 0.6
        high_depth = self.exploration_depth > 0.6
        
        if high_novelty and high_diversity:
            return 'explorer'
        elif high_depth and not high_diversity:
            return 'specialist'
        elif high_diversity and not high_depth:
            return 'dabbler'
        else:
            return 'focused'


class CuriosityProfiler:
    """
    Profiles user curiosity based on interactions.
    
    Tracks and updates curiosity dimensions based on:
    - Explicit feedback (ratings, preferences)
    - Implicit signals (dwell time, follow-ups, selections)
    - Choice patterns (what users select, skip, explore)
    
    Example:
        >>> profiler = CuriosityProfiler(config)
        >>> profile = profiler.get_or_create_profile('user_123')
        >>> # After interaction
        >>> profiler.update_from_interaction('user_123', interaction_data)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curiosity profiler.
        
        Args:
            config: Configuration with keys:
                - update_frequency: How often to update ('per_task', 'per_session', 'per_query')
                - persistence: Where to persist profiles ('session_only', 'user_profile')
        """
        self.config = config
        self.update_frequency = config.get('update_frequency', 'per_session')
        self.persistence = config.get('persistence', 'session_only')
        self.profiles: Dict[str, CuriosityProfile] = {}
        self._update_counts: Dict[str, int] = {}
    
    def get_or_create_profile(self, user_id: str) -> CuriosityProfile:
        """
        Get existing profile or create new one.
        
        Args:
            user_id: User identifier
            
        Returns:
            CuriosityProfile for the user
        """
        if user_id not in self.profiles:
            # Try to load from persistence
            loaded = self._load_profile(user_id)
            if loaded:
                self.profiles[user_id] = loaded
            else:
                # Create new profile with defaults
                self.profiles[user_id] = CuriosityProfile(user_id=user_id)
        
        return self.profiles[user_id]
    
    def update_from_interaction(
        self,
        user_id: str,
        interaction: Dict[str, Any]
    ) -> CuriosityProfile:
        """
        Update profile based on user interaction.
        
        Args:
            user_id: User identifier
            interaction: Interaction data with optional keys:
                - selected_items: List of items user selected
                - dwell_times: Dict of item_id -> time spent
                - explicit_ratings: List of rating dicts
                - follow_up_actions: List of follow-up actions
                - skipped_items: List of items user skipped
                
        Returns:
            Updated curiosity profile
        """
        profile = self.get_or_create_profile(user_id)
        
        # Update novelty seeking
        if 'selected_items' in interaction:
            profile.novelty_seeking = self._update_novelty_seeking(
                profile.novelty_seeking,
                interaction['selected_items']
            )
        
        # Update diversity appreciation
        if 'selected_items' in interaction:
            profile.diversity_appreciation = self._update_diversity_appreciation(
                profile.diversity_appreciation,
                interaction['selected_items']
            )
        
        # Update surprise tolerance
        if 'explicit_ratings' in interaction:
            profile.surprise_tolerance = self._update_surprise_tolerance(
                profile.surprise_tolerance,
                interaction['explicit_ratings']
            )
        
        # Update exploration depth
        if 'follow_up_actions' in interaction:
            profile.exploration_depth = self._update_exploration_depth(
                profile.exploration_depth,
                interaction['follow_up_actions']
            )
        
        # Update metadata
        profile.interaction_count += 1
        profile.updated_at = datetime.now()
        
        # Persist if needed
        if self.persistence == 'user_profile':
            self._save_profile(user_id, profile)
        
        return profile
    
    def _update_novelty_seeking(
        self,
        current: float,
        selected_items: List[Dict[str, Any]]
    ) -> float:
        """
        Update novelty seeking based on selection of novel items.
        
        Uses exponential moving average with alpha=0.2
        
        Args:
            current: Current novelty seeking score
            selected_items: Items user selected
            
        Returns:
            Updated novelty seeking score
        """
        if not selected_items:
            return current
        
        # Calculate proportion of novel items selected
        novel_count = sum(1 for item in selected_items if item.get('is_novel', False))
        novel_ratio = novel_count / len(selected_items)
        
        # Exponential moving average update
        alpha = 0.2
        return (1 - alpha) * current + alpha * novel_ratio
    
    def _update_diversity_appreciation(
        self,
        current: float,
        selected_items: List[Dict[str, Any]]
    ) -> float:
        """
        Update diversity appreciation based on category spread.
        
        Uses exponential moving average with alpha=0.2
        
        Args:
            current: Current diversity appreciation score
            selected_items: Items user selected
            
        Returns:
            Updated diversity appreciation score
        """
        if len(selected_items) < 2:
            return current
        
        # Calculate category diversity
        categories = [
            item.get('category')
            for item in selected_items
            if item.get('category')
        ]
        
        if not categories:
            return current
        
        unique_categories = len(set(categories))
        diversity_ratio = unique_categories / len(categories)
        
        # Update with EMA
        alpha = 0.2
        return (1 - alpha) * current + alpha * diversity_ratio
    
    def _update_surprise_tolerance(
        self,
        current: float,
        ratings: List[Dict[str, Any]]
    ) -> float:
        """
        Update surprise tolerance from ratings of surprising items.
        
        Args:
            current: Current surprise tolerance score
            ratings: User ratings with 'was_surprising' and 'rating' keys
            
        Returns:
            Updated surprise tolerance score
        """
        if not ratings:
            return current
        
        # Look for high ratings on surprising items
        surprise_ratings = [
            r['rating'] for r in ratings
            if r.get('was_surprising') and r.get('rating') is not None
        ]
        
        if not surprise_ratings:
            return current
        
        avg_surprise_rating = np.mean(surprise_ratings)
        
        # Update with EMA
        alpha = 0.15
        return (1 - alpha) * current + alpha * avg_surprise_rating
    
    def _update_exploration_depth(
        self,
        current: float,
        follow_ups: List[Dict[str, Any]]
    ) -> float:
        """
        Update exploration depth from follow-up behavior.
        
        Args:
            current: Current exploration depth score
            follow_ups: Follow-up actions with 'depth' key
            
        Returns:
            Updated exploration depth score
        """
        if not follow_ups:
            return current
        
        # Calculate average exploration depth
        depths = [f.get('depth', 0) for f in follow_ups]
        avg_depth = np.mean(depths) if depths else 0
        
        # Normalize to 0-1 (assuming max depth of 5)
        normalized_depth = min(1.0, avg_depth / 5.0)
        
        # Update with EMA
        alpha = 0.15
        return (1 - alpha) * current + alpha * normalized_depth
    
    def _load_profile(self, user_id: str) -> Optional[CuriosityProfile]:
        """Load profile from persistence."""
        # Implementation depends on persistence mechanism
        # This is a placeholder
        return None
    
    def _save_profile(self, user_id: str, profile: CuriosityProfile):
        """Save profile to persistence."""
        # Implementation depends on persistence mechanism
        pass
    
    def get_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user's curiosity profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Profile summary dictionary
        """
        profile = self.get_or_create_profile(user_id)
        
        return {
            'user_id': user_id,
            'exploration_type': profile.get_exploration_type(),
            'dimensions': {
                'novelty_seeking': round(profile.novelty_seeking, 2),
                'diversity_appreciation': round(profile.diversity_appreciation, 2),
                'surprise_tolerance': round(profile.surprise_tolerance, 2),
                'exploration_depth': round(profile.exploration_depth, 2),
            },
            'interaction_count': profile.interaction_count,
            'updated_at': profile.updated_at.isoformat()
        }
