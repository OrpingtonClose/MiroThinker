"""
Discovery Event Models.

Models for tracking and analyzing serendipitous discoveries.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DiscoveryType(Enum):
    """Types of serendipitous discoveries."""
    UNEXPECTED_CONNECTION = "unexpected_connection"
    NOVEL_INSIGHT = "novel_insight"
    CROSS_DOMAIN = "cross_domain"
    SURPRISING_RESULT = "surprising_result"
    HIDDEN_PATTERN = "hidden_pattern"
    ALTERNATIVE_PATH = "alternative_path"


@dataclass
class DiscoveryEvent:
    """
    Represents a serendipitous discovery event.
    
    Attributes:
        id: Unique event identifier
        user_id: User who made the discovery
        session_id: Session in which discovery occurred
        discovery_type: Type of discovery
        source_item: Item that led to discovery
        discovered_item: The serendipitous item discovered
        serendipity_score: Calculated serendipity score
        context: Context in which discovery occurred
        timestamp: When discovery occurred
        user_reaction: User's reaction to discovery
        followed_up: Whether user followed up on discovery
    """
    id: str
    user_id: str
    session_id: str
    discovery_type: DiscoveryType
    source_item: Dict[str, Any]
    discovered_item: Dict[str, Any]
    serendipity_score: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    user_reaction: Optional[str] = None
    followed_up: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'discovery_type': self.discovery_type.value,
            'source_item': self.source_item,
            'discovered_item': self.discovered_item,
            'serendipity_score': self.serendipity_score,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'user_reaction': self.user_reaction,
            'followed_up': self.followed_up
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveryEvent':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            session_id=data['session_id'],
            discovery_type=DiscoveryType(data['discovery_type']),
            source_item=data['source_item'],
            discovered_item=data['discovered_item'],
            serendipity_score=data['serendipity_score'],
            context=data.get('context', {}),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_reaction=data.get('user_reaction'),
            followed_up=data.get('followed_up', False)
        )
    
    def record_reaction(self, reaction: str, followed_up: bool = False):
        """
        Record user reaction to discovery.
        
        Args:
            reaction: User's reaction ('positive', 'negative', 'neutral')
            followed_up: Whether user followed up on discovery
        """
        self.user_reaction = reaction
        self.followed_up = followed_up


class DiscoveryTracker:
    """Tracks and analyzes discovery events."""
    
    def __init__(self):
        self.discoveries: List[DiscoveryEvent] = []
    
    def record_discovery(self, event: DiscoveryEvent):
        """Record a new discovery event."""
        self.discoveries.append(event)
    
    def get_user_discoveries(
        self,
        user_id: str,
        discovery_type: Optional[DiscoveryType] = None
    ) -> List[DiscoveryEvent]:
        """
        Get discoveries for a user.
        
        Args:
            user_id: User identifier
            discovery_type: Optional type filter
            
        Returns:
            List of matching discovery events
        """
        matches = [d for d in self.discoveries if d.user_id == user_id]
        
        if discovery_type:
            matches = [d for d in matches if d.discovery_type == discovery_type]
        
        return matches
    
    def get_discovery_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get discovery statistics.
        
        Args:
            user_id: Optional user filter
            
        Returns:
            Statistics dictionary
        """
        discoveries = self.discoveries
        if user_id:
            discoveries = [d for d in discoveries if d.user_id == user_id]
        
        if not discoveries:
            return {
                'total_discoveries': 0,
                'by_type': {},
                'avg_serendipity_score': 0,
                'follow_up_rate': 0
            }
        
        # Count by type
        by_type = {}
        for d in discoveries:
            type_name = d.discovery_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        # Calculate average serendipity score
        avg_score = sum(d.serendipity_score for d in discoveries) / len(discoveries)
        
        # Calculate follow-up rate
        follow_ups = sum(1 for d in discoveries if d.followed_up)
        follow_up_rate = follow_ups / len(discoveries)
        
        return {
            'total_discoveries': len(discoveries),
            'by_type': by_type,
            'avg_serendipity_score': round(avg_score, 3),
            'follow_up_rate': round(follow_up_rate, 3)
        }
