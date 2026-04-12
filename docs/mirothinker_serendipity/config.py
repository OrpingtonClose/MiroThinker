"""
Serendipity Configuration Models.

Pydantic models for validating and managing serendipity configuration.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class AlgorithmConfig:
    """Configuration for a serendipity algorithm."""
    enabled: bool = True
    lambda_param: float = 0.5
    diversity_weight: float = 0.3
    similarity_threshold: float = 0.7
    k: int = 5
    distance_metric: str = "cosine"
    n_clusters: int = 10
    cluster_method: str = "kmeans"


@dataclass
class RerankingConfig:
    """Configuration for reranking."""
    enabled: bool = True
    top_k: int = 10
    apply_to: List[str] = field(default_factory=lambda: [
        "tool_calls", "research_results"
    ])


@dataclass
class BalanceConfig:
    """Configuration for relevance/serendipity balance."""
    relevance_weight: float = 0.7
    unexpectedness_weight: float = 0.3


@dataclass
class AdaptiveWeightsConfig:
    """Configuration for adaptive weights."""
    enabled: bool = True
    strategy: str = "curiosity_driven"
    min_serendipity: float = 0.1
    max_serendipity: float = 0.7
    adaptation_rate: float = 0.1


@dataclass
class CuriosityProfilingConfig:
    """Configuration for curiosity profiling."""
    enabled: bool = True
    update_frequency: str = "per_session"
    persistence: str = "user_profile"


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategy."""
    strategy: str = "epsilon_greedy"
    epsilon: float = 0.2
    temperature: float = 0.5


@dataclass
class SerendipityConfig:
    """Main serendipity configuration."""
    enabled: bool = True
    phase: str = "quick_wins"
    
    algorithm: Dict[str, Any] = field(default_factory=lambda: {
        "name": "sog",
        "fallback": "sog"
    })
    
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    adaptive_weights: AdaptiveWeightsConfig = field(default_factory=AdaptiveWeightsConfig)
    curiosity: CuriosityProfilingConfig = field(default_factory=CuriosityProfilingConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    
    algorithms: Dict[str, AlgorithmConfig] = field(default_factory=lambda: {
        "sog": AlgorithmConfig(),
        "kfn": AlgorithmConfig(),
        "sccf": AlgorithmConfig()
    })
    
    user_control: Dict[str, Any] = field(default_factory=lambda: {
        "opt_in": True,
        "default_curiosity_level": "medium",
        "curiosity_levels": {
            "low": {"serendipity_weight": 0.2, "exploration_rate": 0.1},
            "medium": {"serendipity_weight": 0.4, "exploration_rate": 0.2},
            "high": {"serendipity_weight": 0.6, "exploration_rate": 0.3}
        }
    })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SerendipityConfig':
        """Create config from dictionary."""
        return cls(
            enabled=data.get('enabled', True),
            phase=data.get('phase', 'quick_wins'),
            algorithm=data.get('algorithm', {'name': 'sog'}),
            reranking=RerankingConfig(**data.get('reranking', {})),
            balance=BalanceConfig(**data.get('balance', {})),
            adaptive_weights=AdaptiveWeightsConfig(**data.get('adaptive_weights', {})),
            curiosity=CuriosityProfilingConfig(**data.get('curiosity', {}).get('profiling', {})),
            exploration=ExplorationConfig(**data.get('exploration', {})),
            user_control=data.get('user_control', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'phase': self.phase,
            'algorithm': self.algorithm,
            'reranking': {
                'enabled': self.reranking.enabled,
                'top_k': self.reranking.top_k,
                'apply_to': self.reranking.apply_to
            },
            'balance': {
                'relevance_weight': self.balance.relevance_weight,
                'unexpectedness_weight': self.balance.unexpectedness_weight
            },
            'adaptive_weights': {
                'enabled': self.adaptive_weights.enabled,
                'strategy': self.adaptive_weights.strategy,
                'min_serendipity': self.adaptive_weights.min_serendipity,
                'max_serendipity': self.adaptive_weights.max_serendipity,
                'adaptation_rate': self.adaptive_weights.adaptation_rate
            },
            'curiosity': {
                'profiling': {
                    'enabled': self.curiosity.enabled,
                    'update_frequency': self.curiosity.update_frequency,
                    'persistence': self.curiosity.persistence
                }
            },
            'exploration': {
                'strategy': self.exploration.strategy,
                'epsilon': self.exploration.epsilon,
                'temperature': self.exploration.temperature
            },
            'user_control': self.user_control
        }
