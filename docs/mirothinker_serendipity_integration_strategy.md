# MiroThinker Serendipity Integration Strategy

## Executive Summary

This document outlines a phased integration strategy for introducing serendipity algorithms into the MiroThinker deep research agent framework. The strategy prioritizes quick wins while building toward a comprehensive serendipity-aware system.

---

## 1. Architecture Overview

### 1.1 Current MiroThinker Architecture
```
mirothinker/
├── config/          # YAML configuration management
├── core/            # Agent orchestration, task management
├── io/              # Input/output handling
├── llm/             # LLM integration and prompting
├── logging/         # Logging infrastructure
├── utils/           # Utility functions
└── mcp/             # MCP tool server integrations
```

### 1.2 Proposed Serendipity Module Structure
```
mirothinker/
├── config/
├── core/
├── io/
├── llm/
├── logging/
├── utils/
├── mcp/
└── serendipity/              # NEW MODULE
    ├── __init__.py
    ├── config.py             # Serendipity configuration models
    ├── algorithms/           # Algorithm implementations
    │   ├── __init__.py
    │   ├── base.py           # Base algorithm interface
    │   ├── sog.py            # SOG implementation
    │   ├── kfn.py            # k-Furthest Neighbors
    │   ├── sccf.py           # SC-CF clustering
    │   ├── sirup.py          # SIRUP content-based
    │   ├── tangent.py        # TANGENT graph-based
    │   └── serencdr.py       # SerenCDR deep learning
    ├── models/               # Data models
    │   ├── __init__.py
    │   ├── discovery.py      # Discovery event models
    │   ├── curiosity.py      # Curiosity profile models
    │   └── metrics.py        # Serendipity metrics
    ├── curiosity_engine/     # Curiosity modeling
    │   ├── __init__.py
    │   ├── profiler.py       # User curiosity profiling
    │   ├── explorer.py       # Exploration controller
    │   └── decay.py          # Interest decay models
    ├── reranker/             # Re-ranking components
    │   ├── __init__.py
    │   ├── base.py           # Base reranker interface
    │   ├── hybrid.py         # Hybrid serendipity reranker
    │   └── adaptive.py       # Adaptive weight controller
    ├── metrics/              # Serendipity measurement
    │   ├── __init__.py
    │   ├── calculator.py     # Serendipity score calculator
    │   ├── unexpectedness.py # Unexpectedness measures
    │   └── relevance.py      # Relevance assessment
    └── integration/          # Framework integration
        ├── __init__.py
        ├── tool_filter.py    # Tool call filtering
        ├── result_enhancer.py # Result enhancement
        └── pipeline.py       # Full serendipity pipeline
```

---

## 2. Phased Integration Roadmap

### Phase 1: Quick Wins (Weeks 1-2)
**Goal**: Minimal changes, immediate impact, proof of concept

#### 2.1.1 What to Implement

| Component | Description | Effort |
|-----------|-------------|--------|
| SOG Algorithm | Serendipity-Oriented Greedy re-ranking | Low |
| Basic Reranker | Simple wrapper for existing results | Low |
| Config Toggle | Enable/disable serendipity | Low |
| Static Weights | Fixed serendipity/relevance balance | Low |

#### 2.1.2 Files/Modules to Create/Modify

**New Files:**
```
mirothinker/serendipity/__init__.py
mirothinker/serendipity/config.py
mirothinker/serendipity/algorithms/__init__.py
mirothinker/serendipity/algorithms/base.py
mirothinker/serendipity/algorithms/sog.py
mirothinker/serendipity/reranker/__init__.py
mirothinker/serendipity/reranker/base.py
mirothinker/serendipity/reranker/hybrid.py
mirothinker/serendipity/models/discovery.py
```

**Modified Files:**
```
mirothinker/config/settings.py          # Add serendipity config
mirothinker/core/agent.py               # Integrate reranker
mirothinker/llm/prompts.py              # Add serendipity instructions
```

#### 2.1.3 Configuration Options

```yaml
# config/serendipity.yaml
serendipity:
  enabled: true
  phase: "quick_wins"  # quick_wins | core | advanced
  
  algorithm:
    name: "sog"  # sog | kfn | sccf | sirup | tangent | serencdr
    params:
      lambda_param: 0.5  # Serendipity weight (0-1)
      
  reranking:
    enabled: true
    top_k: 10
    apply_to: ["tool_calls", "research_results"]
    
  balance:
    relevance_weight: 0.7
    unexpectedness_weight: 0.3
    
  user_control:
    opt_in: true
    curiosity_level: "medium"  # low | medium | high
```

#### 2.1.4 Testing Approach

```python
# tests/serendipity/test_phase1.py
class TestPhase1QuickWins:
    """Test Phase 1 serendipity integration"""
    
    def test_sog_reranking(self):
        """Verify SOG algorithm produces valid rankings"""
        pass
    
    def test_config_toggle(self):
        """Verify enable/disable works correctly"""
        pass
    
    def test_weight_balance(self):
        """Test relevance/unexpectedness balance"""
        pass
    
    def test_integration_agent(self):
        """Test agent integration without breaking existing flow"""
        pass
```

---

### Phase 2: Core Integration (Weeks 3-6)
**Goal**: Deeper architectural changes, curiosity modeling, adaptive weights

#### 2.2.1 What to Implement

| Component | Description | Effort |
|-----------|-------------|--------|
| Curiosity Profiler | Track and model user curiosity | Medium |
| k-Furthest Neighbors | CF-based serendipity | Medium |
| Adaptive Weights | Dynamic balance adjustment | Medium |
| SC-CF | Clustering-based approach | Medium |
| Metrics Dashboard | Track serendipity scores | Medium |

#### 2.2.2 Files/Modules to Create/Modify

**New Files:**
```
mirothinker/serendipity/curiosity_engine/__init__.py
mirothinker/serendipity/curiosity_engine/profiler.py
mirothinker/serendipity/curiosity_engine/explorer.py
mirothinker/serendipity/curiosity_engine/decay.py
mirothinker/serendipity/algorithms/kfn.py
mirothinker/serendipity/algorithms/sccf.py
mirothinker/serendipity/reranker/adaptive.py
mirothinker/serendipity/metrics/__init__.py
mirothinker/serendipity/metrics/calculator.py
mirothinker/serendipity/metrics/unexpectedness.py
mirothinker/serendipity/metrics/relevance.py
```

**Modified Files:**
```
mirothinker/core/agent.py               # Deep curiosity integration
mirothinker/core/task_manager.py        # Task-level serendipity
mirothinker/io/session_store.py         # Store curiosity profiles
mirothinker/llm/context_manager.py      # Context-aware serendipity
```

#### 2.2.3 Configuration Options

```yaml
# config/serendipity.yaml
serendipity:
  enabled: true
  phase: "core"
  
  algorithm:
    name: "adaptive"  # Uses multiple algorithms based on context
    fallback: "sog"
    
  curiosity:
    profiling:
      enabled: true
      update_frequency: "per_session"  # per_task | per_session | per_query
      persistence: "user_profile"      # session_only | user_profile
      
    dimensions:
      - novelty_seeking
      - diversity_appreciation
      - surprise_tolerance
      - exploration_depth
      
    initial_profile:
      novelty_seeking: 0.5
      diversity_appreciation: 0.5
      surprise_tolerance: 0.5
      exploration_depth: 0.5
      
  adaptive_weights:
    enabled: true
    strategy: "curiosity_driven"  # curiosity_driven | performance_driven | hybrid
    min_serendipity: 0.1
    max_serendipity: 0.7
    
  exploration:
    strategy: "epsilon_greedy"  # epsilon_greedy | ucb | thompson_sampling
    epsilon: 0.2
    temperature: 0.5
    
  algorithms:
    sog:
      enabled: true
      lambda_param: 0.5
    kfn:
      enabled: true
      k: 5
      distance_metric: "cosine"
    sccf:
      enabled: true
      n_clusters: 10
      cluster_method: "kmeans"
```

#### 2.2.4 Testing Approach

```python
# tests/serendipity/test_phase2.py
class TestPhase2CoreIntegration:
    """Test Phase 2 core integration"""
    
    def test_curiosity_profiler(self):
        """Test curiosity profile creation and updates"""
        pass
    
    def test_adaptive_weights(self):
        """Test dynamic weight adjustment"""
        pass
    
    def test_kfn_algorithm(self):
        """Test k-Furthest Neighbors implementation"""
        pass
    
    def test_sc_cf_algorithm(self):
        """Test SC-CF clustering"""
        pass
    
    def test_metrics_calculation(self):
        """Test serendipity metrics"""
        pass
```

---

### Phase 3: Advanced Features (Weeks 7-10)
**Goal**: Full serendipity optimization, cross-domain, deep learning

#### 2.3.1 What to Implement

| Component | Description | Effort |
|-----------|-------------|--------|
| SIRUP | Content-based with curiosity | High |
| TANGENT | Graph-based recommendations | High |
| SerenCDR | Deep learning cross-domain | High |
| Full Pipeline | End-to-end serendipity pipeline | High |
| A/B Testing | Built-in experimentation | Medium |
| Feedback Loop | User feedback integration | Medium |

#### 2.3.2 Files/Modules to Create/Modify

**New Files:**
```
mirothinker/serendipity/algorithms/sirup.py
mirothinker/serendipity/algorithms/tangent.py
mirothinker/serendipity/algorithms/serencdr.py
mirothinker/serendipity/integration/pipeline.py
mirothinker/serendipity/integration/tool_filter.py
mirothinker/serendipity/integration/result_enhancer.py
mirothinker/serendipity/experimentation/__init__.py
mirothinker/serendipity/experimentation/ab_test.py
mirothinker/serendipity/experimentation/feedback.py
```

**Modified Files:**
```
mirothinker/core/agent.py               # Full pipeline integration
mirothinker/mcp/tool_registry.py        # Tool serendipity scoring
mirothinker/llm/prompt_builder.py       # Dynamic prompt enhancement
```

#### 2.3.3 Configuration Options

```yaml
# config/serendipity.yaml
serendipity:
  enabled: true
  phase: "advanced"
  
  pipeline:
    enabled: true
    stages:
      - tool_selection
      - result_reranking
      - context_enhancement
      - follow_up_generation
      
  algorithms:
    sirup:
      enabled: true
      curiosity_model: "multi_dim"
      content_features: ["semantic", "structural", "temporal"]
    tangent:
      enabled: true
      graph_depth: 3
      relationship_types: ["similarity", "complementarity", "contrast"]
    serencdr:
      enabled: true
      model_path: "models/serencdr_v1.pt"
      domains: ["research", "code", "documentation"]
      
  experimentation:
    ab_testing:
      enabled: true
      variants:
        - name: "control"
          serendipity_level: 0.0
        - name: "low"
          serendipity_level: 0.3
        - name: "medium"
          serendipity_level: 0.5
        - name: "high"
          serendipity_level: 0.7
      metrics: ["task_completion", "user_satisfaction", "discovery_rate"]
      
  feedback:
    enabled: true
    types:
      - explicit_rating
      - implicit_dwell_time
      - follow_up_actions
    update_frequency: "immediate"
```

#### 2.3.4 Testing Approach

```python
# tests/serendipity/test_phase3.py
class TestPhase3Advanced:
    """Test Phase 3 advanced features"""
    
    def test_sirup_algorithm(self):
        """Test SIRUP content-based serendipity"""
        pass
    
    def test_tangent_algorithm(self):
        """Test TANGENT graph-based recommendations"""
        pass
    
    def test_full_pipeline(self):
        """Test end-to-end serendipity pipeline"""
        pass
    
    def test_ab_testing(self):
        """Test A/B testing framework"""
        pass
    
    def test_feedback_loop(self):
        """Test user feedback integration"""
        pass
```

---

## 3. Serendipity Module Implementation

### 3.1 Base Algorithm Interface

```python
# mirothinker/serendipity/algorithms/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SerendipityItem:
    """Item that can be evaluated for serendipity"""
    id: str
    content: Any
    relevance_score: float
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SerendipityScore:
    """Serendipity score breakdown"""
    overall: float
    relevance: float
    unexpectedness: float
    usefulness: Optional[float] = None
    
    @property
    def is_serendipitous(self, threshold: float = 0.5) -> bool:
        return self.overall >= threshold


class BaseSerendipityAlgorithm(ABC):
    """Base class for all serendipity algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
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
            context: Context including user profile, history, etc.
            
        Returns:
            SerendipityScore with breakdown
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
        
        Args:
            relevance: Relevance score (0-1)
            unexpectedness: Unexpectedness score (0-1)
            
        Returns:
            Serendipity score (0-1)
        """
        return relevance * unexpectedness
```

### 3.2 SOG (Serendipity-Oriented Greedy) Implementation

```python
# mirothinker/serendipity/algorithms/sog.py
from typing import List, Dict, Any
import numpy as np
from .base import BaseSerendipityAlgorithm, SerendipityItem, SerendipityScore


class SOGAlgorithm(BaseSerendipityAlgorithm):
    """
    Serendipity-Oriented Greedy algorithm.
    
    Re-ranks items by greedily selecting items that maximize
    the serendipity formula while maintaining diversity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.diversity_weight = config.get('diversity_weight', 0.3)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
    
    def calculate_serendipity(
        self,
        item: SerendipityItem,
        context: Dict[str, Any]
    ) -> SerendipityScore:
        """Calculate serendipity using SOG approach"""
        # Get user profile from context
        user_profile = context.get('user_profile', {})
        history = context.get('history', [])
        
        # Calculate unexpectedness based on deviation from user history
        unexpectedness = self._calculate_unexpectedness(item, history)
        
        # Calculate relevance (may be pre-computed)
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
        """Calculate unexpectedness as distance from historical items"""
        if not history or item.features is None:
            return 0.5  # Neutral if no history or features
        
        # Calculate average similarity to history
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
        """Compute cosine similarity between two vectors"""
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
        """Greedy re-ranking for serendipity"""
        if not items:
            return []
        
        # Calculate serendipity scores for all items
        scored_items = []
        for item in items:
            score = self.calculate_serendipity(item, context)
            scored_items.append((item, score))
        
        # Greedy selection with diversity
        selected = []
        remaining = scored_items.copy()
        
        while len(selected) < top_k and remaining:
            # Score each remaining item
            best_idx = 0
            best_score = -1
            
            for idx, (item, ser_score) in enumerate(remaining):
                # Base score from serendipity
                score = ser_score.overall
                
                # Penalize similarity to already selected items
                if selected and item.features is not None:
                    diversity_penalty = self._calculate_diversity_penalty(
                        item, [s[0] for s in selected]
                    )
                    score = (1 - self.diversity_weight) * score + \
                            self.diversity_weight * diversity_penalty
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return [item for item, _ in selected]
    
    def _calculate_diversity_penalty(
        self,
        item: SerendipityItem,
        selected: List[SerendipityItem]
    ) -> float:
        """Calculate diversity penalty (higher = more diverse)"""
        if item.features is None:
            return 0.5
        
        min_similarity = 1.0
        for sel_item in selected:
            if sel_item.features is not None:
                sim = self._cosine_similarity(item.features, sel_item.features)
                min_similarity = min(min_similarity, sim)
        
        # Return diversity score (1 - similarity)
        return 1.0 - min_similarity
```

### 3.3 Hybrid Reranker

```python
# mirothinker/serendipity/reranker/hybrid.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..algorithms.base import SerendipityItem, SerendipityScore
from ..algorithms.sog import SOGAlgorithm


@dataclass
class RerankerConfig:
    """Configuration for hybrid reranker"""
    relevance_weight: float = 0.7
    serendipity_weight: float = 0.3
    diversity_weight: float = 0.0
    algorithm_name: str = "sog"
    top_k: int = 10


class HybridSerendipityReranker:
    """
    Hybrid reranker that balances relevance and serendipity.
    
    Combines multiple algorithms and strategies to provide
    configurable balance between accuracy and discovery.
    """
    
    def __init__(self, config: Dict[str, Any]):
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
        """Load the specified serendipity algorithm"""
        algorithms = {
            'sog': SOGAlgorithm,
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
            top_k: Number of items to return
            
        Returns:
            Re-ranked items
        """
        if not items:
            return []
        
        k = top_k or self.config.top_k
        
        # Calculate hybrid scores
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
        
        # Sort by hybrid score
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
        """Get serendipity scores for all items"""
        scores = {}
        for item in items:
            scores[item.id] = self.algorithm.calculate_serendipity(item, context)
        return scores
```

### 3.4 Adaptive Weight Controller

```python
# mirothinker/serendipity/reranker/adaptive.py
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np

from .hybrid import HybridSerendipityReranker, RerankerConfig


@dataclass
class AdaptiveState:
    """Current adaptive state"""
    current_relevance_weight: float = 0.7
    current_serendipity_weight: float = 0.3
    exploration_rate: float = 0.2
    performance_history: List[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


class AdaptiveWeightController:
    """
    Dynamically adjusts serendipity/relevance balance based on:
    - User curiosity profile
    - Task context
    - Historical performance
    - Explicit user feedback
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_serendipity = config.get('min_serendipity', 0.1)
        self.max_serendipity = config.get('max_serendipity', 0.7)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
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
            Updated weights dict
        """
        # Base weight from curiosity profile
        base_serendipity = self._curiosity_to_serendipity(curiosity_profile)
        
        # Adjust based on task context
        context_adjustment = self._context_adjustment(task_context)
        
        # Adjust based on feedback
        feedback_adjustment = self._feedback_adjustment(recent_feedback)
        
        # Combine adjustments
        new_serendipity = base_serendipity + context_adjustment + feedback_adjustment
        
        # Clamp to valid range
        new_serendipity = max(
            self.min_serendipity,
            min(self.max_serendipity, new_serendipity)
        )
        
        # Update state
        self.state.current_serendipity_weight = new_serendipity
        self.state.current_relevance_weight = 1.0 - new_serendipity
        
        return {
            'relevance_weight': self.state.current_relevance_weight,
            'serendipity_weight': self.state.current_serendipity_weight
        }
    
    def _curiosity_to_serendipity(
        self,
        profile: Dict[str, float]
    ) -> float:
        """Convert curiosity profile to serendipity weight"""
        # Weighted combination of curiosity dimensions
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
        """Calculate adjustment based on task context"""
        adjustment = 0.0
        
        # Reduce serendipity for time-critical tasks
        if context.get('urgency') == 'high':
            adjustment -= 0.2
        
        # Increase serendipity for exploratory tasks
        if context.get('task_type') == 'exploration':
            adjustment += 0.15
        
        # Adjust based on task progress
        progress = context.get('progress', 0.5)
        if progress < 0.3:
            # Early stage: more exploration
            adjustment += 0.1
        elif progress > 0.8:
            # Late stage: more focus
            adjustment -= 0.1
        
        return adjustment
    
    def _feedback_adjustment(
        self,
        feedback: List[Dict[str, Any]]
    ) -> float:
        """Calculate adjustment based on user feedback"""
        if not feedback:
            return 0.0
        
        # Calculate average satisfaction
        satisfactions = [f.get('satisfaction', 0.5) for f in feedback]
        avg_satisfaction = np.mean(satisfactions)
        
        # Calculate discovery appreciation
        discoveries = [f.get('appreciated_discovery', False) for f in feedback]
        discovery_rate = sum(discoveries) / len(discoveries)
        
        # Adjust based on feedback
        if avg_satisfaction > 0.7 and discovery_rate > 0.5:
            # User is happy and appreciates discoveries
            return 0.1
        elif avg_satisfaction < 0.4:
            # User is not satisfied, reduce serendipity
            return -0.15
        
        return 0.0
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        return {
            'relevance_weight': self.state.current_relevance_weight,
            'serendipity_weight': self.state.current_serendipity_weight
        }
```

### 3.5 Curiosity Profiler

```python
# mirothinker/serendipity/curiosity_engine/profiler.py
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class CuriosityProfile:
    """User curiosity profile with multiple dimensions"""
    user_id: str
    novelty_seeking: float = 0.5
    diversity_appreciation: float = 0.5
    surprise_tolerance: float = 0.5
    exploration_depth: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'novelty_seeking': self.novelty_seeking,
            'diversity_appreciation': self.diversity_appreciation,
            'surprise_tolerance': self.surprise_tolerance,
            'exploration_depth': self.exploration_depth,
            'interaction_count': self.interaction_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CuriosityProfile':
        return cls(
            user_id=data['user_id'],
            novelty_seeking=data.get('novelty_seeking', 0.5),
            diversity_appreciation=data.get('diversity_appreciation', 0.5),
            surprise_tolerance=data.get('surprise_tolerance', 0.5),
            exploration_depth=data.get('exploration_depth', 0.5),
            interaction_count=data.get('interaction_count', 0)
        )


class CuriosityProfiler:
    """
    Profiles user curiosity based on interactions.
    
    Tracks and updates curiosity dimensions based on:
    - Explicit feedback
    - Implicit signals (dwell time, follow-ups)
    - Choice patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.update_frequency = config.get('update_frequency', 'per_session')
        self.persistence = config.get('persistence', 'session_only')
        self.profiles: Dict[str, CuriosityProfile] = {}
    
    def get_or_create_profile(self, user_id: str) -> CuriosityProfile:
        """Get existing profile or create new one"""
        if user_id not in self.profiles:
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
            interaction: Interaction data including:
                - selected_items: Items user selected
                - dwell_times: Time spent on each item
                - explicit_ratings: User ratings
                - follow_up_actions: Subsequent actions
                
        Returns:
            Updated curiosity profile
        """
        profile = self.get_or_create_profile(user_id)
        
        # Update novelty seeking based on selection patterns
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
        
        # Update surprise tolerance from ratings
        if 'explicit_ratings' in interaction:
            profile.surprise_tolerance = self._update_surprise_tolerance(
                profile.surprise_tolerance,
                interaction['explicit_ratings']
            )
        
        # Update exploration depth from follow-ups
        if 'follow_up_actions' in interaction:
            profile.exploration_depth = self._update_exploration_depth(
                profile.exploration_depth,
                interaction['follow_up_actions']
            )
        
        profile.interaction_count += 1
        profile.updated_at = datetime.now()
        
        return profile
    
    def _update_novelty_seeking(
        self,
        current: float,
        selected_items: List[Dict[str, Any]]
    ) -> float:
        """Update novelty seeking based on selection of novel items"""
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
        """Update diversity appreciation based on category spread"""
        if len(selected_items) < 2:
            return current
        
        # Calculate category diversity
        categories = [item.get('category') for item in selected_items if item.get('category')]
        if not categories:
            return current
        
        unique_categories = len(set(categories))
        diversity_ratio = unique_categories / len(categories)
        
        # Update
        alpha = 0.2
        return (1 - alpha) * current + alpha * diversity_ratio
    
    def _update_surprise_tolerance(
        self,
        current: float,
        ratings: List[Dict[str, Any]]
    ) -> float:
        """Update surprise tolerance from ratings of surprising items"""
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
        
        # Update
        alpha = 0.15
        return (1 - alpha) * current + alpha * avg_surprise_rating
    
    def _update_exploration_depth(
        self,
        current: float,
        follow_ups: List[Dict[str, Any]]
    ) -> float:
        """Update exploration depth from follow-up behavior"""
        if not follow_ups:
            return current
        
        # Calculate average exploration depth
        depths = [f.get('depth', 0) for f in follow_ups]
        avg_depth = np.mean(depths) if depths else 0
        
        # Normalize to 0-1 (assuming max depth of 5)
        normalized_depth = min(1.0, avg_depth / 5.0)
        
        # Update
        alpha = 0.15
        return (1 - alpha) * current + alpha * normalized_depth
```

---

## 4. Integration Points

### 4.1 Agent Integration

```python
# mirothinker/core/agent.py (modifications)
from typing import Dict, Any, List, Optional
from ..serendipity.reranker.hybrid import HybridSerendipityReranker
from ..serendipity.reranker.adaptive import AdaptiveWeightController
from ..serendipity.curiosity_engine.profiler import CuriosityProfiler


class MiroThinkerAgent:
    """Enhanced agent with serendipity support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.serendipity_config = config.get('serendipity', {})
        
        # Initialize serendipity components if enabled
        if self.serendipity_config.get('enabled', False):
            self._init_serendipity()
    
    def _init_serendipity(self):
        """Initialize serendipity components"""
        # Initialize reranker
        self.serendipity_reranker = HybridSerendipityReranker(
            self.serendipity_config
        )
        
        # Initialize adaptive controller if enabled
        if self.serendipity_config.get('adaptive_weights', {}).get('enabled'):
            self.adaptive_controller = AdaptiveWeightController(
                self.serendipity_config['adaptive_weights']
            )
        
        # Initialize curiosity profiler if enabled
        if self.serendipity_config.get('curiosity', {}).get('profiling', {}).get('enabled'):
            self.curiosity_profiler = CuriosityProfiler(
                self.serendipity_config['curiosity']['profiling']
            )
    
    async def process_task(
        self,
        task: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process task with optional serendipity enhancement.
        
        Args:
            task: Task specification
            user_id: Optional user identifier
            
        Returns:
            Task results with serendipity enhancements
        """
        # Get curiosity profile if available
        curiosity_profile = None
        if user_id and hasattr(self, 'curiosity_profiler'):
            curiosity_profile = self.curiosity_profiler.get_or_create_profile(user_id)
        
        # Get adaptive weights if enabled
        weights = None
        if hasattr(self, 'adaptive_controller') and curiosity_profile:
            weights = self.adaptive_controller.adapt_weights(
                curiosity_profile.to_dict(),
                task,
                []  # Recent feedback
            )
        
        # Execute base task processing
        results = await self._execute_task(task)
        
        # Apply serendipity reranking if enabled
        if self.serendipity_config.get('enabled') and hasattr(self, 'serendipity_reranker'):
            results = await self._apply_serendipity(results, task, weights)
        
        return results
    
    async def _apply_serendipity(
        self,
        results: Dict[str, Any],
        task: Dict[str, Any],
        weights: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Apply serendipity enhancement to results"""
        # Convert results to serendipity items
        items = self._convert_to_serendipity_items(results)
        
        # Build context
        context = {
            'task': task,
            'user_profile': task.get('user_profile', {}),
            'history': task.get('history', [])
        }
        
        # Apply reranking
        reranked = self.serendipity_reranker.rerank(items, context)
        
        # Convert back to results format
        enhanced_results = self._convert_from_serendipity_items(reranked)
        
        # Add serendipity metadata
        enhanced_results['_serendipity'] = {
            'applied': True,
            'weights': weights or self.serendipity_reranker.config.to_dict(),
            'item_count': len(reranked)
        }
        
        return enhanced_results
```

### 4.2 Tool Call Integration

```python
# mirothinker/serendipity/integration/tool_filter.py
from typing import Dict, Any, List, Optional
import random


class SerendipityToolFilter:
    """
    Filters and enhances tool calls with serendipity.
    
    Can inject unexpected but potentially useful tools
    into the tool call sequence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exploration_strategy = config.get('exploration', {}).get('strategy', 'epsilon_greedy')
        self.epsilon = config.get('exploration', {}).get('epsilon', 0.2)
        self.temperature = config.get('exploration', {}).get('temperature', 0.5)
    
    def filter_tools(
        self,
        available_tools: List[Dict[str, Any]],
        selected_tools: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Filter and potentially enhance tool selection.
        
        Args:
            available_tools: All available tools
            selected_tools: Tools selected by the agent
            context: Task context
            
        Returns:
            Potentially enhanced tool list
        """
        if not self.config.get('enabled', False):
            return selected_tools
        
        # Apply exploration strategy
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_selection(
                available_tools, selected_tools, context
            )
        
        return selected_tools
    
    def _epsilon_greedy_selection(
        self,
        available_tools: List[Dict[str, Any]],
        selected_tools: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Epsilon-greedy tool selection.
        
        With probability epsilon, inject a serendipitous tool.
        """
        if random.random() > self.epsilon:
            return selected_tools
        
        # Find tools not in selected set
        available_ids = {t['id'] for t in available_tools}
        selected_set = set(selected_tools)
        unexplored = available_ids - selected_set
        
        if not unexplored:
            return selected_tools
        
        # Select a serendipitous tool
        # In practice, this would use more sophisticated selection
        serendipitous_tool = random.choice(list(unexplored))
        
        # Inject at a random position
        enhanced = selected_tools.copy()
        insert_pos = random.randint(0, len(enhanced))
        enhanced.insert(insert_pos, serendipitous_tool)
        
        return enhanced
```

---

## 5. Configuration Schema

### 5.1 Complete Configuration Example

```yaml
# config/serendipity.yaml
version: "1.0"

serendipity:
  # Master toggle
  enabled: true
  
  # Implementation phase
  phase: "advanced"  # quick_wins | core | advanced
  
  # Logging and metrics
  logging:
    level: "INFO"
    log_scores: true
    log_decisions: true
    
  # Algorithm selection
  algorithm:
    # Primary algorithm
    name: "adaptive"
    
    # Fallback algorithm
    fallback: "sog"
    
    # Algorithm-specific parameters
    params:
      lambda_param: 0.5
      diversity_weight: 0.3
      
  # Reranking configuration
  reranking:
    enabled: true
    top_k: 10
    
    # Where to apply reranking
    apply_to:
      - tool_calls
      - research_results
      - knowledge_retrieval
      
  # Balance configuration
  balance:
    # Static weights (used when adaptive is disabled)
    relevance_weight: 0.7
    unexpectedness_weight: 0.3
    
  # Adaptive weight configuration
  adaptive_weights:
    enabled: true
    strategy: "curiosity_driven"  # curiosity_driven | performance_driven | hybrid
    
    # Bounds for serendipity weight
    min_serendipity: 0.1
    max_serendipity: 0.7
    
    # Adaptation rate
    adaptation_rate: 0.1
    
  # Curiosity profiling
  curiosity:
    profiling:
      enabled: true
      
      # How often to update profile
      update_frequency: "per_session"  # per_task | per_session | per_query
      
      # Profile persistence
      persistence: "user_profile"  # session_only | user_profile
      
    # Curiosity dimensions to track
    dimensions:
      - novelty_seeking
      - diversity_appreciation
      - surprise_tolerance
      - exploration_depth
      
    # Initial values for new users
    initial_profile:
      novelty_seeking: 0.5
      diversity_appreciation: 0.5
      surprise_tolerance: 0.5
      exploration_depth: 0.5
      
  # Exploration strategy
  exploration:
    strategy: "epsilon_greedy"  # epsilon_greedy | ucb | thompson_sampling
    epsilon: 0.2
    temperature: 0.5
    
  # Individual algorithm configurations
  algorithms:
    sog:
      enabled: true
      lambda_param: 0.5
      diversity_weight: 0.3
      similarity_threshold: 0.7
      
    kfn:
      enabled: true
      k: 5
      distance_metric: "cosine"
      
    sccf:
      enabled: true
      n_clusters: 10
      cluster_method: "kmeans"
      
    sirup:
      enabled: true
      curiosity_model: "multi_dim"
      content_features:
        - semantic
        - structural
        - temporal
        
    tangent:
      enabled: true
      graph_depth: 3
      relationship_types:
        - similarity
        - complementarity
        - contrast
        
    serencdr:
      enabled: true
      model_path: "models/serencdr_v1.pt"
      domains:
        - research
        - code
        - documentation
        
  # Full pipeline configuration
  pipeline:
    enabled: true
    stages:
      - name: "tool_selection"
        enabled: true
        serendipity_weight: 0.3
        
      - name: "result_reranking"
        enabled: true
        serendipity_weight: 0.4
        
      - name: "context_enhancement"
        enabled: true
        serendipity_weight: 0.2
        
      - name: "follow_up_generation"
        enabled: true
        serendipity_weight: 0.5
        
  # User control settings
  user_control:
    # Allow users to opt-in/opt-out
    opt_in: true
    
    # Default curiosity level
    default_curiosity_level: "medium"  # low | medium | high
    
    # Level definitions
    curiosity_levels:
      low:
        serendipity_weight: 0.2
        exploration_rate: 0.1
      medium:
        serendipity_weight: 0.4
        exploration_rate: 0.2
      high:
        serendipity_weight: 0.6
        exploration_rate: 0.3
        
  # Experimentation framework
  experimentation:
    ab_testing:
      enabled: true
      
      # Test variants
      variants:
        - name: "control"
          serendipity_level: 0.0
          description: "No serendipity"
          
        - name: "low"
          serendipity_level: 0.3
          description: "Low serendipity"
          
        - name: "medium"
          serendipity_level: 0.5
          description: "Medium serendipity"
          
        - name: "high"
          serendipity_level: 0.7
          description: "High serendipity"
          
      # Success metrics
      metrics:
        - task_completion
        - user_satisfaction
        - discovery_rate
        - time_to_completion
        
  # Feedback loop
  feedback:
    enabled: true
    
    # Types of feedback to collect
    types:
      - explicit_rating
      - implicit_dwell_time
      - follow_up_actions
      - skip_behavior
      
    # How quickly to incorporate feedback
    update_frequency: "immediate"  # immediate | batch | daily
    
    # Feedback weights
    weights:
      explicit_rating: 1.0
      implicit_dwell_time: 0.5
      follow_up_actions: 0.7
      skip_behavior: 0.3
```

---

## 6. Trade-offs and Optimization

### 6.1 Serendipity vs. Accuracy Balance

| Approach | Serendipity | Accuracy | Use Case |
|----------|-------------|----------|----------|
| Pure Relevance | Low | High | Time-critical tasks |
| Balanced (0.5) | Medium | Medium | General research |
| High Serendipity | High | Medium | Exploration tasks |
| Adaptive | Variable | Variable | Dynamic contexts |

**Recommendation**: Start with adaptive weights based on task type and user profile.

### 6.2 Computational Overhead Management

```python
# Optimization strategies

class SerendipityOptimizer:
    """Manages computational overhead of serendipity algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.caching_enabled = config.get('caching', True)
        self.cache = {}
        self.max_cache_size = config.get('max_cache_size', 1000)
        
    def should_compute_serendipity(
        self,
        context: Dict[str, Any]
    ) -> bool:
        """Decide if serendipity computation is warranted"""
        # Skip for very small result sets
        if context.get('result_count', 0) < 3:
            return False
        
        # Skip for time-critical tasks
        if context.get('urgency') == 'high':
            return False
        
        # Check cache
        cache_key = self._get_cache_key(context)
        if cache_key in self.cache:
            return False
        
        return True
    
    def get_cached_or_compute(
        self,
        cache_key: str,
        compute_fn: Callable,
        *args
    ):
        """Get cached result or compute and cache"""
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = compute_fn(*args)
        
        if len(self.cache) >= self.max_cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
```

### 6.3 User Control Options

```python
# User control implementation

class UserSerendipityControl:
    """Handles user preferences for serendipity"""
    
    def __init__(self, config: Dict[str, Any]):
        self.opt_in_required = config.get('opt_in', True)
        self.levels = config.get('curiosity_levels', {})
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's serendipity preferences"""
        # Load from user profile
        preferences = self._load_preferences(user_id)
        
        return {
            'enabled': preferences.get('enabled', not self.opt_in_required),
            'level': preferences.get('level', 'medium'),
            'weights': self.levels.get(
                preferences.get('level', 'medium'),
                self.levels.get('medium')
            )
        }
    
    def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any
    ):
        """Update user preference"""
        preferences = self._load_preferences(user_id)
        preferences[key] = value
        self._save_preferences(user_id, preferences)
```

---

## 7. Migration Path

### 7.1 From Current System

```python
# Migration guide

class SerendipityMigration:
    """Handles migration from non-serendipity system"""
    
    def __init__(self, agent_instance):
        self.agent = agent_instance
        self.migration_steps = [
            self._step1_add_config,
            self._step2_add_reranker,
            self._step3_add_curiosity,
            self._step4_add_adaptive,
            self._step5_full_pipeline
        ]
    
    def migrate(self, target_phase: str = "quick_wins"):
        """
        Migrate agent to serendipity-enabled version.
        
        Args:
            target_phase: Target migration phase
        """
        phase_map = {
            "quick_wins": 1,
            "core": 3,
            "advanced": 5
        }
        
        target_step = phase_map.get(target_phase, 1)
        
        for i, step in enumerate(self.migration_steps[:target_step], 1):
            print(f"Executing migration step {i}/{target_step}...")
            step()
    
    def _step1_add_config(self):
        """Step 1: Add serendipity configuration"""
        # Add default config to agent
        default_config = {
            'serendipity': {
                'enabled': True,
                'phase': 'quick_wins',
                'algorithm': {'name': 'sog', 'lambda_param': 0.5},
                'balance': {'relevance_weight': 0.7, 'unexpectedness_weight': 0.3}
            }
        }
        self.agent.config.update(default_config)
    
    def _step2_add_reranker(self):
        """Step 2: Add basic reranker"""
        from ..serendipity.reranker.hybrid import HybridSerendipityReranker
        
        self.agent.serendipity_reranker = HybridSerendipityReranker(
            self.agent.config['serendipity']
        )
    
    def _step3_add_curiosity(self):
        """Step 3: Add curiosity profiling"""
        from ..serendipity.curiosity_engine.profiler import CuriosityProfiler
        
        self.agent.curiosity_profiler = CuriosityProfiler(
            self.agent.config['serendipity']['curiosity']['profiling']
        )
    
    def _step4_add_adaptive(self):
        """Step 4: Add adaptive weights"""
        from ..serendipity.reranker.adaptive import AdaptiveWeightController
        
        self.agent.adaptive_controller = AdaptiveWeightController(
            self.agent.config['serendipity']['adaptive_weights']
        )
    
    def _step5_full_pipeline(self):
        """Step 5: Enable full pipeline"""
        from ..serendipity.integration.pipeline import SerendipityPipeline
        
        self.agent.serendipity_pipeline = SerendipityPipeline(
            self.agent.config['serendipity']['pipeline']
        )
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/serendipity/test_algorithms.py

class TestSOGAlgorithm:
    """Unit tests for SOG algorithm"""
    
    def test_serendipity_formula(self):
        """Test core serendipity formula"""
        algo = SOGAlgorithm({})
        
        # Perfect relevance, perfect unexpectedness
        assert algo.compute_serendipity_formula(1.0, 1.0) == 1.0
        
        # No relevance
        assert algo.compute_serendipity_formula(0.0, 1.0) == 0.0
        
        # No unexpectedness
        assert algo.compute_serendipity_formula(1.0, 0.0) == 0.0
    
    def test_reranking_preserves_items(self):
        """Test that reranking doesn't lose items"""
        algo = SOGAlgorithm({'lambda_param': 0.5})
        
        items = [
            SerendipityItem(id=f"item_{i}", content=f"content_{i}", relevance_score=0.5)
            for i in range(10)
        ]
        
        reranked = algo.rerank(items, {}, top_k=10)
        
        assert len(reranked) == 10
        assert set(item.id for item in reranked) == set(item.id for item in items)
```

### 8.2 Integration Tests

```python
# tests/serendipity/test_integration.py

class TestAgentIntegration:
    """Integration tests for serendipity in agent"""
    
    async def test_serendipity_disabled(self):
        """Test agent works without serendipity"""
        config = {'serendipity': {'enabled': False}}
        agent = MiroThinkerAgent(config)
        
        result = await agent.process_task({'query': 'test'})
        assert '_serendipity' not in result
    
    async def test_serendipity_enabled(self):
        """Test agent with serendipity enabled"""
        config = {
            'serendipity': {
                'enabled': True,
                'phase': 'quick_wins',
                'algorithm': {'name': 'sog'}
            }
        }
        agent = MiroThinkerAgent(config)
        
        result = await agent.process_task({'query': 'test'})
        assert '_serendipity' in result
        assert result['_serendipity']['applied']
```

### 8.3 Performance Tests

```python
# tests/serendipity/test_performance.py

class TestSerendipityPerformance:
    """Performance tests for serendipity algorithms"""
    
    def test_reranking_latency(self):
        """Test that reranking adds acceptable latency"""
        algo = SOGAlgorithm({})
        
        items = [
            SerendipityItem(id=f"item_{i}", content=f"content_{i}", relevance_score=0.5)
            for i in range(100)
        ]
        
        import time
        start = time.time()
        algo.rerank(items, {}, top_k=10)
        elapsed = time.time() - start
        
        # Should complete in under 100ms for 100 items
        assert elapsed < 0.1
```

---

## 9. Summary

### 9.1 Implementation Checklist

**Phase 1 (Quick Wins)**
- [ ] Create serendipity module structure
- [ ] Implement SOG algorithm
- [ ] Create basic hybrid reranker
- [ ] Add configuration schema
- [ ] Integrate with agent
- [ ] Write Phase 1 tests

**Phase 2 (Core Integration)**
- [ ] Implement curiosity profiler
- [ ] Add k-Furthest Neighbors algorithm
- [ ] Add SC-CF algorithm
- [ ] Implement adaptive weight controller
- [ ] Add serendipity metrics
- [ ] Write Phase 2 tests

**Phase 3 (Advanced Features)**
- [ ] Implement SIRUP algorithm
- [ ] Implement TANGENT algorithm
- [ ] Implement SerenCDR algorithm
- [ ] Create full serendipity pipeline
- [ ] Add A/B testing framework
- [ ] Implement feedback loop
- [ ] Write Phase 3 tests

### 9.2 Key Success Metrics

1. **Serendipity Rate**: % of results with serendipity score > 0.5
2. **User Satisfaction**: Ratings for serendipitous discoveries
3. **Task Completion**: Success rate with serendipity enabled
4. **Discovery Rate**: New useful findings per session
5. **Latency Overhead**: Additional time from serendipity processing

### 9.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Reduced accuracy | Configurable weights, opt-out option |
| Increased latency | Caching, lazy evaluation, algorithm selection |
| User confusion | Clear labeling, explanation of discoveries |
| Over-exploration | Bounds on serendipity weight, feedback loop |

---

## 10. Appendix: File Structure

```
mirothinker/
├── serendipity/
│   ├── __init__.py
│   ├── config.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sog.py
│   │   ├── kfn.py
│   │   ├── sccf.py
│   │   ├── sirup.py
│   │   ├── tangent.py
│   │   └── serencdr.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── discovery.py
│   │   ├── curiosity.py
│   │   └── metrics.py
│   ├── curiosity_engine/
│   │   ├── __init__.py
│   │   ├── profiler.py
│   │   ├── explorer.py
│   │   └── decay.py
│   ├── reranker/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hybrid.py
│   │   └── adaptive.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   ├── unexpectedness.py
│   │   └── relevance.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── tool_filter.py
│   │   ├── result_enhancer.py
│   │   └── pipeline.py
│   └── experimentation/
│       ├── __init__.py
│       ├── ab_test.py
│       └── feedback.py
└── config/
    └── serendipity.yaml
```

---

*Document Version: 1.0*
*Last Updated: 2024*
