# Serendipity Algorithms for Deep Research Agents
## Expert Implementation Guide for MiroThinker

---

## Executive Summary

For a deep research agent like MiroThinker, serendipity algorithms must balance:
- **Relevance**: Information must advance the research goal
- **Unexpectedness**: Sources/paths should not be obvious
- **Computational Efficiency**: Must work within 600-turn constraints
- **Actionability**: Results must lead to valid tool sequences

---

## TOP 3 RECOMMENDED ALGORITHMS

---

## 1. SOG-Research: Serendipity-Oriented Greedy for Research Agents

### Why It's Suitable

**Primary Use Cases:**
- Search result re-ranking (diversifying web search outputs)
- Tool sequence optimization (discovering novel tool combinations)
- Research trajectory selection (choosing next investigation direction)

SOG is ideal because it:
1. Works as a post-processor (integrates with existing search/retrieval)
2. Balances multiple objectives via tunable weights
3. Has O(n²) complexity - feasible for typical result sets (10-100 items)
4. Naturally handles the multi-turn research context

### Mathematical Formulation

```
Score(i, u, B) = α₁·Relevance(i,u) + α₂·Diversity(i,B) + α₃·ProfileDistance(i,u) + α₄·Novelty(i)

Where:
- Relevance(i,u) = cosine(query_embedding, item_embedding)
- Diversity(i,B) = min_{j∈B} (1 - cosine(item_i, item_j))
- ProfileDistance(i,u) = 1 - cosine(item_i, user_profile)
- Novelty(i) = 1 / log(1 + popularity(i))
- B = already selected items (growing set)
```

### Pseudocode

```python
def sog_research_ranking(candidate_items, user_profile, query_embedding, 
                         alpha=(0.4, 0.2, 0.2, 0.2), k=10):
    """
    SOG for research agent search result diversification.
    
    Args:
        candidate_items: List of (item_id, embedding, metadata) from search
        user_profile: Aggregated embedding of user's research history
        query_embedding: Current research query embedding
        alpha: Weights for (relevance, diversity, profile_dist, novelty)
        k: Number of items to select
    
    Returns:
        Selected item IDs in serendipitous order
    """
    selected = []
    remaining = set(range(len(candidate_items)))
    
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_item = None
        
        for idx in remaining:
            item = candidate_items[idx]
            
            # Component 1: Relevance to current query
            rel = cosine_similarity(item.embedding, query_embedding)
            
            # Component 2: Diversity from already selected
            if selected:
                div = min(1 - cosine_similarity(item.embedding, 
                         candidate_items[s].embedding) for s in selected)
            else:
                div = 1.0
            
            # Component 3: Distance from user profile (unexpectedness)
            prof_dist = 1 - cosine_similarity(item.embedding, user_profile)
            
            # Component 4: Novelty (inverse popularity)
            novelty = 1.0 / math.log(1 + item.citation_count)
            
            # Combined score
            score = (alpha[0] * rel + 
                    alpha[1] * div + 
                    alpha[2] * prof_dist + 
                    alpha[3] * novelty)
            
            if score > best_score:
                best_score = score
                best_item = idx
        
        selected.append(best_item)
        remaining.remove(best_item)
    
    return [candidate_items[i].id for i in selected]
```

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Score computation | O(n) per item | n = candidate count |
| Selection loop | O(k) iterations | k = output size |
| Diversity calc | O(k) per item | Against selected set |
| **Total** | **O(k·n²)** | Dominated by pairwise comparisons |
| **Optimized** | **O(k·n·d)** | d = embedding dimension (with approximations) |

**Practical Notes:**
- For n=50 candidates, k=10: ~2,500 similarity computations
- With 768-dim embeddings: ~2ms on modern CPU
- Use FAISS for approximate nearest neighbors to reduce to O(k·n·log(n))

### Required Inputs/Outputs

**Inputs:**
```python
{
    "candidate_results": [
        {
            "id": "source_id",
            "embedding": np.array(...),  # e.g., 768-dim from sentence-transformer
            "metadata": {
                "url": "...",
                "citation_count": 150,
                "domain": "arxiv.org",
                "publish_date": "2023-01-15"
            }
        }
    ],
    "user_profile": np.array(...),  # Aggregated research history embedding
    "query_embedding": np.array(...),  # Current research focus
    "weights": {"relevance": 0.4, "diversity": 0.2, 
                "unexpectedness": 0.2, "novelty": 0.2}
}
```

**Outputs:**
```python
{
    "ranked_results": ["id1", "id2", "id3", ...],  # Serendipitous ordering
    "scores": {
        "id1": {"relevance": 0.85, "diversity": 0.70, 
                "unexpectedness": 0.60, "novelty": 0.90},
        ...
    },
    "selection_trace": [...]  # For explainability
}
```

### Integration with MiroThinker

```python
class SerendipitousSearchModule:
    def __init__(self, embedder, alpha=(0.4, 0.2, 0.2, 0.2)):
        self.embedder = embedder
        self.alpha = alpha
        self.research_history = []  # Accumulated embeddings
    
    def diversify_search_results(self, raw_results, current_query):
        # Embed all results
        embeddings = self.embedder.encode([r.content for r in raw_results])
        
        # Build user profile from research history
        user_profile = np.mean(self.research_history, axis=0) \
                       if self.research_history else np.zeros(768)
        
        # Apply SOG re-ranking
        ranked_ids = sog_research_ranking(
            candidate_items=zip(raw_results, embeddings),
            user_profile=user_profile,
            query_embedding=self.embedder.encode(current_query),
            alpha=self.alpha,
            k=min(10, len(raw_results))
        )
        
        # Update history with selected items
        for rid in ranked_ids[:3]:  # Top 3 become part of profile
            self.research_history.append(embeddings[rid])
        
        return ranked_ids
```

---

## 2. RWR-Path: Random Walk with Restarts for Research Trajectory

### Why It's Suitable

**Primary Use Cases:**
- Research path exploration (finding alternative investigation directions)
- Cross-domain connection discovery (linking seemingly unrelated topics)
- Serendipitous question generation (what to investigate next)

RWR-Path is ideal because it:
1. Models research as graph traversal (natural fit for multi-turn agents)
2. Restart probability controls exploration vs. exploitation
3. Finds "bridge" concepts that connect distant knowledge areas
4. Provides probabilistic rankings with theoretical guarantees

### Graph Construction

```
Research Knowledge Graph G = (V, E)

Vertices V:
- Research topics (nodes from search queries)
- Information sources (papers, web pages)
- Concepts/entities (extracted from content)
- Tools used (search, browse, code execution)

Edges E:
- Topic → Source (relevance-weighted)
- Source → Concept (TF-IDF or embedding similarity)
- Concept → Topic (co-occurrence)
- Tool → Source (which tools retrieved which sources)

Edge weights: w(u,v) = similarity(u,v) × reliability_factor
```

### Pseudocode

```python
def rwr_path_discovery(graph, seed_nodes, restart_prob=0.3, 
                       max_steps=1000, convergence_threshold=1e-6):
    """
    Random Walk with Restarts for research path discovery.
    
    Args:
        graph: NetworkX graph (V, E with weights)
        seed_nodes: Starting research topics/concepts
        restart_prob: Probability of returning to seeds (α)
        max_steps: Maximum iterations
        convergence_threshold: L1 norm convergence criterion
    
    Returns:
        Stationary distribution over nodes (serendipity scores)
    """
    n = graph.number_of_nodes()
    node_idx = {node: i for i, node in enumerate(graph.nodes())}
    
    # Initialize: uniform over seed nodes
    p = np.zeros(n)
    seed_indices = [node_idx[s] for s in seed_nodes if s in node_idx]
    for idx in seed_indices:
        p[idx] = 1.0 / len(seed_indices)
    
    # Build transition matrix
    W = nx.to_numpy_array(graph, weight='weight')
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    P = W / row_sums  # Row-stochastic transition matrix
    
    # Power iteration
    for step in range(max_steps):
        # RWR update: α·restart + (1-α)·transition
        p_new = restart_prob * p_seed + (1 - restart_prob) * P.T @ p
        
        if np.linalg.norm(p_new - p, 1) < convergence_threshold:
            break
        p = p_new
    
    # Return node scores
    return {node: p[node_idx[node]] for node in graph.nodes()}


def discover_serendipitous_paths(graph, current_topic, user_profile_topics,
                                  n_paths=5, path_length=4):
    """
    Find unexpected research paths using RWR scores.
    
    Strategy: High RWR score but far from user profile = serendipitous
    """
    # Run RWR from current topic
    rwr_scores = rwr_path_discovery(graph, [current_topic])
    
    # Calculate distance from user profile for each node
    profile_distance = {}
    for node in graph.nodes():
        if node in user_profile_topics:
            profile_distance[node] = 0
        else:
            # Shortest path distance to any profile topic
            dists = [nx.shortest_path_length(graph, node, p, 
                     weight='weight') for p in user_profile_topics]
            profile_distance[node] = min(dists) if dists else float('inf')
    
    # Serendipity = RWR score × profile distance
    serendipity = {}
    for node in graph.nodes():
        if node != current_topic:
            serendipity[node] = rwr_scores[node] * profile_distance[node]
    
    # Extract paths to top serendipitous nodes
    paths = []
    for target in sorted(serendipity, key=serendipity.get, reverse=True)[:n_paths]:
        try:
            path = nx.shortest_path(graph, current_topic, target, 
                                   weight='weight')[:path_length]
            paths.append({
                'path': path,
                'serendipity_score': serendipity[target],
                'rwr_score': rwr_scores[target],
                'profile_distance': profile_distance[target]
            })
        except nx.NetworkXNoPath:
            continue
    
    return paths
```

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Transition matrix | O(n²) | n = graph nodes |
| Each iteration | O(n²) | Matrix-vector multiply |
| Convergence | O(τ·n²) | τ = iterations to converge |
| Path extraction | O(k·(n+m)·log n) | k paths, m edges |
| **Total** | **O(τ·n²)** | Typically τ < 50 |
| **Sparse optimization** | **O(τ·m)** | m = edges (often m << n²) |

**Practical Notes:**
- For research graph with 1,000 nodes, 5,000 edges: ~0.1s per RWR
- Precompute transition matrix for repeated queries
- Use personalized PageRank libraries (networkx, igraph) for efficiency

### Required Inputs/Outputs

**Inputs:**
```python
{
    "knowledge_graph": {
        "nodes": ["topic1", "paper1", "concept1", "tool1", ...],
        "edges": [
            {"source": "topic1", "target": "paper1", "weight": 0.85},
            {"source": "paper1", "target": "concept1", "weight": 0.72},
            ...
        ]
    },
    "current_focus": "topic1",  # Current research topic
    "user_profile": ["topic2", "topic3", "concept5"],  # Known interests
    "parameters": {
        "restart_probability": 0.3,
        "max_path_length": 4,
        "num_paths": 5
    }
}
```

**Outputs:**
```python
{
    "serendipitous_paths": [
        {
            "path": ["topic1", "paper1", "concept1", "topic4"],
            "serendipity_score": 0.78,
            "explanation": "Connects current topic to unexpected area via shared concept",
            "suggested_tools": ["search", "browse"]
        },
        ...
    ],
    "node_scores": {
        "topic4": {"rwr": 0.15, "distance": 5.2, "serendipity": 0.78},
        ...
    }
}
```

### Integration with MiroThinker

```python
class ResearchPathExplorer:
    def __init__(self):
        self.kg = nx.DiGraph()  # Knowledge graph
        self.topic_history = []
    
    def build_graph_from_research(self, search_results, extracted_concepts):
        """Incrementally build knowledge graph from research activity."""
        for result in search_results:
            # Add source node
            self.kg.add_node(result.id, type='source', 
                           embedding=result.embedding)
            
            # Connect to concepts
            for concept in extracted_concepts[result.id]:
                self.kg.add_node(concept, type='concept')
                self.kg.add_edge(result.id, concept, 
                               weight=result.concept_scores[concept])
        
        # Connect concepts to each other (co-occurrence)
        for i, c1 in enumerate(extracted_concepts):
            for c2 in extracted_concepts[i+1:]:
                if self.kg.has_edge(c1, c2):
                    self.kg[c1][c2]['weight'] += 0.1
                else:
                    self.kg.add_edge(c1, c2, weight=0.1)
    
    def suggest_next_directions(self, current_topic, n_suggestions=3):
        """Suggest serendipitous research directions."""
        paths = discover_serendipitous_paths(
            self.kg, current_topic, self.topic_history, n_paths=n_suggestions
        )
        
        # Convert paths to actionable suggestions
        suggestions = []
        for path in paths:
            target = path['path'][-1]
            suggestions.append({
                'direction': target,
                'reasoning': f"Found via: {' → '.join(path['path'])}",
                'confidence': path['serendipity_score'],
                'query_template': f"How does {path['path'][0]} relate to {target}?"
            })
        
        return suggestions
```

---

## 3. Curiosity-Driven Tool Sequence Discovery (CTS)

### Why It's Suitable

**Primary Use Cases:**
- Novel tool combination discovery (e.g., search→browse→code→search)
- Alternative investigation strategies (different tool orderings)
- Breaking out of repetitive tool patterns

CTS is ideal because it:
1. Models tool sequences as Markov Decision Processes
2. Uses curiosity (prediction error) as intrinsic reward
3. Balances exploitation (known good sequences) with exploration
4. Learns from successful research trajectories

### Mathematical Formulation

```
State s_t = (research_context, available_tools, previous_tools_used)
Action a_t = select_tool ∈ {search, browse, code_execute, read_file, ...}
Reward R(s,a) = β₁·extrinsic_reward + β₂·curiosity_reward

Extrinsic reward = relevance of output to research goal
Curiosity reward = ||φ(s_{t+1}) - φ̂(s_{t+1})||²  (prediction error)

Where φ is a forward model predicting next state
```

### Pseudocode

```python
class CuriosityToolSelector:
    def __init__(self, tools, embedding_dim=128):
        self.tools = tools
        self.forward_model = ForwardModel(embedding_dim)
        self.inverse_model = InverseModel(embedding_dim)
        self.feature_encoder = FeatureEncoder(embedding_dim)
        
        # Track tool sequence patterns
        self.successful_sequences = []
        self.tool_transition_counts = defaultdict(lambda: defaultdict(int))
    
    def compute_curiosity_reward(self, state, action, next_state):
        """
        Compute intrinsic motivation via prediction error.
        """
        # Encode states
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Predict next state feature
        predicted_next = self.forward_model(state_feat, action)
        
        # Curiosity = prediction error
        curiosity = torch.norm(next_state_feat - predicted_next, p=2).item()
        
        return curiosity
    
    def select_tool(self, research_context, available_results, 
                   exploration_rate=0.3):
        """
        Select next tool using curiosity-driven exploration.
        """
        state = self._build_state(research_context, available_results)
        
        # Compute scores for each tool
        tool_scores = {}
        for tool in self.tools:
            # Base score: historical success rate
            base_score = self._get_historical_success(tool, state)
            
            # Simulate using this tool
            simulated_next = self._simulate_tool_use(tool, state)
            
            # Curiosity bonus
            curiosity = self.compute_curiosity_reward(state, tool, simulated_next)
            
            # Novelty: how different from recent sequences
            novelty = self._compute_sequence_novelty(tool)
            
            # Combined score
            tool_scores[tool] = (
                0.5 * base_score + 
                0.3 * curiosity + 
                0.2 * novelty
            )
        
        # Epsilon-greedy selection
        if random.random() < exploration_rate:
            return random.choice(self.tools)
        else:
            return max(tool_scores, key=tool_scores.get)
    
    def _compute_sequence_novelty(self, proposed_tool):
        """
        Measure how different this tool choice is from typical patterns.
        """
        if len(self.successful_sequences) < 5:
            return 1.0  # Maximum novelty initially
        
        # Count transitions to this tool in recent history
        recent_seqs = self.successful_sequences[-20:]
        transition_count = sum(
            1 for seq in recent_seqs 
            for i in range(len(seq)-1) 
            if seq[i+1] == proposed_tool
        )
        
        # Novelty = inverse frequency
        max_count = len(recent_seqs) * 5  # Assume avg sequence length of 5
        novelty = 1.0 - (transition_count / max_count)
        
        return novelty
    
    def update_from_trajectory(self, trajectory, success_score):
        """
        Learn from completed research trajectory.
        
        Args:
            trajectory: List of (state, action, reward) tuples
            success_score: Final relevance/success metric
        """
        if success_score > 0.7:  # Threshold for "successful"
            tool_sequence = [t[1] for t in trajectory]
            self.successful_sequences.append(tool_sequence)
            
            # Update transition counts
            for i in range(len(tool_sequence) - 1):
                self.tool_transition_counts[tool_sequence[i]][tool_sequence[i+1]] += 1
        
        # Update forward model (curiosity learning)
        for state, action, next_state in trajectory:
            state_feat = self.feature_encoder(state)
            next_state_feat = self.feature_encoder(next_state)
            predicted = self.forward_model(state_feat, action)
            
            loss = F.mse_loss(predicted, next_state_feat)
            self.forward_model.optimizer.zero_grad()
            loss.backward()
            self.forward_model.optimizer.step()


def discover_novel_tool_sequences(selector, research_goal, n_trials=10):
    """
    Discover novel tool sequences for a research goal.
    """
    discovered_sequences = []
    
    for trial in range(n_trials):
        sequence = []
        context = {"goal": research_goal, "collected_info": []}
        
        for step in range(10):  # Max 10 tools per trial
            # Select tool
            tool = selector.select_tool(context, [], exploration_rate=0.5)
            sequence.append(tool)
            
            # Simulate execution (in real system, actually execute)
            result = simulate_tool_execution(tool, context)
            context["collected_info"].append(result)
            
            # Check if goal achieved
            if check_goal_achievement(context, research_goal):
                discovered_sequences.append({
                    'sequence': sequence,
                    'length': len(sequence),
                    'novelty_score': compute_sequence_novelty(sequence, selector)
                })
                break
    
    # Return most novel successful sequences
    return sorted(discovered_sequences, 
                  key=lambda x: x['novelty_score'], reverse=True)[:5]
```

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Feature encoding | O(d) | d = embedding dimension |
| Forward model prediction | O(h·d) | h = hidden layer size |
| Tool scoring | O(|A|·h·d) | |A| = number of tools |
| Sequence generation | O(L·|A|·h·d) | L = sequence length |
| Model update | O(N·h·d) | N = trajectory length |
| **Total (per decision)** | **O(|A|·h·d)** | Typically < 1ms |
| **Training** | **O(E·N·h·d)** | E = epochs |

**Practical Notes:**
- Neural models can be small (2-layer MLP with 128 hidden units)
- Inference is fast enough for real-time tool selection
- Can pre-train on historical research trajectories

### Required Inputs/Outputs

**Inputs:**
```python
{
    "research_context": {
        "goal": "Analyze climate change impacts on agriculture",
        "collected_information": [...],
        "current_hypothesis": "Temperature rise affects crop yields",
        "uncertainty_areas": ["regional variations", "adaptation strategies"]
    },
    "available_tools": ["web_search", "browse", "code_execute", 
                       "read_file", "generate_image"],
    "tool_history": ["web_search", "browse", "web_search"],
    "parameters": {
        "exploration_rate": 0.3,
        "max_sequence_length": 10
    }
}
```

**Outputs:**
```python
{
    "selected_tool": "code_execute",
    "confidence": 0.72,
    "reasoning": {
        "base_score": 0.60,
        "curiosity_bonus": 0.85,
        "novelty_bonus": 0.70,
        "explanation": "High curiosity due to uncertainty about data patterns"
    },
    "alternative_tools": [
        {"tool": "browse", "score": 0.68},
        {"tool": "web_search", "score": 0.45}
    ]
}
```

### Integration with MiroThinker

```python
class SerendipitousToolOrchestrator:
    def __init__(self):
        self.tool_selector = CuriosityToolSelector([
            'web_search', 'browse', 'code_execute', 
            'read_file', 'generate_image'
        ])
        self.current_trajectory = []
    
    def decide_next_action(self, research_state):
        """
        Decide next tool using curiosity-driven selection.
        """
        tool = self.tool_selector.select_tool(
            research_state.context,
            research_state.available_results,
            exploration_rate=0.3
        )
        
        # Log decision
        self.current_trajectory.append({
            'state': research_state.to_vector(),
            'action': tool
        })
        
        return tool
    
    def feedback(self, result, success_score):
        """
        Provide feedback to learn from trajectory.
        """
        # Complete trajectory with results
        for i, step in enumerate(self.current_trajectory):
            if i < len(self.current_trajectory) - 1:
                step['next_state'] = self.current_trajectory[i+1]['state']
            else:
                step['next_state'] = result.to_vector()
        
        # Update selector
        self.tool_selector.update_from_trajectory(
            self.current_trajectory, success_score
        )
        
        # Reset for next trajectory
        self.current_trajectory = []
```

---

## Algorithm Comparison Summary

| Algorithm | Best For | Complexity | Training Required | Integration Difficulty |
|-----------|----------|------------|-------------------|----------------------|
| **SOG-Research** | Search result diversification | O(k·n²) | No | Easy |
| **RWR-Path** | Research trajectory exploration | O(τ·m) | No | Medium |
| **CTS** | Tool sequence discovery | O(|A|·h·d) | Yes (online) | Hard |

---

## Evaluation Metrics for Research Serendipity

### 1. Composite Serendipity Score

```
Serendipity(R, u) = Relevance(R, u) × Unexpectedness(R, u)

Where:
- Relevance(R, u) = (1/|R|) Σ relevance(r_i, query)
- Unexpectedness(R, u) = (1/|R|) Σ (1 - max_similarity(r_i, user_history))
```

### 2. Research Path Diversity

```python
def path_diversity(trajectories):
    """
    Measure diversity of research paths taken.
    """
    # Embed each trajectory
    traj_embeddings = [embed_trajectory(t) for t in trajectories]
    
    # Compute pairwise distances
    distances = []
    for i, t1 in enumerate(traj_embeddings):
        for t2 in traj_embeddings[i+1:]:
            distances.append(1 - cosine_similarity(t1, t2))
    
    return np.mean(distances)
```

### 3. Topic Coverage Expansion

```python
def topic_coverage_expansion(research_history, new_findings):
    """
    Measure how much new topic area was discovered.
    """
    # Extract topics from history and new findings
    history_topics = extract_topics(research_history)
    new_topics = extract_topics(new_findings)
    
    # Jaccard distance from known topics
    expansion = len(new_topics - history_topics) / len(new_topics)
    
    return expansion
```

### 4. Tool Combination Novelty

```python
def tool_combination_novelty(tool_sequence, historical_sequences):
    """
    Measure novelty of tool sequence.
    """
    # Convert to n-grams
    ngrams = set(zip(*[tool_sequence[i:] for i in range(3)]))
    
    # Count novel n-grams
    historical_ngrams = set()
    for seq in historical_sequences:
        historical_ngrams.update(zip(*[seq[i:] for i in range(3)]))
    
    novelty = len(ngrams - historical_ngrams) / len(ngrams)
    return novelty
```

### 5. Human Evaluation Protocol

```
For each research output, have annotators rate:
1. Relevance (1-5): How relevant to the research goal?
2. Unexpectedness (1-5): How surprising is this finding?
3. Usefulness (1-5): How useful for advancing the research?
4. Actionability (1-5): Can this lead to concrete next steps?

Serendipity Score = (Relevance × Unexpectedness × Usefulness)^(1/3)
```

---

## Implementation Roadmap

### Phase 1: SOG-Research (Week 1-2)
- Implement basic re-ranking
- Integrate with search results
- Tune alpha parameters via A/B testing

### Phase 2: RWR-Path (Week 3-4)
- Build knowledge graph infrastructure
- Implement RWR algorithm
- Connect to research trajectory suggestions

### Phase 3: CTS (Week 5-6)
- Implement curiosity models
- Train on historical trajectories
- Integrate with tool orchestration

### Phase 4: Evaluation (Week 7-8)
- Deploy metrics
- Run user studies
- Iterate on parameters

---

## References

1. Zhang et al. (2012) - "Auralist: Introducing Serendipity into Music Recommendation"
2. Adamopoulos & Tuzhilin (2014) - "On Unexpectedness in Recommender Systems"
3. Kaminskas & Bridge (2016) - "Diversity, Serendipity, Novelty, and Coverage"
4. Ge et al. (2010) - "Beyond Accuracy: Evaluating Recommender Systems by Coverage and Serendipity"
5. Pathak et al. (2017) - "Curiosity-driven Exploration by Self-supervised Prediction"

---

*Generated for MiroThinker Deep Research Agent Integration*
