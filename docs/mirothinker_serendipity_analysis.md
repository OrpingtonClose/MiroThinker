# MiroThinker Codebase Architecture Analysis
## Serendipity Algorithm Integration Points

---

## 1. Executive Summary

MiroThinker is a deep research agent framework built on a **modular, configuration-driven architecture** with clear separation of concerns. The framework provides multiple promising integration points for serendipity algorithms across three main categories:

1. **Search Result Diversification** - Enhancing web search results with unexpected but relevant content
2. **Tool Call Exploration** - Discovering alternative tool sequences and combinations
3. **Research Path Discovery** - Identifying unexpected research directions during agent execution

---

## 2. Core Architecture Overview

### 2.1 Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiroThinker Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Config    │  │    Core     │  │      MCP Tools          │ │
│  │  (Hydra)    │  │  (Agent)    │  │   (miroflow-tools)      │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Orchestrator (orchestrator.py)              │   │
│  │  - Main execution loop (run_main_agent)                  │   │
│  │  - Sub-agent coordination (run_sub_agent)                │   │
│  │  - Tool execution management                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Tool      │  │   Answer    │  │      LLM Client         │ │
│  │  Executor   │  │  Generator  │  │   (base_client.py)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Critical Files for Serendipity Integration

| File | Purpose | Lines | Integration Potential |
|------|---------|-------|---------------------|
| `orchestrator.py` | Main execution orchestration | 1,202 | **HIGH** - Core decision loop |
| `tool_executor.py` | Tool call execution & processing | 356 | **HIGH** - Tool result processing |
| `answer_generator.py` | LLM response handling | 591 | **MEDIUM** - Response diversification |
| `pipeline.py` | Pipeline initialization | 217 | **MEDIUM** - Component injection |
| `searching_*.py` | Search MCP servers | ~200 each | **HIGH** - Result diversification |
| `serper_mcp_server.py` | Serper search integration | ~150 | **HIGH** - Search enhancement |

---

## 3. Integration Point Analysis

### 3.1 SEARCH RESULT DIVERSIFICATION

#### Location: MCP Search Servers
**Files:**
- `libs/miroflow-tools/src/miroflow_tools/mcp_servers/searching_google_mcp_server.py`
- `libs/miroflow-tools/src/miroflow_tools/mcp_servers/searching_sogou_mcp_server.py`
- `libs/miroflow-tools/src/miroflow_tools/mcp_servers/serper_mcp_server.py`

**Current Flow:**
```python
# Search → Results → LLM Processing
search_query → search_api() → raw_results → format_results() → LLM
```

**Serendipity Integration Points:**

1. **Post-Search Result Enhancement** (Recommended)
   - **Location:** After `search_api()` call, before `format_results()`
   - **Method:** Inject `SerendipityEnhancer` class
   - **Hook:** Add diversification layer to search results

```python
# Proposed Integration in serper_mcp_server.py
class SerendipityEnhancer:
    def enhance_results(self, original_results, query, diversity_factor=0.2):
        # 1. Analyze result embeddings
        # 2. Identify clusters
        # 3. Inject diverse/related content
        # 4. Return enhanced results with serendipity markers
        pass
```

2. **Query Expansion for Serendipity**
   - **Location:** Before search execution
   - **Method:** Expand query with related but unexpected terms
   - **Hook:** `expand_query_for_serendipity(original_query)`

#### Configuration Hook:
```yaml
# conf/agent/mirothinker_1.7_keep5_max200.yaml
main_agent:
  serendipity:
    enabled: true
    search_diversification: 0.2  # 20% diverse results
    unexpected_term_injection: true
```

---

### 3.2 TOOL CALL EXPLORATION

#### Location: Tool Executor & Orchestrator
**Files:**
- `apps/miroflow-agent/src/core/tool_executor.py`
- `apps/miroflow-agent/src/core/orchestrator.py` (lines 900-1100)

**Current Flow:**
```python
# LLM → Tool Selection → Execution → Result Processing
llm_response → parse_tool_calls() → execute_tool() → process_result()
```

**Serendipity Integration Points:**

1. **Alternative Tool Suggestion** (HIGH PRIORITY)
   - **Location:** `orchestrator.py` after tool call parsing, before execution
   - **Method:** `SerendipityToolExplorer` class
   - **Hook:** Lines 950-980 in `run_main_agent()` loop

```python
# Proposed Integration in orchestrator.py ~line 960
async def _execute_with_serendipity(self, tool_calls, message_history):
    # Original tool calls from LLM
    primary_tools = tool_calls
    
    # Serendipity: Suggest alternative tools
    explorer = SerendipityToolExplorer(self.tool_definitions)
    alternative_tools = explorer.suggest_alternatives(
        primary_tools, 
        message_history,
        exploration_rate=0.15
    )
    
    # Execute both primary and alternative (A/B style)
    # or inject as suggestions for LLM
```

2. **Tool Sequence Discovery**
   - **Location:** `tool_executor.py` in `post_process_tool_call_result()`
   - **Method:** Analyze tool result patterns to suggest next tools
   - **Hook:** After tool execution, before returning to LLM

```python
# In tool_executor.py
class ToolSequenceRecommender:
    def recommend_next_tools(self, current_tool, result, context):
        # Based on historical patterns, suggest unexpected but useful tools
        pass
```

#### Key Integration Method in `orchestrator.py`:
```python
# Around line 960 - Tool execution loop
for call in tool_calls:
    server_name = call["server_name"]
    tool_name = call["tool_name"]
    
    # SERENDIPITY INJECTION POINT
    # Add alternative tool exploration here
    if self.cfg.agent.get('serendipity', {}).get('tool_exploration', False):
        alternative = self.serendipity_explorer.get_alternative_tool(
            tool_name, message_history
        )
        if alternative:
            # Inject as suggestion or execute in parallel
            pass
```

---

### 3.3 RESEARCH PATH DISCOVERY

#### Location: Main Agent Loop & Answer Generator
**Files:**
- `apps/miroflow-agent/src/core/orchestrator.py` (lines 800-1000)
- `apps/miroflow-agent/src/core/answer_generator.py`

**Current Flow:**
```python
# Main Agent Loop (simplified)
while turn_count < max_turns:
    llm_response = await call_llm(message_history, tools)
    tool_calls = parse_tool_calls(llm_response)
    results = await execute_tools(tool_calls)
    message_history = update_history(message_history, results)
```

**Serendipity Integration Points:**

1. **Research Direction Injector** (HIGHEST PRIORITY)
   - **Location:** `orchestrator.py` in main loop, after each turn
   - **Method:** `ResearchPathExplorer` class
   - **Hook:** Lines 880-920, after message history update

```python
# Proposed Integration in orchestrator.py ~line 900
class ResearchPathExplorer:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.explored_paths = set()
    
    async def suggest_alternative_path(self, message_history, current_focus):
        # Analyze current research direction
        # Suggest unexpected but relevant tangent
        # Return: suggested_query, reasoning, confidence
        pass
    
    def should_explore_tangent(self, turn_count, total_turns):
        # Dynamic exploration rate based on progress
        exploration_rate = 0.1 + (turn_count / total_turns) * 0.2
        return random.random() < exploration_rate

# Integration in main loop:
if self.serendipity_explorer.should_explore_tangent(turn_count, max_turns):
    tangent = await self.serendipity_explorer.suggest_alternative_path(
        message_history, current_focus
    )
    if tangent.confidence > 0.6:
        # Inject tangent suggestion into message history
        message_history.append({
            "role": "system",
            "content": f"Consider exploring: {tangent.suggested_query}"
        })
```

2. **Context-Aware Serendipity Trigger**
   - **Location:** `answer_generator.py` in `handle_llm_call()`
   - **Method:** Detect when agent is stuck and inject new directions

```python
# In answer_generator.py
class SerendipityTrigger:
    def detect_stagnation(self, message_history, recent_turns=5):
        # Detect repetitive patterns
        # Trigger serendipitous exploration when stuck
        pass
```

---

## 4. Detailed Integration Architecture

### 4.1 Proposed Serendipity Module Structure

```
libs/
  miroflow-tools/
    src/
      miroflow_tools/
        serendipity/                    # NEW MODULE
          __init__.py
          base.py                       # Abstract base classes
          search_diversifier.py         # Search result enhancement
          tool_explorer.py              # Alternative tool discovery
          path_explorer.py              # Research path suggestions
          embedding_utils.py            # Vector similarity utilities
          config.py                     # Serendipity configuration
```

### 4.2 Configuration Schema

```yaml
# conf/agent/mirothinker_1.7_keep5_max200.yaml
main_agent:
  tools:
    - search_and_scrape_webpage
    - jina_scrape_llm_summary
    - tool-python

  # Serendipity Configuration
  serendipity:
    enabled: true
    
    # Search Diversification
    search:
      enabled: true
      diversity_factor: 0.2          # 20% diverse results
      max_diverse_results: 3
      unexpected_term_boost: 0.15
    
    # Tool Exploration
    tool_exploration:
      enabled: true
      exploration_rate: 0.15         # 15% chance to suggest alt tool
      max_alternatives_per_call: 2
      parallel_execution: false      # Execute alt tools in parallel
    
    # Research Path Discovery
    path_discovery:
      enabled: true
      trigger_stagnation_threshold: 5  # turns without progress
      tangent_injection_rate: 0.1
      max_tangent_depth: 2
      
    # Embedding Model (for similarity calculations)
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      cache_dir: "./cache/embeddings"
```

---

## 5. Specific Integration Points by File

### 5.1 `orchestrator.py` - Main Integration Hub

**Line 800-1000: Main Agent Loop**
- **Integration Type:** Research Path Discovery
- **Hook Location:** After each turn completion, before next LLM call
- **Code Pattern:**
```python
# After line ~900 (message history update)
if self.cfg.agent.serendipity.path_discovery.enabled:
    if self.path_explorer.should_suggest_tangent(turn_count, message_history):
        tangent = await self.path_explorer.generate_tangent(message_history)
        message_history = self._inject_tangent(message_history, tangent)
```

**Line 950-980: Tool Execution Loop**
- **Integration Type:** Tool Call Exploration
- **Hook Location:** Before tool execution
- **Code Pattern:**
```python
# Around line 960
if self.cfg.agent.serendipity.tool_exploration.enabled:
    alt_tools = self.tool_explorer.suggest_alternatives(tool_calls)
    if alt_tools:
        tool_calls = self._merge_with_alternatives(tool_calls, alt_tools)
```

### 5.2 `tool_executor.py` - Result Processing

**Line 150-250: `post_process_tool_call_result()`**
- **Integration Type:** Result Enhancement
- **Hook Location:** After result processing, before return
- **Code Pattern:**
```python
def post_process_tool_call_result(self, tool_name, tool_result):
    # Existing processing...
    
    # Serendipity enhancement
    if self.serendipity_config.search.enabled and 'search' in tool_name:
        tool_result = self.search_diversifier.enhance(tool_result)
    
    return tool_result
```

### 5.3 MCP Search Servers

**`serper_mcp_server.py` - Line 50-150**
- **Integration Type:** Search Result Diversification
- **Hook Location:** After API call, before formatting
- **Code Pattern:**
```python
async def search_tool(query: str) -> str:
    raw_results = await call_serper_api(query)
    
    # Serendipity injection
    if serendipity_config.enabled:
        enhanced_results = await serendipity_enhancer.enhance(
            raw_results, query
        )
        return format_results(enhanced_results)
    
    return format_results(raw_results)
```

---

## 6. Implementation Roadmap

### Phase 1: Search Diversification (Week 1-2)
1. Create `serendipity/` module structure
2. Implement `SearchDiversifier` class
3. Integrate into `serper_mcp_server.py`
4. Add configuration schema
5. Test with sample queries

### Phase 2: Tool Exploration (Week 3-4)
1. Implement `ToolExplorer` class
2. Create tool similarity matrix
3. Integrate into `orchestrator.py` tool loop
4. Add A/B testing framework for tool suggestions
5. Measure impact on task completion

### Phase 3: Research Path Discovery (Week 5-6)
1. Implement `PathExplorer` class
2. Create stagnation detection algorithm
3. Integrate into main agent loop
4. Add tangent depth tracking
5. Evaluate exploration quality

### Phase 4: Integration & Optimization (Week 7-8)
1. Unify all serendipity components
2. Add embedding-based similarity
3. Implement adaptive exploration rates
4. Add comprehensive logging
5. Performance optimization

---

## 7. Key Classes and Interfaces

### 7.1 Abstract Base Classes

```python
# libs/miroflow-tools/src/miroflow_tools/serendipity/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SerendipitySuggestion:
    content: Any
    confidence: float
    reasoning: str
    source: str
    metadata: Dict[str, Any]

class BaseSerendipityProvider(ABC):
    @abstractmethod
    async def generate_suggestions(
        self, 
        context: Dict[str, Any],
        num_suggestions: int = 1
    ) -> List[SerendipitySuggestion]:
        pass
    
    @abstractmethod
    def should_trigger(self, context: Dict[str, Any]) -> bool:
        pass

class SearchDiversifier(BaseSerendipityProvider):
    @abstractmethod
    async def diversify_results(
        self,
        original_results: List[Dict],
        query: str,
        diversity_factor: float
    ) -> List[Dict]:
        pass

class ToolExplorer(BaseSerendipityProvider):
    @abstractmethod
    def suggest_alternative_tools(
        self,
        current_tool: str,
        tool_definitions: List[Dict],
        context: Dict[str, Any]
    ) -> List[SerendipitySuggestion]:
        pass

class PathExplorer(BaseSerendipityProvider):
    @abstractmethod
    async def suggest_research_tangent(
        self,
        message_history: List[Dict],
        current_focus: str
    ) -> Optional[SerendipitySuggestion]:
        pass
```

---

## 8. Configuration-Driven Integration

The framework's Hydra-based configuration system makes serendipity integration straightforward:

```python
# apps/miroflow-agent/src/config/settings.py
# Add serendipity configuration

from dataclasses import dataclass
from typing import Optional

@dataclass
class SerendipityConfig:
    enabled: bool = False
    
    # Search diversification
    search_diversity_factor: float = 0.2
    search_max_diverse: int = 3
    
    # Tool exploration
    tool_exploration_rate: float = 0.15
    tool_max_alternatives: int = 2
    
    # Path discovery
    path_stagnation_threshold: int = 5
    path_tangent_rate: float = 0.1
```

---

## 9. Monitoring and Evaluation

### 9.1 Metrics to Track

1. **Search Diversification**
   - Diversity score of results (embedding-based)
   - Click-through rate on diverse results
   - Task completion with vs. without diversification

2. **Tool Exploration**
   - Alternative tool acceptance rate
   - Success rate of serendipitous tool calls
   - Impact on task completion time

3. **Research Path Discovery**
   - Tangent exploration rate
   - Tangent success rate (leads to useful info)
   - Overall task success improvement

### 9.2 Logging Integration

```python
# Leverage existing TaskLog system
self.task_log.log_step(
    "serendipity",
    "Search Diversification",
    f"Added {num_diverse} diverse results to query: {query}"
)
```

---

## 10. Summary of Integration Points

| Priority | Component | File | Line | Effort | Impact |
|----------|-----------|------|------|--------|--------|
| **P0** | Search Diversification | `serper_mcp_server.py` | 50-150 | Low | High |
| **P0** | Tool Exploration | `orchestrator.py` | 950-980 | Medium | High |
| **P1** | Research Path Discovery | `orchestrator.py` | 880-920 | Medium | High |
| **P1** | Result Enhancement | `tool_executor.py` | 150-250 | Low | Medium |
| **P2** | Stagnation Detection | `answer_generator.py` | 200-300 | Medium | Medium |
| **P2** | Alternative Tool Execution | `orchestrator.py` | 980-1050 | High | High |

---

## 11. Conclusion

MiroThinker's modular architecture with clear separation between:
- **Configuration** (Hydra YAML files)
- **Orchestration** (orchestrator.py)
- **Tool Execution** (tool_executor.py + MCP servers)
- **LLM Integration** (answer_generator.py)

...provides **ideal integration points** for serendipity algorithms. The configuration-driven approach allows for easy experimentation with different serendipity strategies without code changes.

**Recommended Starting Point:**
Begin with **Search Diversification** in `serper_mcp_server.py` as it:
1. Has the lowest implementation complexity
2. Provides immediate visual feedback
3. Can be A/B tested easily
4. Builds foundation for other serendipity features

---

## Appendix: File Paths Reference

### Core Agent Files
- `apps/miroflow-agent/src/core/orchestrator.py` (1,202 lines)
- `apps/miroflow-agent/src/core/tool_executor.py` (356 lines)
- `apps/miroflow-agent/src/core/answer_generator.py` (591 lines)
- `apps/miroflow-agent/src/core/pipeline.py` (217 lines)
- `apps/miroflow-agent/src/core/stream_handler.py`

### Configuration Files
- `apps/miroflow-agent/conf/config.yaml`
- `apps/miroflow-agent/conf/agent/default.yaml`
- `apps/miroflow-agent/conf/agent/mirothinker_1.7_keep5_max200.yaml`
- `apps/miroflow-agent/conf/agent/mirothinker_v1.5_keep5_max200.yaml`

### MCP Tool Servers
- `libs/miroflow-tools/src/miroflow_tools/mcp_servers/serper_mcp_server.py`
- `libs/miroflow-tools/src/miroflow_tools/mcp_servers/searching_google_mcp_server.py`
- `libs/miroflow-tools/src/miroflow_tools/mcp_servers/searching_sogou_mcp_server.py`
- `libs/miroflow-tools/src/miroflow_tools/manager.py`

### Supporting Modules
- `apps/miroflow-agent/src/llm/base_client.py`
- `apps/miroflow-agent/src/io/output_formatter.py`
- `apps/miroflow-agent/src/config/settings.py`
- `apps/miroflow-agent/src/utils/prompt_utils.py`

---

*Analysis generated for MiroThinker codebase*
*Repository: https://github.com/OrpingtonClose/MiroThinker*
