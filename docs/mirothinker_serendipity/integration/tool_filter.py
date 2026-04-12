"""
Serendipity Tool Filter for MCP Tool Calls.

Filters and enhances tool calls with serendipity by potentially
injecting unexpected but potentially useful tools into the sequence.
"""

from typing import Dict, Any, List, Optional
import random


class SerendipityToolFilter:
    """
    Filters and enhances tool calls with serendipity.
    
    Can inject unexpected but potentially useful tools
    into the tool call sequence based on exploration strategy.
    
    Exploration Strategies:
        - epsilon_greedy: With probability epsilon, inject serendipitous tool
        - ucb: Upper Confidence Bound for exploration/exploitation tradeoff
        - thompson_sampling: Bayesian approach to exploration
    
    Example:
        >>> filter = SerendipityToolFilter(config)
        >>> enhanced_tools = filter.filter_tools(
        ...     available_tools,
        ...     selected_tools,
        ...     context
        ... )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tool filter.
        
        Args:
            config: Configuration with keys:
                - enabled: Master toggle
                - exploration: Exploration settings
                    - strategy: Exploration strategy
                    - epsilon: Epsilon for epsilon-greedy
                    - temperature: Temperature for sampling
        """
        self.config = config
        exploration_config = config.get('exploration', {})
        self.exploration_strategy = exploration_config.get('strategy', 'epsilon_greedy')
        self.epsilon = exploration_config.get('epsilon', 0.2)
        self.temperature = exploration_config.get('temperature', 0.5)
        self.tool_history: Dict[str, int] = {}
    
    def filter_tools(
        self,
        available_tools: List[Dict[str, Any]],
        selected_tools: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Filter and potentially enhance tool selection.
        
        Args:
            available_tools: All available tools with metadata
            selected_tools: Tools selected by the agent
            context: Task context
            
        Returns:
            Potentially enhanced tool list
        """
        if not self.config.get('enabled', False):
            return selected_tools
        
        # Update tool usage history
        for tool in selected_tools:
            self.tool_history[tool] = self.tool_history.get(tool, 0) + 1
        
        # Apply exploration strategy
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_selection(
                available_tools, selected_tools, context
            )
        elif self.exploration_strategy == 'ucb':
            return self._ucb_selection(
                available_tools, selected_tools, context
            )
        elif self.exploration_strategy == 'thompson_sampling':
            return self._thompson_sampling_selection(
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
        
        With probability epsilon, inject a serendipitous tool
        that hasn't been used recently.
        
        Args:
            available_tools: All available tools
            selected_tools: Currently selected tools
            context: Task context
            
        Returns:
            Enhanced tool list
        """
        # With probability (1-epsilon), return original selection
        if random.random() > self.epsilon:
            return selected_tools
        
        # Find tools not in selected set
        available_ids = {t['id'] for t in available_tools}
        selected_set = set(selected_tools)
        unexplored = available_ids - selected_set
        
        if not unexplored:
            return selected_tools
        
        # Prefer less frequently used tools
        unexplored_list = list(unexplored)
        usage_counts = [self.tool_history.get(t, 0) for t in unexplored_list]
        
        # Inverse weighting - less used = higher probability
        weights = [1.0 / (1 + count) for count in usage_counts]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select serendipitous tool
        serendipitous_tool = random.choices(
            unexplored_list,
            weights=probabilities,
            k=1
        )[0]
        
        # Inject at appropriate position
        enhanced = selected_tools.copy()
        
        # For exploratory tasks, inject early
        # For focused tasks, inject later
        task_type = context.get('task_type', 'general')
        if task_type == 'exploration':
            insert_pos = min(1, len(enhanced))
        else:
            insert_pos = len(enhanced)
        
        enhanced.insert(insert_pos, serendipitous_tool)
        
        return enhanced
    
    def _ucb_selection(
        self,
        available_tools: List[Dict[str, Any]],
        selected_tools: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Upper Confidence Bound selection.
        
        Balances exploration and exploitation using UCB formula.
        
        Args:
            available_tools: All available tools
            selected_tools: Currently selected tools
            context: Task context
            
        Returns:
            Enhanced tool list
        """
        # UCB implementation would go here
        # For now, fall back to epsilon-greedy
        return self._epsilon_greedy_selection(available_tools, selected_tools, context)
    
    def _thompson_sampling_selection(
        self,
        available_tools: List[Dict[str, Any]],
        selected_tools: List[str],
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Thompson sampling selection.
        
        Bayesian approach to exploration/exploitation.
        
        Args:
            available_tools: All available tools
            selected_tools: Currently selected tools
            context: Task context
            
        Returns:
            Enhanced tool list
        """
        # Thompson sampling implementation would go here
        # For now, fall back to epsilon-greedy
        return self._epsilon_greedy_selection(available_tools, selected_tools, context)
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics on tool usage.
        
        Returns:
            Dictionary with tool usage statistics
        """
        if not self.tool_history:
            return {'total_uses': 0, 'unique_tools': 0, 'most_used': None}
        
        total_uses = sum(self.tool_history.values())
        most_used = max(self.tool_history.items(), key=lambda x: x[1])
        
        return {
            'total_uses': total_uses,
            'unique_tools': len(self.tool_history),
            'most_used': {'tool': most_used[0], 'count': most_used[1]},
            'usage_distribution': self.tool_history
        }
