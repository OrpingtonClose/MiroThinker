# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Strands SDK plugins for the Miro research agent.

Plugins extend agent behavior via the hook system:
- KnowledgePlugin: cross-conversation knowledge persistence
  (BeforeInvocation / AfterToolCall / AfterInvocation hooks + @tool methods)
- ToolRouterPlugin: query-aware tool routing (BeforeInvocationEvent)
- ToolAuditPlugin: post-invocation tool usage verification (AfterInvocationEvent)
"""

from plugins.knowledge import KnowledgePlugin
from plugins.tool_audit import ToolAuditPlugin
from plugins.tool_router import ToolRouterPlugin

__all__ = ["KnowledgePlugin", "ToolRouterPlugin", "ToolAuditPlugin"]
