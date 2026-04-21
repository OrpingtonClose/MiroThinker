# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Strands SDK plugins for the Miro research agent.

Plugins extend agent behavior via the hook system:
- KnowledgePlugin: cross-conversation knowledge persistence
  (BeforeInvocation / AfterToolCall / AfterInvocation hooks + @tool methods)
"""

from plugins.knowledge import KnowledgePlugin

__all__ = ["KnowledgePlugin"]
