# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Input handler re-export.

Imports process_input from the shared miroflow-agent library so that
the ADK agent can process task inputs identically.
"""

from miroflow_tools.mcp_servers.utils import process_input  # noqa: F401

# If the shared library doesn't expose process_input directly,
# provide a minimal fallback that just returns the task description unchanged.
try:
    from miroflow_tools.mcp_servers.utils import process_input  # noqa: F811
except ImportError:

    def process_input(task_description: str, task_file_name: str = ""):
        """Minimal fallback: return the task description unchanged."""
        return task_description, task_description
