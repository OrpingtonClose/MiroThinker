# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Input handler utilities.

Attempts to import ``process_input`` from the shared miroflow-tools library.
Falls back to a minimal implementation that returns the task description
unchanged if the import is unavailable.
"""

from __future__ import annotations

from typing import Tuple

try:
    from miroflow_tools.mcp_servers.utils import process_input  # noqa: F401
except ImportError:

    def process_input(
        task_description: str, task_file_name: str = ""
    ) -> Tuple[str, str]:
        """Minimal fallback: return the task description unchanged."""
        return task_description, task_description
