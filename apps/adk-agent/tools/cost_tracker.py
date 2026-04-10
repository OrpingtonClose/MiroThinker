# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Cost tracker for deep research tools.

Tracks per-session and monthly spending on commercial deep research
services (Perplexity, Grok, Tavily).  Enforces configurable budget
limits so runaway research loops don't exhaust API credits.

Cost data is persisted to a JSONL ledger at ``$FINDINGS_DIR/cost_ledger.jsonl``
so monthly totals survive across sessions.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FINDINGS_DIR = Path(os.environ.get(
    "FINDINGS_DIR", os.path.join(os.path.expanduser("~"), ".mirothinker")
))

SESSION_BUDGET = float(
    os.environ.get("DEEP_RESEARCH_SESSION_BUDGET", "10.0")
)
MONTHLY_BUDGET = float(
    os.environ.get("DEEP_RESEARCH_MONTHLY_BUDGET", "200.0")
)

# Estimated costs per call (conservative upper bounds)
ESTIMATED_COSTS: dict[str, float] = {
    "perplexity": 5.00,
    "grok": 2.00,
    "tavily": 0.50,
}


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Track API costs per session and monthly, with budget enforcement.

    Thread-safe via ``threading.Lock`` (the before_tool_callback is sync).
    """

    def __init__(self) -> None:
        self._session_total: float = 0.0
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._ledger_path = _FINDINGS_DIR / "cost_ledger.jsonl"
        try:
            _FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create cost log dir: %s", exc)

    def reset_session(self) -> None:
        """Reset session counters for a new pipeline run.

        Call this at pipeline init so that each research session starts
        with a fresh budget.  Monthly totals (from the JSONL ledger)
        are unaffected.
        """
        with self._lock:
            self._session_total = 0.0
            self._entries = []
        logger.info("Cost tracker session reset")

    def record_cost(
        self,
        provider: str,
        cost: float,
        query: str,
    ) -> None:
        """Record an API call and its estimated cost."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "query": query[:200],
            "estimated_cost_usd": round(cost, 4),
            "session_total_usd": 0.0,
        }
        with self._lock:
            self._session_total += cost
            entry["session_total_usd"] = round(self._session_total, 4)
            self._entries.append(entry)

        # Append to JSONL ledger
        try:
            with open(self._ledger_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Failed to write cost ledger: %s", exc)

    def check_budget(self, provider: str) -> Optional[str]:
        """Check if budget allows another API call.

        Returns a warning string if budget is exceeded, None if OK.
        This is called from the synchronous before_tool_callback.
        """
        estimated = ESTIMATED_COSTS.get(provider, 1.0)

        with self._lock:
            if self._session_total + estimated > SESSION_BUDGET:
                return (
                    f"[BUDGET EXCEEDED] Session budget of "
                    f"${SESSION_BUDGET:.2f} would be exceeded "
                    f"(current: ${self._session_total:.2f}, "
                    f"estimated call: ${estimated:.2f}). "
                    f"Use regular search tools (Brave, Exa, Kagi) instead."
                )

        # Check monthly budget from JSONL ledger
        try:
            monthly_total = self._get_monthly_total()
            if monthly_total + estimated > MONTHLY_BUDGET:
                return (
                    f"[BUDGET EXCEEDED] Monthly budget of "
                    f"${MONTHLY_BUDGET:.2f} would be exceeded "
                    f"(current: ${monthly_total:.2f}). "
                    f"Deep research tools disabled until next month."
                )
        except Exception:
            pass  # If we can't read the ledger, allow the call

        return None

    def _get_monthly_total(self) -> float:
        """Sum costs from this month's JSONL ledger entries."""
        if not self._ledger_path.exists():
            return 0.0

        month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
        total = 0.0
        try:
            with open(self._ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", "")
                        if ts.startswith(month_prefix):
                            total += entry.get("estimated_cost_usd", 0.0)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return total

    def get_session_stats(self) -> dict[str, Any]:
        """Return session cost statistics."""
        with self._lock:
            return {
                "session_total_usd": round(self._session_total, 4),
                "session_budget_usd": SESSION_BUDGET,
                "session_remaining_usd": round(
                    max(0, SESSION_BUDGET - self._session_total), 4
                ),
                "call_count": len(self._entries),
                "last_entries": self._entries[-5:],
            }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Return the global cost tracker instance."""
    return _cost_tracker


def reset_session_tracker() -> None:
    """Reset the global cost tracker for a new session."""
    _cost_tracker.reset_session()
