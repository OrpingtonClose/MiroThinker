"""Shared fixtures for API key integration tests.

Loads .env from strands-agent before tests run, making all
API keys available as environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_PATH = Path(__file__).resolve().parents[2] / "apps" / "strands-agent" / ".env"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no external dependency required."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def pytest_configure(config: pytest.Config) -> None:
    """Load .env at collection time so marks can check env vars."""
    _load_dotenv(_ENV_PATH)


@pytest.fixture(scope="session", autouse=True)
def _ensure_env():
    """Ensure .env is loaded for the entire test session."""
    _load_dotenv(_ENV_PATH)
