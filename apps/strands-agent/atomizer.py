"""LLM-based text atomisation via local vLLM endpoint.

Replaces Flock's in-SQL llm_complete for decomposing free text into
atomic factual claims. Each atom is a dict ready for ConditionStore.admit().

Uses the same localhost-only guard as swarm_bridge.py — no remote APIs.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ATOMIZER_MODEL = os.environ.get(
    "ATOMIZER_MODEL",
    os.environ.get("SWARM_WORKER_MODEL", "gemma-4-uncensored"),
)

# Regex patterns for parsing atomised output
_URL_RE = re.compile(r"\[(?:URL|url):\s*(https?://\S+)\]")
_BARE_URL_RE = re.compile(r"(https?://\S+)")
_CONF_RE = re.compile(r"\(confidence[=:]\s*([\d.]+)\)")


ATOMISATION_PROMPT = """\
You are a research analyst. Extract atomic factual claims from the text below.

{query_clause}
Rules:
- Extract ONLY facts relevant to the research query (skip off-topic, \
promotional, or meta-commentary material)
- One fact per FACT: line
- Preserve ALL specific data: names, numbers, dates, prices, URLs
- If a URL is associated with a fact, append it as [URL: ...] at the end
- If the text expresses confidence/uncertainty, append (confidence=X.X) \
where X.X is 0.0-1.0
- If the text is a single atomic fact already, return one FACT: line

Output format (follow exactly):
FACT: [statement] [URL: ...] (confidence=X.X)
FACT: [statement]

Text to analyse:
{text}"""


async def atomize_text(
    text: str,
    user_query: str = "",
    model: str | None = None,
) -> list[dict]:
    """Decompose free text into atomic factual claims via LLM.

    Each atom is a dict with: fact, source_url, confidence, angle.
    Uses local Ollama endpoint (same as swarm workers).

    Args:
        text: Raw text to atomise.
        user_query: Research query for relevance context.
        model: LLM model to use. Defaults to ATOMIZER_MODEL.

    Returns:
        List of atom dicts ready for ConditionStore.admit().
    """
    if not text or not text.strip():
        return []

    query_clause = f"Research query: {user_query}\n\n" if user_query else ""
    prompt = ATOMISATION_PROMPT.format(query_clause=query_clause, text=text)

    raw_output = await _local_complete(
        prompt, model or _ATOMIZER_MODEL,
    )

    if not raw_output or not raw_output.strip():
        # Fallback: treat entire text as a single fact
        return [{"fact": text.strip(), "source_url": "", "confidence": 0.5}]

    return _parse_atoms(raw_output)


def _parse_atoms(raw_output: str) -> list[dict]:
    """Parse FACT: lines from LLM output into atom dicts."""
    atoms: list[dict] = []

    for line in raw_output.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Skip non-FACT lines (RELATES, comments, etc.)
        if stripped.upper().startswith("RELATES:"):
            continue

        # Strip FACT: prefix if present
        if stripped.upper().startswith("FACT:"):
            stripped = stripped[5:].strip()

        # Strip bullet/number prefixes
        stripped = re.sub(
            r"^\s*(?:\d+[.)\]]\s*|[-*\u2022]\s+)", "", stripped,
        ).strip()
        if not stripped:
            continue

        # Reject junk atoms
        if _is_junk(stripped):
            continue

        # Extract URL
        source_url = ""
        url_match = _URL_RE.search(stripped)
        if url_match:
            source_url = url_match.group(1)
            stripped = _URL_RE.sub("", stripped).strip()
        else:
            bare_match = _BARE_URL_RE.search(stripped)
            if bare_match:
                source_url = bare_match.group(1).rstrip(".,;:")

        # Extract confidence
        confidence = 0.5
        conf_match = _CONF_RE.search(stripped)
        if conf_match:
            try:
                confidence = max(0.0, min(1.0, float(conf_match.group(1))))
            except (ValueError, TypeError):
                confidence = 0.5
            stripped = _CONF_RE.sub("", stripped).strip()

        if not stripped:
            continue

        atoms.append({
            "fact": stripped,
            "source_url": source_url,
            "confidence": confidence,
        })

    return atoms


def _is_junk(text: str) -> bool:
    """Reject non-factual atoms (headers, labels, meta-commentary)."""
    lower = text.lower().strip()
    if len(lower) < 10:
        return True
    junk_prefixes = [
        "the text", "this text", "the above", "summary:", "note:",
        "conclusion:", "in summary", "overall,", "key takeaway",
        "no relevant", "no facts", "none found",
    ]
    return any(lower.startswith(p) for p in junk_prefixes)


def _get_atomizer_base() -> str:
    """Resolve the atomizer API base URL, defaulting to localhost vLLM."""
    return os.environ.get("SWARM_API_BASE", "http://localhost:8000/v1")


def _assert_localhost(url: str) -> None:
    """Guard: only allow localhost URLs for model endpoints."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if hostname not in ("localhost", "127.0.0.1", "::1"):
        msg = f"Model endpoint must be localhost, got: {url}"
        raise RuntimeError(msg)


async def _local_complete(
    prompt: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Call localhost vLLM for atomisation. Low temperature for precision."""
    base = _get_atomizer_base()
    _assert_localhost(base)

    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Extract the atomic facts now."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("model=<%s> | atomisation LLM call failed", model)
            return ""
