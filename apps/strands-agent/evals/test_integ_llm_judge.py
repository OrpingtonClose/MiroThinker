# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""LLM-as-judge evals for agent output quality.

Uses a separate LLM call (Gemma 4) to score the agent's output on
multiple quality dimensions: accuracy, completeness, clarity,
specificity, and source usage.

This replaces weak heuristic checks (keyword presence, length checks)
with semantic evaluation that can detect whether the response is
*actually helpful* and *contains substantive information*.

Requires: VENICE_API_KEY

Usage::

    VENICE_API_KEY=... pytest evals/test_integ_llm_judge.py -v -m integ
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
import pytest

pytestmark = pytest.mark.integ

logger = logging.getLogger(__name__)

# ── Judge configuration ───────────────────────────────────────────────

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google-gemma-4-26b-a4b-it")
JUDGE_API_BASE = os.environ.get(
    "JUDGE_API_BASE",
    os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1"),
)
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY",
    os.environ.get("VENICE_API_KEY", ""),
)

VENICE_API_KEY = os.environ.get("VENICE_API_KEY", "")
VENICE_API_BASE = os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic")

# Minimum acceptable scores (out of 10)
MIN_ACCURACY = 5
MIN_COMPLETENESS = 5
MIN_CLARITY = 6
MIN_SPECIFICITY = 5
MIN_SOURCE_USAGE = 4

JUDGE_SYSTEM_PROMPT = """\
You are a quality evaluator for AI research assistant responses. You will receive:
1. QUERY: the user's original question
2. RESPONSE: the agent's answer

Score the RESPONSE on these 5 dimensions (1-10 each):

- accuracy: Does the response contain factually correct information? \
(1=major errors, 10=fully accurate)
- completeness: Does it address all aspects of the query? \
(1=barely addresses it, 10=thorough coverage)
- clarity: Is the response well-written, easy to understand, and logically \
structured? (1=confusing/incoherent, 10=crystal clear)
- specificity: Does it include concrete details — names, numbers, dates, \
technical terms — rather than vague generalities? (1=completely generic, 10=rich details)
- source_usage: Does the response cite or reference specific sources, studies, \
or data points? (1=no sources at all, 10=well-sourced with citations)

Respond with ONLY a JSON object, no other text:
{"accuracy": N, "completeness": N, "clarity": N, "specificity": N, "source_usage": N, "notes": "brief explanation"}\
"""

JUDGE_USER_TEMPLATE = """\
QUERY:
{query}

RESPONSE:
{response}

Score the response (JSON only):"""


# ── Test queries ─────────────────────────────────────────────────────

SAMPLES = {
    "mesh_networking": {
        "query": (
            "Explain how mesh networking protocols like Tor, I2P, and Nym "
            "differ in their approach to censorship circumvention."
        ),
        "expected_specifics": ["tor", "i2p", "nym", "mesh", "censorship"],
    },
    "quantum_crypto": {
        "query": (
            "What are the main threats that quantum computing poses to "
            "current cryptographic standards, and what alternatives are "
            "being developed?"
        ),
        "expected_specifics": ["quantum", "rsa", "post-quantum", "nist"],
    },
    "simple_factual": {
        "query": "What is the boiling point of water at sea level in Celsius?",
        "expected_specifics": ["100"],
    },
    "web_protocols": {
        "query": (
            "Compare HTTP/2 and HTTP/3 in terms of performance, security, "
            "and adoption."
        ),
        "expected_specifics": ["http/2", "http/3", "quic", "tcp"],
    },
}


# ── Helpers ──────────────────────────────────────────────────────────


def _build_agent():
    """Build a Venice-backed agent for judge evals."""
    from strands import Agent
    from strands.models.openai import OpenAIModel

    model = OpenAIModel(
        client_args={
            "api_key": VENICE_API_KEY,
            "base_url": VENICE_API_BASE,
        },
        model_id=VENICE_MODEL,
        params={
            "extra_body": {
                "venice_parameters": {"include_venice_system_prompt": False},
                "reasoning": {"effort": "high"},
            }
        },
    )
    return Agent(
        model=model,
        callback_handler=None,
        system_prompt=(
            "You are a research assistant. Answer questions thoroughly "
            "and accurately with specific details."
        ),
    )


def _call_judge(query: str, response: str) -> dict[str, int | str]:
    """Call the judge LLM and return parsed scores."""
    if not JUDGE_API_KEY:
        pytest.skip("JUDGE_API_KEY / VENICE_API_KEY not set")

    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                query=query, response=response
            )},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
        "stream": False,
        "venice_parameters": {"include_venice_system_prompt": False},
        "reasoning": {"effort": "none"},
    }

    resp = httpx.post(
        f"{JUDGE_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {JUDGE_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    msg = data.get("choices", [{}])[0].get("message", {})
    text = msg.get("content", "") or msg.get("reasoning_content", "") or ""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    scores = json.loads(text)
    logger.info("Judge scores: %s", scores)
    return scores


def _query_and_judge(query: str) -> tuple[str, dict[str, int | str]]:
    """Run a query through the agent and score with the judge."""
    if not VENICE_API_KEY:
        pytest.skip("VENICE_API_KEY not set")

    agent = _build_agent()
    result = agent(query)
    response = str(result)
    time.sleep(2)  # rate limit buffer between agent and judge calls
    scores = _call_judge(query, response)
    return response, scores


# ── Per-sample quality tests ─────────────────────────────────────────


class TestJudgeResearchQuality:
    """Judge-scored evals for research-style queries."""

    def test_mesh_networking_quality(self) -> None:
        sample = SAMPLES["mesh_networking"]
        response, scores = _query_and_judge(sample["query"])

        assert scores["accuracy"] >= MIN_ACCURACY, (
            f"accuracy={scores['accuracy']} < {MIN_ACCURACY}"
        )
        assert scores["clarity"] >= MIN_CLARITY, (
            f"clarity={scores['clarity']} < {MIN_CLARITY}"
        )
        assert scores["specificity"] >= MIN_SPECIFICITY, (
            f"specificity={scores['specificity']} < {MIN_SPECIFICITY}"
        )

    def test_quantum_crypto_quality(self) -> None:
        time.sleep(2)
        sample = SAMPLES["quantum_crypto"]
        response, scores = _query_and_judge(sample["query"])

        assert scores["accuracy"] >= MIN_ACCURACY
        assert scores["completeness"] >= MIN_COMPLETENESS
        assert scores["specificity"] >= MIN_SPECIFICITY


class TestJudgeSimpleQuality:
    """Judge-scored evals for simple factual queries."""

    def test_simple_factual_quality(self) -> None:
        time.sleep(2)
        sample = SAMPLES["simple_factual"]
        response, scores = _query_and_judge(sample["query"])

        assert scores["accuracy"] >= MIN_ACCURACY
        assert scores["clarity"] >= MIN_CLARITY

    def test_web_protocols_quality(self) -> None:
        time.sleep(2)
        sample = SAMPLES["web_protocols"]
        response, scores = _query_and_judge(sample["query"])

        assert scores["accuracy"] >= MIN_ACCURACY
        assert scores["specificity"] >= MIN_SPECIFICITY
        assert scores["completeness"] >= MIN_COMPLETENESS


class TestJudgeAverageScores:
    """Aggregate quality thresholds across all samples."""

    def test_average_accuracy_above_threshold(self) -> None:
        time.sleep(2)
        all_scores: list[int] = []
        for key, sample in SAMPLES.items():
            _, scores = _query_and_judge(sample["query"])
            all_scores.append(scores["accuracy"])
            time.sleep(2)

        avg = sum(all_scores) / len(all_scores)
        logger.info("Average accuracy across %d samples: %.1f", len(all_scores), avg)
        assert avg >= MIN_ACCURACY, f"Average accuracy {avg:.1f} < {MIN_ACCURACY}"

    def test_average_clarity_above_threshold(self) -> None:
        time.sleep(2)
        all_scores: list[int] = []
        for key, sample in SAMPLES.items():
            _, scores = _query_and_judge(sample["query"])
            all_scores.append(scores["clarity"])
            time.sleep(2)

        avg = sum(all_scores) / len(all_scores)
        logger.info("Average clarity across %d samples: %.1f", len(all_scores), avg)
        assert avg >= MIN_CLARITY, f"Average clarity {avg:.1f} < {MIN_CLARITY}"
