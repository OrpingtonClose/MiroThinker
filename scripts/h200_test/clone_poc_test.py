#!/usr/bin/env python3
"""Stage 2 PoC: Clone-scored relevance vs keyword-scored relevance.

This is the critical fork test from SWARM_WAVE_ARCHITECTURE.md § Stage 2.

Goal: Answer ONE question — does a cloned worker context produce better
relevance judgments than keyword RAG?

Protocol:
    1. Run 1 tool-free worker through 1 wave on the test corpus
    2. Clone its conversation (system prompt + data package + response)
    3. Register the clone with the session proxy
    4. Generate 50 test relevance queries from the corpus
    5. For each query, get relevance judgments from:
       a) Clone (via session proxy — prepends worker context)
       b) Keyword scoring (rag.py — term frequency overlap)
    6. Human-judge ground truth OR use a separate LLM as judge
    7. Compare: clone precision vs keyword precision

Decision criteria (from architecture doc):
    - IF clone precision > keyword precision by ≥15% → proceed to Stage 3
    - IF clone precision ≤ keyword precision → architecture's core bet is wrong
    - IF prefix caching doesn't work → clone pattern too expensive

Usage:
    export OPENROUTER_API_KEY=sk-...
    python scripts/h200_test/clone_poc_test.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# Add paths for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "swarm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "apps" / "strands-agent"))

from swarm.agent_worker import run_tool_free_worker
from swarm.data_package import DataPackage
from swarm.rag import extract_concepts, score_relevance
from swarm.session_proxy import register_clone, proxy_chat_completion, set_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

CORPUS_PATH = Path(__file__).parent / "test_corpus.txt"
TEST_QUERY = (
    "Milos Sarcev insulin protocol for bodybuilding in the context of "
    "taking insulin, hgh and trenbolone based cycles with full complexity "
    "breakdown between all food related nutrients, supplements and PEDs"
)
WORKER_ANGLE = "insulin timing and nutrient partitioning"
NUM_TEST_QUERIES = 50


@dataclass
class RelevanceJudgment:
    """A single relevance judgment for one query-finding pair."""

    query: str
    finding: str
    clone_score: float = 0.0
    keyword_score: float = 0.0
    judge_relevant: bool = False  # ground truth from LLM judge


@dataclass
class PoCResults:
    """Aggregated results from the clone PoC test."""

    worker_angle: str = ""
    worker_output_chars: int = 0
    worker_elapsed_s: float = 0.0
    clone_messages: int = 0
    num_queries: int = 0
    judgments: list[RelevanceJudgment] = field(default_factory=list)

    # Aggregated metrics
    clone_precision: float = 0.0
    keyword_precision: float = 0.0
    clone_recall: float = 0.0
    keyword_recall: float = 0.0
    clone_wins: int = 0
    keyword_wins: int = 0
    ties: int = 0
    total_clone_calls_ms: float = 0.0
    total_keyword_calls_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Generate test queries from corpus
# ═══════════════════════════════════════════════════════════════════════


def generate_test_queries(corpus: str, num_queries: int = 50) -> list[dict[str, Any]]:
    """Generate relevance test queries from corpus paragraphs.

    Each query is a relevance question: "Is finding X relevant to angle Y?"
    We pair corpus sentences with angles to create diverse test cases.

    Args:
        corpus: Full corpus text.
        num_queries: Number of queries to generate.

    Returns:
        List of dicts with 'query', 'finding', 'angle', 'expected_relevant'.
    """
    # Extract sections from corpus
    sections: list[tuple[str, list[str]]] = []
    current_section = ""
    current_sentences: list[str] = []

    for line in corpus.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            if current_section and current_sentences:
                sections.append((current_section, current_sentences))
            current_section = line[3:].strip()
            current_sentences = []
        elif len(line) > 30:
            current_sentences.append(line)

    if current_section and current_sentences:
        sections.append((current_section, current_sentences))

    # Define test angles for relevance scoring
    test_angles = [
        "insulin timing and nutrient partitioning",
        "hematological effects and blood management",
        "growth hormone and IGF-1 protocols",
        "testosterone and trenbolone pharmacokinetics",
        "oral compounds and hepatoprotection",
    ]

    queries: list[dict[str, Any]] = []
    random.seed(42)  # reproducible

    for _ in range(num_queries):
        # Pick a random section and sentence
        section_name, sentences = random.choice(sections)
        finding = random.choice(sentences)

        # Pick a random angle to test relevance against
        angle = random.choice(test_angles)

        # Ground truth: is this finding from a section that matches the angle?
        # Simple heuristic: if the section name overlaps with the angle keywords
        section_lower = section_name.lower()
        angle_lower = angle.lower()
        expected_relevant = any(
            word in section_lower
            for word in angle_lower.split()
            if len(word) > 3
        )

        queries.append({
            "query": f"Is this finding relevant to {angle}?",
            "finding": finding,
            "angle": angle,
            "source_section": section_name,
            "expected_relevant": expected_relevant,
        })

    return queries


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Clone-scored relevance (via session proxy)
# ═══════════════════════════════════════════════════════════════════════


async def clone_score_relevance(
    finding: str,
    angle: str,
    clone_id: str,
    *,
    api_base: str,
    api_key: str,
    backend_model: str,
) -> tuple[float, float]:
    """Score relevance using the cloned worker context.

    Sends a relevance query through the session proxy, which prepends
    the worker's conversation history. The clone — having already
    reasoned about this domain — can make nuanced relevance judgments.

    Args:
        finding: The text to judge relevance for.
        angle: Research angle to judge against.
        clone_id: The registered clone model name.
        api_base: Backend LLM API base URL.
        api_key: Backend API key.
        backend_model: Actual model name for the backend.

    Returns:
        Tuple of (relevance_score 0.0-1.0, latency_ms).
    """
    prompt = (
        f"Given your expertise in {angle}, rate the relevance of this finding "
        f"to your research domain.\n\n"
        f"FINDING: {finding}\n\n"
        f"Respond with ONLY a number from 0.0 to 1.0 where:\n"
        f"  0.0 = completely irrelevant\n"
        f"  0.5 = tangentially related\n"
        f"  1.0 = directly relevant and important\n\n"
        f"Score:"
    )

    t0 = time.monotonic()

    try:
        result = await proxy_chat_completion({
            "model": clone_id,
            "_backend_model": backend_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
        })

        latency_ms = (time.monotonic() - t0) * 1000

        # Parse the response — expect a float
        text = result["choices"][0]["message"]["content"].strip()
        # Extract first float from response
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        score = float(match.group(1)) if match else 0.5
        score = max(0.0, min(1.0, score))

        return score, latency_ms

    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        logger.warning("clone scoring failed: %s", exc)
        return 0.5, latency_ms


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Keyword-scored relevance (rag.py)
# ═══════════════════════════════════════════════════════════════════════


def keyword_score_relevance(finding: str, angle: str) -> tuple[float, float]:
    """Score relevance using keyword overlap (current rag.py approach).

    Args:
        finding: The text to judge relevance for.
        angle: Research angle to judge against.

    Returns:
        Tuple of (relevance_score 0.0-1.0, latency_ms).
    """
    t0 = time.monotonic()

    # Extract concepts from the angle description
    concepts = extract_concepts(angle, top_k=15)
    raw_score = score_relevance(concepts, finding)

    # Normalize to 0.0-1.0 range (keyword scores are unbounded)
    # Max possible: ~15 concepts × 1.0 weight = 15.0
    normalized = min(1.0, raw_score / 5.0)

    latency_ms = (time.monotonic() - t0) * 1000
    return normalized, latency_ms


# ═══════════════════════════════════════════════════════════════════════
# Step 4: LLM judge for ground truth
# ═══════════════════════════════════════════════════════════════════════


async def judge_relevance(
    finding: str,
    angle: str,
    *,
    api_base: str,
    api_key: str,
    model: str,
) -> bool:
    """Use a separate LLM as an impartial judge of relevance.

    This provides ground truth for comparing clone vs keyword scoring.

    Args:
        finding: The text to judge.
        angle: The research angle.
        api_base: LLM API base URL.
        api_key: API key.
        model: Model to use as judge.

    Returns:
        True if the judge considers the finding relevant.
    """
    prompt = (
        f"You are an impartial relevance judge. Determine if the following "
        f"finding is directly relevant to the research angle.\n\n"
        f"RESEARCH ANGLE: {angle}\n\n"
        f"FINDING: {finding}\n\n"
        f"Is this finding directly relevant to the research angle? "
        f"Answer ONLY 'yes' or 'no'."
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip().lower()
            return text.startswith("yes")
    except Exception as exc:
        logger.warning("judge failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════
# Step 5: Run the full PoC test
# ═══════════════════════════════════════════════════════════════════════


async def run_poc_test(
    *,
    api_base: str,
    api_key: str,
    model: str,
    judge_model: str | None = None,
) -> PoCResults:
    """Run the full Stage 2 clone PoC test.

    Args:
        api_base: OpenAI-compatible API base URL.
        api_key: API key.
        model: Model for worker and clone queries.
        judge_model: Model for ground truth judging (defaults to same model).

    Returns:
        PoCResults with full comparison data.
    """
    judge_model = judge_model or model
    results = PoCResults(worker_angle=WORKER_ANGLE)

    # Configure session proxy backend
    set_backend(api_base, api_key)

    # Load corpus
    corpus = CORPUS_PATH.read_text()
    logger.info("corpus loaded: %d chars", len(corpus))

    # ── Step 1: Run 1 tool-free worker through 1 wave ──
    logger.info("=" * 60)
    logger.info("STEP 1: Running tool-free worker (1 wave)")
    logger.info("=" * 60)

    # Build a minimal data package for wave 1
    # Just §2 (corpus material) — as specified for wave 1
    insulin_section = ""
    for line in corpus.split("\n"):
        if line.startswith("## Insulin"):
            insulin_section = line + "\n"
        elif insulin_section and line.startswith("## "):
            break
        elif insulin_section:
            insulin_section += line + "\n"

    package = DataPackage(
        worker_id="worker_insulin_poc",
        angle=WORKER_ANGLE,
        wave=1,
        corpus_material=insulin_section.strip(),
    )

    worker_result = await run_tool_free_worker(
        package,
        TEST_QUERY,
        api_base=api_base,
        model=model,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.3,
    )

    results.worker_output_chars = worker_result["output_chars"]
    results.worker_elapsed_s = worker_result["elapsed_s"]

    if worker_result["status"] != "success":
        logger.error("Worker failed: %s", worker_result.get("error", "unknown"))
        return results

    logger.info(
        "Worker complete: %d chars output in %.1fs",
        results.worker_output_chars, results.worker_elapsed_s,
    )

    # ── Step 2: Clone the worker's conversation ──
    logger.info("=" * 60)
    logger.info("STEP 2: Registering clone")
    logger.info("=" * 60)

    system_prompt = (
        f"You are a {WORKER_ANGLE} specialist conducting deep research.\n\n"
        f"You will receive a structured research brief containing corpus "
        f"material, peer findings, knowledge summaries, and identified gaps. "
        f"Your job is to reason deeply over ALL provided material through "
        f"your {WORKER_ANGLE} lens."
    )

    clone_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": package.render(TEST_QUERY)},
        {"role": "assistant", "content": worker_result["response"]},
    ]

    clone_id = f"clone_{WORKER_ANGLE.replace(' ', '_')}"
    register_clone(
        clone_id=clone_id,
        messages=clone_messages,
        angle=WORKER_ANGLE,
        wave=1,
    )

    results.clone_messages = len(clone_messages)
    logger.info("Clone registered: %s (%d messages)", clone_id, len(clone_messages))

    # ── Step 3: Generate test queries ──
    logger.info("=" * 60)
    logger.info("STEP 3: Generating %d test queries", NUM_TEST_QUERIES)
    logger.info("=" * 60)

    test_queries = generate_test_queries(corpus, NUM_TEST_QUERIES)
    results.num_queries = len(test_queries)
    logger.info("Generated %d test queries", len(test_queries))

    # ── Step 4: Run clone vs keyword scoring on each query ──
    logger.info("=" * 60)
    logger.info("STEP 4: Scoring %d queries (clone vs keyword)", len(test_queries))
    logger.info("=" * 60)

    for i, tq in enumerate(test_queries):
        finding = tq["finding"]
        angle = tq["angle"]

        # Clone score (async LLM call via session proxy)
        clone_score, clone_ms = await clone_score_relevance(
            finding=finding,
            angle=angle,
            clone_id=clone_id,
            api_base=api_base,
            api_key=api_key,
            backend_model=model,
        )

        # Keyword score (local, instant)
        kw_score, kw_ms = keyword_score_relevance(finding, angle)

        # LLM judge ground truth
        judge_relevant = await judge_relevance(
            finding=finding,
            angle=angle,
            api_base=api_base,
            api_key=api_key,
            model=judge_model,
        )

        j = RelevanceJudgment(
            query=tq["query"],
            finding=finding[:200],
            clone_score=clone_score,
            keyword_score=kw_score,
            judge_relevant=judge_relevant,
        )
        results.judgments.append(j)
        results.total_clone_calls_ms += clone_ms
        results.total_keyword_calls_ms += kw_ms

        if (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d queries scored", i + 1, len(test_queries),
            )

    # ── Step 5: Compute precision/recall ──
    logger.info("=" * 60)
    logger.info("STEP 5: Computing metrics")
    logger.info("=" * 60)

    _compute_metrics(results)

    return results


def _compute_metrics(results: PoCResults) -> None:
    """Compute precision, recall, and comparison metrics."""
    clone_threshold = 0.5
    keyword_threshold = 0.3  # keyword scores are lower scale

    clone_tp = 0
    clone_fp = 0
    clone_fn = 0
    keyword_tp = 0
    keyword_fp = 0
    keyword_fn = 0

    for j in results.judgments:
        clone_predicts_relevant = j.clone_score >= clone_threshold
        keyword_predicts_relevant = j.keyword_score >= keyword_threshold

        if j.judge_relevant:
            # Positive case
            if clone_predicts_relevant:
                clone_tp += 1
            else:
                clone_fn += 1

            if keyword_predicts_relevant:
                keyword_tp += 1
            else:
                keyword_fn += 1
        else:
            # Negative case
            if clone_predicts_relevant:
                clone_fp += 1
            if keyword_predicts_relevant:
                keyword_fp += 1

        # Head-to-head comparison
        if clone_predicts_relevant == j.judge_relevant and keyword_predicts_relevant != j.judge_relevant:
            results.clone_wins += 1
        elif keyword_predicts_relevant == j.judge_relevant and clone_predicts_relevant != j.judge_relevant:
            results.keyword_wins += 1
        else:
            results.ties += 1

    # Precision = TP / (TP + FP)
    results.clone_precision = clone_tp / max(1, clone_tp + clone_fp)
    results.keyword_precision = keyword_tp / max(1, keyword_tp + keyword_fp)

    # Recall = TP / (TP + FN)
    results.clone_recall = clone_tp / max(1, clone_tp + clone_fn)
    results.keyword_recall = keyword_tp / max(1, keyword_tp + keyword_fn)


def print_results(results: PoCResults) -> None:
    """Print formatted results to stdout."""
    print("\n" + "=" * 70)
    print("  STAGE 2 POC RESULTS: Clone-Scored vs Keyword-Scored Relevance")
    print("=" * 70)

    print(f"\nWorker: {results.worker_angle}")
    print(f"  Output: {results.worker_output_chars} chars in {results.worker_elapsed_s}s")
    print(f"  Clone: {results.clone_messages} messages registered")

    print(f"\nTest queries: {results.num_queries}")

    print(f"\n{'Metric':<30} {'Clone':>10} {'Keyword':>10} {'Delta':>10}")
    print("-" * 62)

    precision_delta = results.clone_precision - results.keyword_precision
    recall_delta = results.clone_recall - results.keyword_recall

    print(f"{'Precision':<30} {results.clone_precision:>10.1%} {results.keyword_precision:>10.1%} {precision_delta:>+10.1%}")
    print(f"{'Recall':<30} {results.clone_recall:>10.1%} {results.keyword_recall:>10.1%} {recall_delta:>+10.1%}")

    avg_clone_ms = results.total_clone_calls_ms / max(1, results.num_queries)
    avg_kw_ms = results.total_keyword_calls_ms / max(1, results.num_queries)
    print(f"{'Avg latency (ms)':<30} {avg_clone_ms:>10.0f} {avg_kw_ms:>10.0f} {'':>10}")

    print(f"\nHead-to-head:")
    print(f"  Clone wins:   {results.clone_wins}")
    print(f"  Keyword wins: {results.keyword_wins}")
    print(f"  Ties:         {results.ties}")

    print(f"\n{'=' * 70}")

    # Decision
    if precision_delta >= 0.15:
        print("  DECISION: Clone precision > keyword by ≥15%")
        print("  → PROCEED to Stage 3 (tool-free workers validated)")
    elif precision_delta > 0:
        print(f"  DECISION: Clone slightly better (+{precision_delta:.1%}) but <15% threshold")
        print("  → Consider hybrid approach or deeper investigation")
    else:
        print("  DECISION: Clone NOT better than keyword scoring")
        print("  → Architecture's core bet may be wrong — investigate alternatives")

    print("=" * 70)


def save_results(results: PoCResults, path: Path) -> None:
    """Save results to a JSON file."""
    data = {
        "worker_angle": results.worker_angle,
        "worker_output_chars": results.worker_output_chars,
        "worker_elapsed_s": results.worker_elapsed_s,
        "clone_messages": results.clone_messages,
        "num_queries": results.num_queries,
        "clone_precision": results.clone_precision,
        "keyword_precision": results.keyword_precision,
        "clone_recall": results.clone_recall,
        "keyword_recall": results.keyword_recall,
        "clone_wins": results.clone_wins,
        "keyword_wins": results.keyword_wins,
        "ties": results.ties,
        "avg_clone_latency_ms": results.total_clone_calls_ms / max(1, results.num_queries),
        "avg_keyword_latency_ms": results.total_keyword_calls_ms / max(1, results.num_queries),
        "precision_delta": results.clone_precision - results.keyword_precision,
        "decision": (
            "proceed_stage3" if results.clone_precision - results.keyword_precision >= 0.15
            else "investigate" if results.clone_precision > results.keyword_precision
            else "pivot"
        ),
        "judgments": [
            {
                "query": j.query,
                "finding": j.finding,
                "clone_score": j.clone_score,
                "keyword_score": j.keyword_score,
                "judge_relevant": j.judge_relevant,
            }
            for j in results.judgments
        ],
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Results saved to %s", path)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


async def main() -> None:
    """Entry point for the clone PoC test."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable required")
        sys.exit(1)

    api_base = "https://openrouter.ai/api/v1"
    model = "meta-llama/llama-3.1-8b-instruct"

    logger.info("Starting Stage 2 clone PoC test")
    logger.info("  API: %s", api_base)
    logger.info("  Model: %s", model)
    logger.info("  Queries: %d", NUM_TEST_QUERIES)

    results = await run_poc_test(
        api_base=api_base,
        api_key=api_key,
        model=model,
    )

    print_results(results)

    output_path = Path(__file__).parent / "clone_poc_results.json"
    save_results(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
