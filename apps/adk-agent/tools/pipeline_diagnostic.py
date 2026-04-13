"""MiroThinker Pipeline Diagnostic Tool (BAT.AI pattern).

Standalone self-corrective RAG diagnostic that analyses pipeline logs and
DuckDB corpus state to answer:

    1. **Health check** — Did the pipeline work well?  What could improve?
    2. **Post-failure forensics** — What went wrong and why?
    3. **Remediation** — What specific code/config change would fix it?

Usage (standalone):
    python -m tools.pipeline_diagnostic \
        --log /tmp/mirothinker_pipeline.log \
        --db  /path/to/corpus.duckdb \
        --question "Why did the pipeline stop after iteration 1?"

    # Or just health-check mode (no question):
    python -m tools.pipeline_diagnostic \
        --log /tmp/mirothinker_pipeline.log \
        --db  /path/to/corpus.duckdb

Architecture (adapted from NVIDIA BAT.AI):
    1. Ingest log file + DuckDB corpus → structured chunks
    2. Hybrid retrieval: keyword search + DuckDB SQL queries
    3. Relevance grading via LLM
    4. Generate diagnosis
    5. Self-correct: if diagnosis isn't grounded or doesn't answer
       the question, rewrite query and re-retrieve (max 2 transforms)
    6. Output structured health report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import litellm
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

# Use the same LLM the pipeline uses — no new dependencies.
def _resolve_model() -> str:
    """Resolve the LLM model string for direct litellm calls.

    ADK_MODEL uses a ``litellm/openai/openai/gpt-4o-mini`` format that only
    works inside ADK's wrapper.  For raw ``litellm.completion()`` calls we
    strip the ``litellm/`` prefix and collapse to ``openai/<model>``.
    """
    explicit = os.environ.get("DIAGNOSTIC_MODEL")
    if explicit:
        return explicit
    adk_model = os.environ.get("ADK_MODEL", "")
    if adk_model.startswith("litellm/"):
        # "litellm/openai/openai/gpt-4o-mini" → "openai/gpt-4o-mini"
        parts = adk_model.split("/")
        # Keep provider + model (last two segments)
        return "/".join(parts[-2:]) if len(parts) >= 3 else adk_model
    return adk_model or "openai/gpt-4o-mini"

_MODEL = _resolve_model()
_BASE_URL = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
_API_KEY = os.environ.get("OPENAI_API_KEY", "")

_MAX_TRANSFORMS = 2          # Self-correction retries
_CHUNK_SIZE = 8_000           # Characters per log chunk
_CHUNK_OVERLAP = 2_000        # Overlap between chunks
_TOP_K = 8                    # Chunks to retrieve per question
_MAX_CONTEXT_CHARS = 60_000   # Hard cap on context sent to LLM


# ── Data models ───────────────────────────────────────────────────

@dataclass
class DiagnosticState:
    """Tracks the diagnostic workflow state (mirrors BAT.AI GraphState)."""
    question: str = ""
    log_path: str = ""
    db_path: str = ""
    log_chunks: list[str] = field(default_factory=list)
    corpus_summary: str = ""
    retrieved_docs: list[str] = field(default_factory=list)
    generation: str = ""
    transform_count: int = 0
    grounded: bool = False


# ── Step 1: Ingest ────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
                overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _extract_corpus_summary(db_path: str) -> str:
    """Query a DuckDB corpus file and produce a structured text summary."""
    if not db_path or not Path(db_path).exists():
        return "(no corpus database provided)"

    try:
        conn = duckdb.connect(db_path, read_only=True)
    except Exception as exc:
        return f"(failed to open corpus DB: {exc})"

    sections: list[str] = []
    try:
        # Total counts by row_type
        try:
            rows = conn.execute(
                "SELECT row_type, COUNT(*) as cnt FROM conditions "
                "GROUP BY row_type ORDER BY cnt DESC"
            ).fetchall()
            sections.append("## Row Type Breakdown")
            for rt, cnt in rows:
                sections.append(f"  {rt}: {cnt}")
        except Exception:
            sections.append("(no conditions table found)")

        # Per-angle breakdown
        try:
            rows = conn.execute(
                "SELECT angle, COUNT(*) as cnt, "
                "  AVG(CAST(composite_quality AS FLOAT)) as avg_q, "
                "  SUM(CASE WHEN consider_for_use THEN 1 ELSE 0 END) as active "
                "FROM conditions WHERE row_type = 'finding' "
                "GROUP BY angle ORDER BY cnt DESC"
            ).fetchall()
            sections.append("\n## Angle Breakdown (findings only)")
            for angle, cnt, avg_q, active in rows:
                sections.append(
                    f"  {angle}: {cnt} total, {active} active, "
                    f"avg_quality={avg_q:.3f}" if avg_q else
                    f"  {angle}: {cnt} total, {active} active, avg_quality=N/A"
                )
        except Exception:
            pass

        # Iteration breakdown
        try:
            rows = conn.execute(
                "SELECT iteration, COUNT(*) as cnt FROM conditions "
                "WHERE row_type = 'finding' "
                "GROUP BY iteration ORDER BY iteration"
            ).fetchall()
            sections.append("\n## Iteration Breakdown (findings)")
            for it, cnt in rows:
                sections.append(f"  iteration {it}: {cnt} findings")
        except Exception:
            pass

        # Source type breakdown
        try:
            rows = conn.execute(
                "SELECT source_type, COUNT(*) as cnt FROM conditions "
                "WHERE row_type = 'finding' AND consider_for_use = TRUE "
                "GROUP BY source_type ORDER BY cnt DESC"
            ).fetchall()
            sections.append("\n## Source Type Breakdown (active findings)")
            for st, cnt in rows:
                sections.append(f"  {st}: {cnt}")
        except Exception:
            pass

        # Quality distribution
        try:
            rows = conn.execute(
                "SELECT "
                "  SUM(CASE WHEN CAST(composite_quality AS FLOAT) >= 0.6 THEN 1 ELSE 0 END) as strong, "
                "  SUM(CASE WHEN CAST(composite_quality AS FLOAT) >= 0.3 AND CAST(composite_quality AS FLOAT) < 0.6 THEN 1 ELSE 0 END) as moderate, "
                "  SUM(CASE WHEN CAST(composite_quality AS FLOAT) < 0.3 THEN 1 ELSE 0 END) as weak "
                "FROM conditions WHERE row_type = 'finding' AND consider_for_use = TRUE"
            ).fetchone()
            if rows:
                sections.append(f"\n## Quality Tiers (active findings)")
                sections.append(f"  strong (>=0.6): {rows[0]}")
                sections.append(f"  moderate (0.3-0.6): {rows[1]}")
                sections.append(f"  weak (<0.3): {rows[2]}")
        except Exception:
            pass

        # Contradictions
        try:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE contradiction_flag = TRUE AND consider_for_use = TRUE"
            ).fetchone()[0]
            sections.append(f"\n## Contradictions: {cnt}")
        except Exception:
            pass

        # Serendipity markers
        try:
            rows = conn.execute(
                "SELECT strategy, COUNT(*) as cnt FROM conditions "
                "WHERE strategy LIKE '%serendip%' "
                "GROUP BY strategy ORDER BY cnt DESC"
            ).fetchall()
            if rows:
                sections.append("\n## Serendipity Rows")
                for strat, cnt in rows:
                    sections.append(f"  {strat}: {cnt}")
            else:
                sections.append("\n## Serendipity Rows: NONE FOUND")
        except Exception:
            pass

        # Thoughts and insights
        try:
            thoughts = conn.execute(
                "SELECT COUNT(*) FROM conditions WHERE row_type = 'thought'"
            ).fetchone()[0]
            insights = conn.execute(
                "SELECT COUNT(*) FROM conditions WHERE row_type = 'insight'"
            ).fetchone()[0]
            sections.append(f"\n## Thought Swarm: {thoughts} thoughts, {insights} insights")
        except Exception:
            pass

        # Sample of top findings (for grounding)
        try:
            rows = conn.execute(
                "SELECT id, angle, SUBSTR(fact, 1, 200) as fact_preview, "
                "  composite_quality, source_type, strategy "
                "FROM conditions WHERE row_type = 'finding' "
                "AND consider_for_use = TRUE "
                "ORDER BY CAST(composite_quality AS FLOAT) DESC LIMIT 10"
            ).fetchall()
            sections.append("\n## Top 10 Findings (by quality)")
            for r in rows:
                sections.append(
                    f"  #{r[0]} [{r[1]}] q={r[3]} src={r[4]} strategy={r[5]}\n"
                    f"    {r[2]}..."
                )
        except Exception:
            pass

        # Sample of lowest findings
        try:
            rows = conn.execute(
                "SELECT id, angle, SUBSTR(fact, 1, 200) as fact_preview, "
                "  composite_quality, source_type "
                "FROM conditions WHERE row_type = 'finding' "
                "AND consider_for_use = TRUE "
                "ORDER BY CAST(composite_quality AS FLOAT) ASC LIMIT 5"
            ).fetchall()
            sections.append("\n## Bottom 5 Findings (lowest quality)")
            for r in rows:
                sections.append(f"  #{r[0]} [{r[1]}] q={r[3]} src={r[4]}\n    {r[2]}...")
        except Exception:
            pass

        # Error/warning patterns in any text fields
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE fact ILIKE '%error%' OR fact ILIKE '%fail%' "
                "OR fact ILIKE '%exception%'"
            ).fetchone()[0]
            sections.append(f"\n## Rows containing error/fail/exception keywords: {rows}")
        except Exception:
            pass

    finally:
        conn.close()

    return "\n".join(sections)


def ingest(state: DiagnosticState) -> DiagnosticState:
    """Load logs and corpus, produce chunks."""
    # Log file
    log_text = ""
    if state.log_path and Path(state.log_path).exists():
        log_text = Path(state.log_path).read_text(errors="replace")
        logger.info("Loaded log file: %d chars", len(log_text))
    else:
        logger.warning("No log file at %s", state.log_path)

    state.log_chunks = _chunk_text(log_text) if log_text else []
    logger.info("Split into %d chunks", len(state.log_chunks))

    # Corpus DB
    state.corpus_summary = _extract_corpus_summary(state.db_path)
    logger.info("Corpus summary: %d chars", len(state.corpus_summary))

    return state


# ── Step 2: Retrieve ──────────────────────────────────────────────

def _keyword_score(chunk: str, question: str) -> float:
    """Simple keyword-overlap retrieval score (BM25-lite)."""
    # Tokenize
    q_tokens = set(re.findall(r'\w+', question.lower()))
    c_tokens = re.findall(r'\w+', chunk.lower())
    if not c_tokens or not q_tokens:
        return 0.0

    c_token_set = set(c_tokens)
    overlap = q_tokens & c_token_set

    # Term frequency component
    tf_sum = sum(c_tokens.count(t) for t in overlap)
    # Normalize by chunk length
    score = tf_sum / (len(c_tokens) + 1)
    # Bonus for fraction of query terms matched
    coverage = len(overlap) / len(q_tokens) if q_tokens else 0
    return score + coverage


def retrieve(state: DiagnosticState) -> DiagnosticState:
    """Retrieve the most relevant log chunks for the question."""
    question = state.question

    # Always include corpus summary as a "document"
    docs = []
    if state.corpus_summary and state.corpus_summary != "(no corpus database provided)":
        docs.append(f"[CORPUS STATE]\n{state.corpus_summary}")

    # Score and rank log chunks
    scored = [(i, _keyword_score(chunk, question), chunk)
              for i, chunk in enumerate(state.log_chunks)]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Also prioritize chunks with ERROR/WARNING/EXCEPTION
    error_chunks = [
        (i, s + 0.5, c) for i, s, c in scored
        if re.search(r'(?i)(error|exception|traceback|fail|warning)', c)
    ]

    # Merge: error chunks first (deduped), then top-k by score
    seen = set()
    for i, s, c in error_chunks[:3]:
        if i not in seen:
            docs.append(f"[LOG CHUNK {i} (error-priority, score={s:.3f})]\n{c}")
            seen.add(i)

    for i, s, c in scored:
        if len(docs) >= _TOP_K + 1:  # +1 for corpus summary
            break
        if i not in seen:
            docs.append(f"[LOG CHUNK {i} (score={s:.3f})]\n{c}")
            seen.add(i)

    state.retrieved_docs = docs
    logger.info("Retrieved %d documents (including corpus summary)", len(docs))
    return state


# ── Step 3: Grade relevance ───────────────────────────────────────

def _llm_call(system: str, user: str) -> str:
    """Make an LLM call via litellm.

    Uses the OpenRouter base URL directly rather than going through litellm's
    provider routing, to avoid model-string parsing issues.
    """
    import openai as _openai

    client = _openai.OpenAI(
        api_key=_API_KEY,
        base_url=_BASE_URL,
    )
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def grade_documents(state: DiagnosticState) -> DiagnosticState:
    """Grade each retrieved document for relevance, keep only relevant ones."""
    if not state.retrieved_docs:
        return state

    graded: list[str] = []
    for doc in state.retrieved_docs:
        # Always keep corpus summary
        if doc.startswith("[CORPUS STATE]"):
            graded.append(doc)
            continue

        try:
            result = _llm_call(
                system=(
                    "You are a document relevance evaluator for a pipeline diagnostic system. "
                    "Determine if this document contains information relevant to diagnosing "
                    "the pipeline's behavior, errors, or performance.\n\n"
                    "Respond with ONLY 'yes' or 'no'."
                ),
                user=(
                    f"QUESTION: {state.question}\n\n"
                    f"DOCUMENT:\n{doc[:3000]}\n\n"
                    "Is this document relevant to answering the diagnostic question?"
                ),
            )
            if "yes" in result.lower():
                graded.append(doc)
                logger.debug("GRADE: RELEVANT — %s", doc[:80])
            else:
                logger.debug("GRADE: NOT RELEVANT — %s", doc[:80])
        except Exception as exc:
            # On grading failure, keep the document (permissive)
            logger.warning("Grading failed for chunk, keeping it: %s", exc)
            graded.append(doc)

    state.retrieved_docs = graded
    logger.info("After grading: %d relevant documents", len(graded))
    return state


# ── Step 4: Generate diagnosis ────────────────────────────────────

_DIAGNOSIS_SYSTEM = textwrap.dedent("""\
    You are an expert MiroThinker pipeline diagnostician.  You analyse
    pipeline logs and DuckDB corpus state to produce structured health
    reports.

    The MiroThinker pipeline architecture:
      SequentialAgent → LoopAgent(Thinker → SearchExecutor → Maestro) → Synthesiser

    Key components:
    - Thinker: decomposes query into angles, assigns specialists
    - Search Executor: multi-API fan-out (Exa, Firecrawl, Perplexity, etc.)
    - Maestro: Flock SQL conductor — scores, clusters, deduplicates,
      runs serendipity templates (contrarian, cross-angle, diversity boost)
    - Thought Swarm: parallel specialists per angle, cross-angle surprise detection
    - Condition Manager: iteration gating, devil's advocate injection
    - Corpus Store: single DuckDB table with gradient-flag columns
    - Synthesiser: produces final research document from 'ready' findings

    Serendipity mechanisms:
    - Contrarian query injection in search fan-out
    - Cross-angle surprise detection after specialists run
    - Angle diversity boost (pure SQL)
    - Devil's advocate injection after 2+ iterations with no contradictions
    - 5 maestro SQL templates (9-13): contrarian challenge, cross-angle bridge,
      surprise scoring, consensus detector, angle diversity boost

    INSTRUCTIONS:
    1. Base your analysis STRICTLY on the provided log/corpus data
    2. Structure your response with these sections:
       ## Pipeline Health Summary
       ## Iteration Analysis
       ## Tool & API Usage
       ## Corpus Quality Assessment
       ## Serendipity Effectiveness
       ## Errors & Warnings
       ## Specific Recommendations
    3. Be specific — cite log lines, row counts, quality scores
    4. If information is missing, say so explicitly
    5. Focus on actionable improvements
""")


def generate(state: DiagnosticState) -> DiagnosticState:
    """Generate the diagnostic report from retrieved documents."""
    context = "\n\n---\n\n".join(state.retrieved_docs)
    # Truncate context if too long
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"

    result = _llm_call(
        system=_DIAGNOSIS_SYSTEM,
        user=(
            f"DIAGNOSTIC QUESTION: {state.question}\n\n"
            f"EVIDENCE:\n{context}\n\n"
            "Produce a comprehensive diagnostic report based solely on the above evidence."
        ),
    )
    state.generation = result
    logger.info("Generated diagnosis: %d chars", len(result))
    return state


# ── Step 5: Self-correct ──────────────────────────────────────────

def grade_generation(state: DiagnosticState) -> str:
    """Grade whether the generation is grounded and answers the question.

    Returns: 'useful', 'not useful', or 'not supported'
    """
    try:
        # Check if generation addresses the question
        result = _llm_call(
            system=(
                "You are a response quality evaluator for a pipeline diagnostic tool. "
                "Determine if the diagnostic report adequately addresses the question "
                "with specific, actionable findings grounded in the provided evidence.\n\n"
                "Respond with ONLY 'yes' or 'no'."
            ),
            user=(
                f"QUESTION: {state.question}\n\n"
                f"DIAGNOSTIC REPORT:\n{state.generation[:3000]}\n\n"
                "Does this report adequately address the diagnostic question with "
                "specific, evidence-grounded findings?"
            ),
        )
        if "yes" in result.lower():
            state.grounded = True
            return "useful"
        else:
            return "not useful"
    except Exception:
        # On failure, accept the generation
        return "useful"


def transform_query(state: DiagnosticState) -> DiagnosticState:
    """Rewrite the diagnostic question to improve retrieval."""
    state.transform_count += 1
    logger.info("Transform attempt %d/%d", state.transform_count, _MAX_TRANSFORMS)

    result = _llm_call(
        system=(
            "You are a prompt optimization specialist for a pipeline diagnostic "
            "RAG system. Rewrite the query to improve log/corpus retrieval.\n\n"
            "Add relevant MiroThinker terms: iteration, angle, composite_quality, "
            "consider_for_use, Flock SQL, maestro, thinker, search executor, "
            "serendipity, thought swarm, corpus store.\n\n"
            "Output ONLY the rewritten query, nothing else."
        ),
        user=f"Original query: {state.question}\n\nRewritten query:",
    )
    old_q = state.question
    state.question = result.strip()
    logger.info("Query rewritten: %r → %r", old_q[:80], state.question[:80])
    return state


# ── Main workflow (BAT.AI graph, flattened) ───────────────────────

def run_diagnostic(
    log_path: str = "",
    db_path: str = "",
    question: str = "",
) -> str:
    """Run the full self-corrective diagnostic pipeline.

    Args:
        log_path: Path to pipeline log file (server stdout/stderr).
        db_path:  Path to DuckDB corpus file.
        question: Specific diagnostic question.  If empty, runs a
                  general health check.

    Returns:
        The diagnostic report as a markdown string.
    """
    if not question:
        question = (
            "Perform a comprehensive health check of this MiroThinker pipeline run. "
            "Assess: Did all tools fire? Did serendipity activate? Was the thought "
            "swarm productive? Are there quality issues in the corpus? What "
            "specific improvements would make the next run better?"
        )

    state = DiagnosticState(
        question=question,
        log_path=log_path,
        db_path=db_path,
    )

    # 1. Ingest
    logger.info("=== INGEST ===")
    state = ingest(state)

    for attempt in range(1 + _MAX_TRANSFORMS):
        # 2. Retrieve
        logger.info("=== RETRIEVE (attempt %d) ===", attempt + 1)
        state = retrieve(state)

        # 3. Grade documents
        logger.info("=== GRADE DOCUMENTS ===")
        state = grade_documents(state)

        if not state.retrieved_docs:
            logger.warning("No relevant documents found — transforming query")
            if state.transform_count < _MAX_TRANSFORMS:
                state = transform_query(state)
                continue
            else:
                state.generation = (
                    "## Diagnostic Failed\n\n"
                    "No relevant log data or corpus state could be retrieved "
                    "after query transformation.  Check that the log file and "
                    "corpus database paths are correct."
                )
                return state.generation

        # 4. Generate
        logger.info("=== GENERATE ===")
        state = generate(state)

        # 5. Grade generation
        logger.info("=== GRADE GENERATION ===")
        verdict = grade_generation(state)

        if verdict == "useful":
            logger.info("Diagnostic complete — grounded and useful")
            return state.generation
        elif state.transform_count >= _MAX_TRANSFORMS:
            logger.info("Max transforms reached — accepting generation")
            return state.generation
        else:
            logger.info("Generation not useful — transforming query")
            state = transform_query(state)

    return state.generation


# ── CLI entry point ───────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiroThinker Pipeline Diagnostic Tool (BAT.AI pattern)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # General health check
              python -m tools.pipeline_diagnostic --log /tmp/pipeline.log --db corpus.duckdb

              # Specific failure forensics
              python -m tools.pipeline_diagnostic --log /tmp/pipeline.log \\
                  --question "Why did the search executor fail after iteration 2?"

              # Post-error remediation
              python -m tools.pipeline_diagnostic --log /tmp/pipeline.log \\
                  --db corpus.duckdb \\
                  --question "The maestro crashed with a Flock SQL error — what fix is needed?"
        """),
    )
    parser.add_argument("--log", default="", help="Path to pipeline log file")
    parser.add_argument("--db", default="", help="Path to DuckDB corpus file")
    parser.add_argument("--question", "-q", default="",
                        help="Diagnostic question (empty = general health check)")
    parser.add_argument("--output", "-o", default="",
                        help="Write report to file (default: stdout)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_diagnostic(
        log_path=args.log,
        db_path=args.db,
        question=args.question,
    )

    if args.output:
        Path(args.output).write_text(report)
        logger.info("Report written to %s", args.output)
    else:
        print("\n" + "=" * 60)
        print("MIROTHINKER PIPELINE DIAGNOSTIC REPORT")
        print("=" * 60 + "\n")
        print(report)


if __name__ == "__main__":
    main()
