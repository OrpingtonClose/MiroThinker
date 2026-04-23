# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Angle detection and corpus splitting for the gossip swarm.

Three strategies for assigning work to specialist workers:

1. **LLM-semantic** (best): Ask the LLM to score each section-angle pair,
   then solve optimal assignment via the Hungarian algorithm.  This ensures
   sections go to the angle that will extract the MOST value, not just the
   first substring match.  Costs 1 extra LLM call (~2-5s).

2. **Keyword** (fast fallback): Match sections to angles by substring overlap
   in titles.  Zero extra cost but greedy — a section about "trenbolone and
   sleep" always goes to the trenbolone worker, even if the sleep/circadian
   worker would extract more value.

3. **Size-based** (structural fallback): Split the corpus into roughly equal
   chunks by character count.  Workers get arbitrary slices.

LLM-semantic is preferred and used by default when a ``complete_fn`` is
available.  Set ``SwarmConfig.enable_semantic_assignment = False`` to
fall back to keyword matching.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CorpusSection:
    """A semantically coherent section of the corpus."""

    title: str
    content: str
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


@dataclass
class WorkerAssignment:
    """What a single worker receives for synthesis."""

    worker_id: int
    angle: str
    raw_content: str  # full original section content (retained for full-corpus gossip)
    char_count: int = 0
    summary: str = ""
    prev_summary: str = ""  # previous round's summary (for convergence detection)
    angle_idx: int = -1  # index into the original angles list (-1 = unknown)

    def __post_init__(self) -> None:
        self.char_count = len(self.raw_content)


def detect_sections(corpus: str) -> list[CorpusSection]:
    """Detect semantic sections in the corpus using structural markers.

    Looks for markdown headers (## / ### / ####), numbered sections,
    and paragraph breaks as section boundaries.
    """
    if not corpus or not corpus.strip():
        return []

    # Try markdown headers first (## Header)
    header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    headers = list(header_pattern.finditer(corpus))

    if len(headers) >= 2:
        sections: list[CorpusSection] = []
        for i, match in enumerate(headers):
            title = match.group(2).strip()
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(corpus)
            content = corpus[start:end].strip()
            if content and len(content) > 50:
                sections.append(CorpusSection(title=title, content=content))
        return sections

    # Try numbered sections (1. / 1) / Section 1:)
    numbered_pattern = re.compile(
        r'^(?:(?:\d+)[.)]\s+|Section\s+\d+[:.]\s*)(.+)$',
        re.MULTILINE | re.IGNORECASE,
    )
    numbered = list(numbered_pattern.finditer(corpus))

    if len(numbered) >= 2:
        sections = []
        for i, match in enumerate(numbered):
            title = match.group(1).strip()
            start = match.end()
            end = numbered[i + 1].start() if i + 1 < len(numbered) else len(corpus)
            content = corpus[start:end].strip()
            if content and len(content) > 50:
                sections.append(CorpusSection(title=title, content=content))
        return sections

    # Fallback: split by double newlines into paragraph clusters
    paragraphs = re.split(r'\n\s*\n', corpus)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]

    if len(paragraphs) <= 1:
        return [CorpusSection(title="Full Corpus", content=corpus)]

    return [
        CorpusSection(
            title=f"Section {i + 1}",
            content=p,
        )
        for i, p in enumerate(paragraphs)
    ]


async def extract_required_angles(
    query: str,
    complete_fn,
) -> list[str]:
    """Extract mandatory research angles from the user's query.

    If the user mentions specific compounds, domains, or topics, each
    one becomes a required angle that the swarm MUST cover regardless
    of corpus composition.  This prevents underrepresented topics from
    being absorbed into dominant ones.

    Args:
        query: The user's research query.
        complete_fn: Async LLM completion callable.

    Returns:
        List of required angle labels (may be empty).
    """
    prompt = (
        "You are a research strategist. Read the user's query and extract "
        "the DISTINCT TOPICS, COMPOUNDS, DOMAINS, or ENTITIES that the user "
        "explicitly wants investigated. Each becomes a mandatory research "
        "angle.\n\n"
        "Rules:\n"
        "- Only extract topics the user EXPLICITLY mentions or clearly implies\n"
        "- Each angle should be a concise label (2-6 words)\n"
        "- Do NOT invent angles the user didn't ask about\n"
        "- If the query mentions 5 compounds, return 5 angles\n"
        "- If the query is broad/vague with no specific topics, return nothing\n\n"
        "Return one per line, prefixed with REQUIRED:\n"
        "If no specific topics are identifiable, return: REQUIRED: NONE\n\n"
        f"USER QUERY: {query}"
    )

    try:
        response = await complete_fn(prompt)
        angles: list[str] = []
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("REQUIRED:"):
                a = line[9:].strip().strip("'\"-.•*")
                if a and len(a) > 2 and a.upper() != "NONE":
                    angles.append(a[:80])
        return angles
    except Exception:
        return []


def merge_angles(
    detected: list[str],
    required: list[str],
    max_angles: int,
) -> list[str]:
    """Merge LLM-detected angles with prompt-required angles.

    Required angles always appear first.  Detected angles fill
    remaining slots, skipping any that duplicate a required angle
    (case-insensitive substring match).

    Args:
        detected: Angles detected from corpus content.
        required: Mandatory angles from user query.
        max_angles: Maximum total angles.

    Returns:
        Merged angle list (at most ``max_angles`` entries).
    """
    if not required:
        return detected[:max_angles]

    merged = list(required)
    required_lower = {r.lower() for r in required}

    for angle in detected:
        if len(merged) >= max_angles:
            break
        # Skip if this detected angle overlaps with a required one
        a_lower = angle.lower()
        if any(r in a_lower or a_lower in r for r in required_lower):
            continue
        merged.append(angle)

    return merged[:max_angles]


async def detect_angles_via_llm(
    corpus: str,
    query: str,
    complete_fn,
    max_angles: int = 6,
) -> list[str]:
    """Use LLM to identify distinct research angles in the corpus.

    Falls back to structural section titles if LLM fails.
    """
    # Use a generous corpus preview for angle detection — local models
    # have large context windows and need enough material to identify
    # underrepresented topics.
    corpus_preview = corpus[:30000]

    prompt = (
        "You are a research strategist. Read the following corpus excerpt "
        "and identify the distinct specialist research ANGLES — the separate "
        "facets, disciplines, or lines of inquiry present in this material.\n\n"
        "Each angle should be a concise label (3-8 words) that names a "
        "specific investigative direction. Do NOT return generic labels like "
        "'further research' or 'additional analysis'. Each angle must name "
        "the actual domain, theory, mechanism, or question.\n\n"
        "CRITICAL RULES:\n"
        "- Each angle must be ORTHOGONAL — covering a different domain or facet\n"
        "- NEVER split a single topic into numbered parts (e.g. 'Topic part 1', "
        "'Topic part 2'). If a topic is large, give it ONE angle.\n"
        "- NEVER duplicate an angle under a different name\n"
        "- Two angles that would investigate the same mechanisms, compounds, "
        "or questions should be MERGED into one\n\n"
        f"Return 2-{max_angles} angles, one per line, prefixed with ANGLE:\n"
        "If the material only has one coherent direction, return just one.\n\n"
        f"USER QUERY: {query}\n\n"
        f"CORPUS (first {len(corpus_preview)} chars):\n{corpus_preview}"
    )

    try:
        response = await complete_fn(prompt)
        angles: list[str] = []
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("ANGLE:"):
                a = line[6:].strip().strip("'\"-.•*")
                if a and len(a) > 2:
                    angles.append(a[:80])
        return angles[:max_angles]
    except Exception:
        return []


def _prepare_sections(
    sections: list[CorpusSection],
    max_workers: int,
    max_section_chars: int,
) -> list[CorpusSection]:
    """Merge small sections and split large ones to fit worker count.

    Args:
        sections: Raw corpus sections.
        max_workers: Maximum number of workers.
        max_section_chars: Maximum chars per section.

    Returns:
        Prepared sections list (at most ``max_workers`` entries).
    """
    if not sections:
        return []

    # Cap sections to max_workers by merging smallest
    if len(sections) > max_workers:
        sections = sorted(sections, key=lambda s: s.char_count, reverse=True)
        while len(sections) > max_workers:
            smallest = sections.pop()
            second_smallest = sections.pop()
            merged = CorpusSection(
                title=f"{second_smallest.title} + {smallest.title}",
                content=f"{second_smallest.content}\n\n{smallest.content}",
            )
            sections.append(merged)
        sections = sorted(sections, key=lambda s: s.char_count, reverse=True)

    # Split sections that are too large
    final_sections: list[CorpusSection] = []
    for section in sections:
        if section.char_count <= max_section_chars:
            final_sections.append(section)
        else:
            parts = re.split(r'\n\s*\n', section.content)
            current_chunk: list[str] = []
            current_size = 0
            part_num = 1
            for part in parts:
                if current_size + len(part) > max_section_chars and current_chunk:
                    final_sections.append(CorpusSection(
                        title=f"{section.title} (part {part_num})",
                        content="\n\n".join(current_chunk),
                    ))
                    part_num += 1
                    current_chunk = []
                    current_size = 0
                current_chunk.append(part)
                current_size += len(part)
            if current_chunk:
                final_sections.append(CorpusSection(
                    title=f"{section.title} (part {part_num})" if part_num > 1 else section.title,
                    content="\n\n".join(current_chunk),
                ))

    return final_sections[:max_workers]


async def score_section_angle_pairs(
    sections: list[CorpusSection],
    angles: list[str],
    complete_fn,
) -> list[list[float]] | None:
    """Use LLM to score how well each section matches each angle.

    Returns an NxM score matrix where N=sections, M=angles.
    Higher scores mean better fit.  Returns ``None`` if the LLM call
    fails or produces unparseable output, so the caller can fall back
    to keyword matching.

    The LLM sees truncated previews of each section (first 500 chars)
    to keep the prompt compact.  This costs 1 extra LLM call.

    Args:
        sections: Prepared corpus sections.
        angles: Detected research angles.
        complete_fn: Async LLM completion callable.

    Returns:
        Score matrix (list of lists of floats, 0-10 scale), or ``None``
        on any failure.
    """
    n_sections = len(sections)
    n_angles = len(angles)

    # Build compact section previews
    section_descs = []
    for i, s in enumerate(sections):
        preview = s.content[:2000].replace("\n", " ")
        section_descs.append(f"S{i}: \"{s.title}\" — {preview}")

    angle_descs = []
    for j, a in enumerate(angles):
        angle_descs.append(f"A{j}: {a}")

    prompt = (
        "You are an assignment optimizer. Score how well each SECTION "
        "matches each ANGLE for a research synthesis task. Each section "
        "should go to the angle whose specialist would extract the MOST "
        "value from it — not just keyword overlap, but deep semantic "
        "relevance.\n\n"
        "SECTIONS:\n" + "\n".join(section_descs) + "\n\n"
        "ANGLES:\n" + "\n".join(angle_descs) + "\n\n"
        f"Return a JSON array of {n_sections} arrays, each with {n_angles} "
        "scores (0-10). Example for 2 sections, 3 angles:\n"
        "[[8, 2, 5], [1, 9, 3]]\n\n"
        "Return ONLY the JSON array, no explanation:"
    )

    try:
        response = await complete_fn(prompt)
        # Extract JSON array from response (handle markdown code blocks)
        cleaned = response.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        scores = json.loads(cleaned)

        # Validate shape
        if (
            isinstance(scores, list)
            and len(scores) == n_sections
            and all(
                isinstance(row, list) and len(row) == n_angles
                for row in scores
            )
        ):
            # Clamp to 0-10 and convert to float
            return [
                [max(0.0, min(10.0, float(v))) for v in row]
                for row in scores
            ]

        logger.warning(
            "section_count=<%d>, angle_count=<%d> | "
            "LLM score matrix has wrong shape, falling back to keyword",
            n_sections, n_angles,
        )
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        logger.warning(
            "error=<%s> | LLM score matrix parse failed, falling back to keyword",
            exc,
        )
    except Exception as exc:
        logger.warning(
            "error=<%s> | LLM scoring call failed, falling back to keyword",
            exc,
        )

    # Return None so caller falls back to keyword matching
    return None


def _optimal_assignment(
    scores: list[list[float]],
    n_sections: int,
    n_angles: int,
) -> list[int]:
    """Solve optimal section-to-angle assignment from a score matrix.

    Uses the Hungarian algorithm (scipy) for optimal bipartite matching
    that maximizes total assignment quality.  Falls back to greedy
    assignment if scipy is not available.

    Args:
        scores: NxM score matrix (higher = better fit).
        n_sections: Number of sections.
        n_angles: Number of angles.

    Returns:
        List of angle indices, one per section.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

        # Hungarian algorithm minimizes cost, so negate scores
        # Handle case where sections != angles by padding
        max_dim = max(n_sections, n_angles)
        cost_matrix = [[0.0] * max_dim for _ in range(max_dim)]
        for i in range(n_sections):
            for j in range(n_angles):
                cost_matrix[i][j] = -scores[i][j]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Extract assignment for actual sections
        assignment = [0] * n_sections
        for r, c in zip(row_ind, col_ind):
            if r < n_sections:
                assignment[r] = c if c < n_angles else r % n_angles
        return assignment

    except ImportError:
        logger.info("scipy not available, using greedy assignment")

    # Greedy fallback: for each section, pick the highest-scoring
    # unused angle.  If more sections than angles, angles repeat.
    used: set[int] = set()
    assignment = []
    for i in range(n_sections):
        row = scores[i]
        # Sort angles by score descending
        ranked = sorted(range(n_angles), key=lambda j: row[j], reverse=True)
        # Pick the best unused angle, or the best overall if all used
        chosen = ranked[0]
        for j in ranked:
            if j not in used:
                chosen = j
                break
        assignment.append(chosen)
        used.add(chosen)
    return assignment


def assign_workers(
    sections: list[CorpusSection],
    angles: list[str] | None = None,
    max_workers: int = 6,
    max_section_chars: int = 30000,
    score_matrix: list[list[float]] | None = None,
) -> list[WorkerAssignment]:
    """Assign corpus sections to workers.

    Three assignment strategies (in order of preference):

    1. **Optimal (score_matrix provided)**: Uses pre-computed LLM scores
       and Hungarian algorithm for globally optimal assignment.
    2. **Keyword (no score_matrix, angles provided)**: Substring matching
       between section titles and angle names.
    3. **Positional (no angles)**: Section titles become angle names.

    Sections larger than ``max_section_chars`` are split into sub-sections.
    Small sections are merged together.

    Args:
        sections: Raw corpus sections from ``detect_sections()``.
        angles: Detected research angles (from LLM or structural).
        max_workers: Maximum number of workers.
        max_section_chars: Maximum chars per worker section.
        score_matrix: Optional NxM score matrix from
            ``score_section_angle_pairs()``.  If provided, uses optimal
            assignment instead of keyword matching.

    Returns:
        List of WorkerAssignment objects, one per worker.
    """
    if not sections:
        return []

    if not angles:
        angles = [s.title for s in sections]

    final_sections = _prepare_sections(sections, max_workers, max_section_chars)
    if not final_sections:
        return []

    # Determine section-to-angle mapping
    if score_matrix is not None and len(score_matrix) == len(final_sections):
        # Optimal assignment via score matrix
        angle_indices = _optimal_assignment(
            score_matrix, len(final_sections), len(angles),
        )
        logger.info(
            "sections=<%d>, angles=<%d> | using optimal semantic assignment",
            len(final_sections), len(angles),
        )
    else:
        # Keyword fallback
        angle_indices = []
        for i, section in enumerate(final_sections):
            best_idx = i % len(angles)
            for j, angle in enumerate(angles):
                if (
                    angle.lower() in section.title.lower()
                    or section.title.lower() in angle.lower()
                ):
                    best_idx = j
                    break
            angle_indices.append(best_idx)

    # Create worker assignments with unique angle keys
    assignments: list[WorkerAssignment] = []
    used_angles: dict[str, int] = {}

    for i, section in enumerate(final_sections):
        angle_idx = angle_indices[i]
        best_angle = angles[angle_idx] if angle_idx < len(angles) else section.title

        if best_angle in used_angles:
            used_angles[best_angle] += 1
            unique_angle = f"{best_angle} (part {used_angles[best_angle]})"
        else:
            used_angles[best_angle] = 1
            unique_angle = best_angle

        assignments.append(WorkerAssignment(
            worker_id=i,
            angle=unique_angle,
            raw_content=section.content,
            angle_idx=angle_idx,
        ))

    return assignments


def apply_misassignment(
    assignments: list[WorkerAssignment],
    score_matrix: list[list[float]] | None = None,
    ratio: float = 0.25,
) -> list[WorkerAssignment]:
    """Inject off-angle raw data into each worker's slice.

    Each worker receives its full on-angle content PLUS a portion of the
    most distant angle's raw content.  The off-angle data is where thread
    discovery happens: the molecular bee reads practitioner data and its
    worldview activates on details the practitioner overlooked.

    Distance heuristic:
    - If a score matrix is available, the most distant angle for worker i
      is the one whose section scored LOWEST for worker i's assigned angle.
    - Without a score matrix, use maximum positional distance (worker 0
      pairs with worker N//2, etc.).

    Args:
        assignments: Worker assignments from ``assign_workers()``.
        score_matrix: Optional NxM score matrix from
            ``score_section_angle_pairs()``.  Used to determine angle distance.
        ratio: Fraction of the distant worker's raw content to inject
            as off-angle data.  Default 0.25 (25%).

    Returns:
        The same assignments list, mutated in-place with off-angle data
        appended to ``raw_content``.
    """
    n = len(assignments)
    if n < 2 or ratio <= 0:
        return assignments

    # Determine the most distant worker for each worker
    distant_map: dict[int, int] = {}

    if score_matrix is not None and len(score_matrix) >= n:
        # Use score matrix: for worker i, find the worker whose assigned
        # angle scored LOWEST for worker i's section (= most semantically
        # distant).  score_matrix is sections x angles, so we must look up
        # each worker's actual angle_idx, not assume worker j = angle j.
        for i in range(n):
            if i >= len(score_matrix):
                distant_map[i] = (i + n // 2) % n
                continue
            row = score_matrix[i]
            min_score = float("inf")
            min_worker = (i + n // 2) % n  # default fallback
            for j in range(n):
                if j == i:
                    continue
                # Look up worker j's actual angle column in the score matrix
                j_angle_idx = assignments[j].angle_idx
                if j_angle_idx < 0 or j_angle_idx >= len(row):
                    angle_score = 5.0  # neutral fallback
                else:
                    angle_score = row[j_angle_idx]
                if angle_score < min_score:
                    min_score = angle_score
                    min_worker = j
            distant_map[i] = min_worker
    else:
        # Positional distance: pair workers maximally apart
        for i in range(n):
            distant_map[i] = (i + n // 2) % n

    # Snapshot original content before mutation to prevent cascading injection
    original_contents: dict[int, str] = {
        i: a.raw_content for i, a in enumerate(assignments)
    }

    # Inject off-angle content
    for i, assignment in enumerate(assignments):
        distant_idx = distant_map[i]
        distant_worker = assignments[distant_idx]
        distant_content = original_contents[distant_idx]

        # Calculate how much off-angle content to inject
        inject_chars = int(len(distant_content) * ratio)
        if inject_chars < 100:
            continue  # too little to be useful

        off_angle_content = distant_content[:inject_chars]

        assignment.raw_content = (
            f"{assignment.raw_content}\n\n"
            f"═══ OFF-ANGLE DATA (from {distant_worker.angle} specialist's "
            f"raw findings — interpret through YOUR domain) ═══\n"
            f"{off_angle_content}"
        )
        assignment.char_count = len(assignment.raw_content)

    injected = sum(1 for i in range(n) if "OFF-ANGLE DATA" in assignments[i].raw_content)
    logger.info(
        "misassignment workers=<%d>, ratio=<%.2f> | "
        "off-angle data injected into %d workers",
        n, ratio, injected,
    )

    return assignments
