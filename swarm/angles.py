# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Angle detection and corpus splitting for the gossip swarm.

Two strategies for assigning work to specialist workers:

1. **Angle-based** (preferred): Detect semantic sections/topics in the corpus
   and assign each to a specialist. Workers get coherent, topic-aligned chunks.
   Scored 9/10 cross-referencing in benchmarks.

2. **Size-based** (fallback): Split the corpus into roughly equal chunks by
   character count. Workers get arbitrary slices. Scored 8/10 in benchmarks.

Angle-based is preferred because it produces better specialist depth and
enables the serendipity bridge to find cross-angle connections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


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


async def detect_angles_via_llm(
    corpus: str,
    query: str,
    complete_fn,
    max_angles: int = 6,
) -> list[str]:
    """Use LLM to identify distinct research angles in the corpus.

    Falls back to structural section titles if LLM fails.
    """
    prompt = (
        "You are a research strategist. Read the following corpus excerpt "
        "and identify the distinct specialist research ANGLES — the separate "
        "facets, disciplines, or lines of inquiry present in this material.\n\n"
        "Each angle should be a concise label (3-8 words) that names a "
        "specific investigative direction. Do NOT return generic labels like "
        "'further research' or 'additional analysis'. Each angle must name "
        "the actual domain, theory, mechanism, or question.\n\n"
        f"Return 2-{max_angles} angles, one per line, prefixed with ANGLE:\n"
        "If the material only has one coherent direction, return just one.\n\n"
        f"USER QUERY: {query}\n\n"
        f"CORPUS (first 8000 chars):\n{corpus[:8000]}"
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


def assign_workers(
    sections: list[CorpusSection],
    angles: list[str] | None = None,
    max_workers: int = 6,
    max_section_chars: int = 30000,
) -> list[WorkerAssignment]:
    """Assign corpus sections to workers.

    If angles are provided (from LLM detection), maps sections to the
    closest matching angle. Otherwise, uses section titles as angles.

    Sections larger than max_section_chars are split into sub-sections.
    Small sections are merged together.
    """
    if not sections:
        return []

    # If no LLM angles, use section titles as angles
    if not angles:
        angles = [s.title for s in sections]

    # Cap sections to max_workers
    if len(sections) > max_workers:
        # Merge smallest sections together
        sections = sorted(sections, key=lambda s: s.char_count, reverse=True)
        while len(sections) > max_workers:
            # Merge the two smallest
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
            # Split into sub-sections at paragraph boundaries
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

    # Cap to max_workers after splitting
    final_sections = final_sections[:max_workers]

    # Create worker assignments
    assignments: list[WorkerAssignment] = []
    for i, section in enumerate(final_sections):
        # Match section to closest angle (simple substring matching)
        best_angle = section.title
        if angles:
            for angle in angles:
                if angle.lower() in section.title.lower() or section.title.lower() in angle.lower():
                    best_angle = angle
                    break
            else:
                # No match — use the angle at this index if available
                if i < len(angles):
                    best_angle = angles[i]

        assignments.append(WorkerAssignment(
            worker_id=i,
            angle=best_angle,
            raw_content=section.content,
        ))

    return assignments
