# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""KnowledgePlugin — persistent cross-session knowledge accumulation.

Hooks into the agent lifecycle to:
1. BeforeInvocationEvent: retrieve relevant past knowledge for the
   current query and inject it as context so the agent doesn't repeat
   work it has already done.
2. AfterInvocationEvent: extract key findings from the conversation
   and store them so future queries can benefit.

The plugin uses the singleton KnowledgeStore (DuckDB + FTS) so all
knowledge persists across server restarts and conversation sessions.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from strands.hooks.events import AfterInvocationEvent, BeforeInvocationEvent
from strands.plugins import Plugin, hook
from strands.types.content import Message

from knowledge_store import Entity, Insight, KnowledgeStore, get_knowledge_store

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_KNOWLEDGE_MARKER = "[PRIOR KNOWLEDGE]"

# Maximum number of insights to inject per query
_MAX_INJECT = 8

# Minimum confidence for injected insights
_MIN_INJECT_CONFIDENCE = 0.3

# Minimum word count for a fact to be worth storing
_MIN_FACT_WORDS = 5


class KnowledgePlugin(Plugin):
    """Accumulate and retrieve persistent knowledge across sessions.

    On each invocation:
    - Before: searches past knowledge for the current query and injects
      relevant findings as context (if any exist)
    - After: extracts factual claims from the assistant's response and
      stores them for future retrieval

    The plugin is designed to be lightweight — extraction uses simple
    heuristics (sentence splitting + source URL detection) rather than
    an LLM call, so it adds zero latency and zero cost.
    """

    name: str = "knowledge"

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        max_inject: int = _MAX_INJECT,
        min_inject_confidence: float = _MIN_INJECT_CONFIDENCE,
    ) -> None:
        """Initialize the knowledge plugin.

        Args:
            store: KnowledgeStore instance. Uses the global singleton
                if not provided.
            max_inject: Maximum past insights to inject per query.
            min_inject_confidence: Minimum confidence for injection.
        """
        super().__init__()
        self._store = store or get_knowledge_store()
        self._max_inject = max_inject
        self._min_inject_confidence = min_inject_confidence
        self._current_query: str = ""

    @hook
    def inject_knowledge(self, event: BeforeInvocationEvent) -> None:
        """Retrieve and inject relevant past knowledge before invocation.

        Searches the knowledge store for insights related to the current
        query and prepends them as a context block. This gives the agent
        a head start on topics it has researched before.
        """
        if event.messages is None:
            return

        query = self._extract_query(event.messages)
        if not query:
            return

        self._current_query = query

        # Strip any stale knowledge markers from previous turns
        msgs = [
            msg for msg in event.messages
            if not self._is_knowledge_message(msg)
        ]

        # Search for relevant past knowledge
        insights = self._store.search_insights(
            query=query,
            limit=self._max_inject,
            min_confidence=self._min_inject_confidence,
        )

        if not insights:
            event.messages = msgs
            return

        # Build knowledge context block
        knowledge_text = self._format_knowledge_block(insights)

        knowledge_msg: Message = {
            "role": "user",
            "content": [{"text": f"{_KNOWLEDGE_MARKER}\n{knowledge_text}"}],
        }

        # Insert before the last user message
        insert_idx = len(msgs) - 1
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                insert_idx = i
                break
        msgs.insert(insert_idx, knowledge_msg)
        event.messages = msgs

        logger.info(
            "injected=<%d>, query=<%s> | prior knowledge injected",
            len(insights),
            query[:80],
        )

    @hook
    def accumulate_knowledge(self, event: AfterInvocationEvent) -> None:
        """Extract and store key findings from the conversation.

        Scans the assistant's response for factual statements and stores
        them as insights. Also extracts named entities for cross-session
        tracking.
        """
        if not self._current_query:
            return

        # Get the last assistant message
        messages = event.messages if event.messages else []
        assistant_text = self._extract_assistant_response(messages)

        if not assistant_text:
            return

        # Extract and store insights
        facts = self._extract_facts(assistant_text)
        stored_count = 0
        for fact_text, source_url in facts:
            if self._store.has_similar_insight(fact_text):
                continue

            insight = Insight(
                fact=fact_text,
                source_url=source_url,
                source_type=self._infer_source_type(source_url),
                topic=self._infer_topic(self._current_query),
                confidence=0.7,
                query_context=self._current_query[:500],
            )
            self._store.store_insight(insight)
            stored_count += 1

        # Extract and store entities
        entities = self._extract_entities(assistant_text)
        for name, etype in entities:
            entity = Entity(
                name=name,
                entity_type=etype,
            )
            self._store.store_entity(entity)

        if stored_count > 0 or entities:
            logger.info(
                "insights=<%d>, entities=<%d> | knowledge accumulated",
                stored_count,
                len(entities),
            )

    # ── Formatting ────────────────────────────────────────────────────

    @staticmethod
    def _format_knowledge_block(insights: list[dict]) -> str:
        """Format insights into a concise context block for the agent.

        Args:
            insights: List of insight dicts from the knowledge store.

        Returns:
            Formatted text block ready for injection.
        """
        lines = [
            "PRIOR KNOWLEDGE (from previous research sessions):",
            "The following facts were gathered in earlier conversations.",
            "Use them as a starting point — verify if needed, don't repeat the search.\n",
        ]
        for i, ins in enumerate(insights, 1):
            conf = ins.get("confidence", 0.0)
            src = ins.get("source_url", "")
            topic = ins.get("topic", "")
            fact = ins.get("fact", "")

            line = f"{i}. {fact}"
            meta_parts = []
            if topic:
                meta_parts.append(f"topic: {topic}")
            if src:
                meta_parts.append(f"source: {src}")
            if conf < 0.5:
                meta_parts.append("low confidence — verify")
            if meta_parts:
                line += f" [{', '.join(meta_parts)}]"
            lines.append(line)

        return "\n".join(lines)

    # ── Extraction helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_query(messages: list[dict] | None) -> str:
        """Extract the user's query from the last user message."""
        if not messages:
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        # Skip injected markers
                        if text and _KNOWLEDGE_MARKER not in text:
                            return text
            elif isinstance(content, str) and _KNOWLEDGE_MARKER not in content:
                return content
        return ""

    @staticmethod
    def _extract_assistant_response(messages: list[dict]) -> str:
        """Extract text from the last assistant message."""
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                if texts:
                    return " ".join(texts)
            elif isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _extract_facts(text: str) -> list[tuple[str, str]]:
        """Extract factual statements from assistant response text.

        Uses heuristics: sentences that contain specific claims (numbers,
        dates, named entities, citations) are more likely to be worth
        storing than generic filler text.

        Args:
            text: The assistant's response text.

        Returns:
            List of (fact_text, source_url) tuples.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        facts: list[tuple[str, str]] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < _MIN_FACT_WORDS:
                continue

            # Skip meta-commentary and hedging
            lower = sentence.lower()
            if any(skip in lower for skip in [
                "let me", "i'll", "i will", "i can", "here are",
                "here is", "based on", "according to my", "as an ai",
                "i found", "i searched", "searching for",
                "let's", "shall we", "would you",
            ]):
                continue

            # Check for factual indicators
            has_number = bool(re.search(r'\d+', sentence))
            has_url = bool(re.search(r'https?://\S+', sentence))
            has_year = bool(re.search(r'\b(19|20)\d{2}\b', sentence))
            has_proper_noun = bool(re.search(r'[A-Z][a-z]{2,}', sentence))
            has_unit = bool(re.search(r'\b(mg|kg|ml|%|µg|nmol|mmol)\b', sentence, re.IGNORECASE))

            indicators = sum([has_number, has_url, has_year, has_proper_noun, has_unit])
            if indicators < 1:
                continue

            # Extract source URL if present
            url_match = re.search(r'(https?://\S+)', sentence)
            source_url = url_match.group(1).rstrip(".,;)") if url_match else ""

            # Clean up the fact text (remove URLs for cleaner storage)
            fact_text = re.sub(r'https?://\S+', '', sentence).strip()
            fact_text = re.sub(r'\s+', ' ', fact_text)

            if len(fact_text.split()) >= _MIN_FACT_WORDS:
                facts.append((fact_text, source_url))

        return facts

    @staticmethod
    def _extract_entities(text: str) -> list[tuple[str, str]]:
        """Extract named entities from text using pattern matching.

        Extracts:
        - Chemical compounds (e.g., GLP-1, BPC-157, Tirzepatide)
        - Organizations (e.g., FDA, WHO, NIH)
        - People (capitalized multi-word names)

        Args:
            text: The assistant's response text.

        Returns:
            List of (entity_name, entity_type) tuples.
        """
        entities: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Chemical/drug compounds (hyphenated alphanumeric patterns)
        compounds = re.findall(
            r'\b([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)\b', text
        )
        for c in compounds:
            key = c.lower()
            if key not in seen and len(c) > 3:
                seen.add(key)
                entities.append((c, "compound"))

        # Well-known abbreviations/organizations (3+ uppercase letters)
        orgs = re.findall(r'\b([A-Z]{3,})\b', text)
        # Filter out common non-entity abbreviations
        skip_orgs = {"THE", "AND", "FOR", "NOT", "BUT", "WITH", "FROM",
                     "HAS", "WAS", "ARE", "THIS", "THAT", "WILL", "CAN",
                     "URL", "PDF", "HTML", "JSON", "API", "HTTP", "HTTPS"}
        for org in orgs:
            if org not in skip_orgs and org.lower() not in seen:
                seen.add(org.lower())
                entities.append((org, "organization"))

        return entities

    @staticmethod
    def _infer_source_type(url: str) -> str:
        """Infer source type from URL domain."""
        if not url:
            return "research"
        lower = url.lower()
        if any(d in lower for d in ["pubmed", "ncbi.nlm"]):
            return "academic"
        if any(d in lower for d in ["arxiv.org"]):
            return "preprint"
        if any(d in lower for d in ["clinicaltrials.gov", "fda.gov", "sec.gov"]):
            return "government"
        if any(d in lower for d in ["reddit.com"]):
            return "forum"
        if any(d in lower for d in ["youtube.com", "youtu.be"]):
            return "video"
        if any(d in lower for d in [".edu", "scholar.google"]):
            return "academic"
        return "research"

    @staticmethod
    def _infer_topic(query: str) -> str:
        """Extract a short topic from the user query.

        Takes the first 3-5 significant words as the topic tag.
        """
        # Remove common question words
        stop_words = {
            "what", "how", "why", "when", "where", "who", "which",
            "is", "are", "was", "were", "do", "does", "did",
            "can", "could", "would", "should", "will",
            "the", "a", "an", "of", "in", "on", "at", "to", "for",
            "and", "or", "but", "not", "with", "from", "by",
            "find", "search", "look", "tell", "me", "about",
            "give", "show", "explain", "describe",
            "papers", "research", "info", "information", "data",
            "latest", "recent", "new", "current", "results",
        }
        words = [
            w for w in query.lower().split()
            if w not in stop_words and len(w) > 2
        ]
        return " ".join(words[:4]) if words else ""

    @staticmethod
    def _is_knowledge_message(msg: dict) -> bool:
        """Check whether a message contains the knowledge marker."""
        if not isinstance(msg, dict) or msg.get("role") != "user":
            return False
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and _KNOWLEDGE_MARKER in block.get("text", ""):
                    return True
        elif isinstance(content, str) and _KNOWLEDGE_MARKER in content:
            return True
        return False
