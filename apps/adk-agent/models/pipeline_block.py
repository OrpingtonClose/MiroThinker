# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Aspect-oriented pipeline block framework.

Every pipeline phase is a fenced ``PipelineBlock`` with declared inputs,
outputs, and typed validation rules.  Cross-cutting concerns (health gates,
DuckDB thread-safety, heartbeats, I/O validation consequences, error
escalation) are applied uniformly as ``Aspect`` instances by the
``PipelineRunner``.

**Key separation of concerns:**

- **Blocks** own business logic + validation RULES (per data type).
- **Aspects** own cross-cutting CONSEQUENCES -- what happens when rules
  fail, when errors occur, when phases start/end.

This mirrors ADK's plugin architecture at the pipeline-phase level.

Architecture::

    PipelineRunner (stateful aspect-application engine)
    +-- aspects: list[Aspect]       <-- applied to EVERY block
    +-- registered blocks
          +-- ScoutBlock / ThinkerBlock / SearchExecutorBlock
          +-- MaestroBlock / SwarmBlock / SynthesiserBlock

    ADK SequentialAgent manages sequencing.
    ADK callbacks delegate to runner.run_block() for each phase.

    For each block:
      1. aspect.before(block, ctx) -- first non-None SHORT-CIRCUITS
      2. block.execute(ctx)        -- business logic only
      3. aspect.after(block, ctx, result) -- reversed
      on error: aspect.on_error(block, ctx, error) -- reversed

Blocks NEVER touch health tracking, DuckDB safety, dashboard events,
or error handling directly.  Those are aspect responsibilities.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.corpus_store import CorpusStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RoutingHint(str, Enum):
    """What a block tells the runner to do next."""
    CONTINUE = "CONTINUE"      # proceed to the next block
    ESCALATE = "ESCALATE"      # exit the current loop (e.g. EVIDENCE_SUFFICIENT)
    ABORT = "ABORT"            # stop the pipeline entirely


class BlockCriticality(str, Enum):
    """How critical a block is -- determines error escalation policy.

    The ErrorEscalationAspect uses this to decide whether to absorb
    or propagate errors.  The runner itself is dumb plumbing.
    """
    CRITICAL = "CRITICAL"        # errors abort the pipeline
    BEST_EFFORT = "BEST_EFFORT"  # errors are logged but pipeline continues


# ---------------------------------------------------------------------------
# I/O Specifications — functional-programming style parameter contracts
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Typed parameter specification for a block's input or output.

    The RULES live here (owned by the block).  The CONSEQUENCES of rule
    failure live in the InputOutputValidationAspect (cross-cutting).
    """
    key: str
    expected_type: type = str
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    required: bool = True       # if False, missing key is OK (optional param)
    default: Any = None         # default value when key is missing and required=False

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this spec.

        Returns (ok, error_message).  Empty error_message when ok=True.
        """
        if value is None and not self.required:
            return True, ""

        if value is None and self.required:
            return False, f"Required key '{self.key}' is missing from state"

        if not isinstance(value, self.expected_type):
            return False, (
                f"Key '{self.key}' has type {type(value).__name__}, "
                f"expected {self.expected_type.__name__}"
            )

        if self.validator is not None:
            try:
                ok = self.validator(value)
            except Exception as exc:
                return False, (
                    f"Validator for '{self.key}' raised: {exc}"
                )
            if not ok:
                desc = self.description or f"validator for '{self.key}'"
                return False, f"Value for '{self.key}' failed validation: {desc}"

        return True, ""


# ---------------------------------------------------------------------------
# Block Context — injected into every block's execute()
# ---------------------------------------------------------------------------

@dataclass
class BlockContext:
    """Everything a block needs to do its work.

    Injected by the PipelineRunner.  Blocks NEVER reach into module
    globals — they receive all dependencies through this context.
    """
    state: dict                                    # session state dict
    corpus: Optional["CorpusStore"] = None         # injected if needs_corpus
    collector: Optional[Any] = None                # dashboard EventCollector
    iteration: int = 0                             # current loop iteration
    user_query: str = ""                           # original research query
    cancel: Optional[threading.Event] = None       # cooperative cancellation
    # Aspect-managed metadata (aspects read/write these, blocks don't)
    _phase_start_time: float = 0.0                 # set by TimingAspect
    _cost_snapshot: float = 0.0                    # set by CostTrackingAspect


# ---------------------------------------------------------------------------
# Block Result — returned by every block's execute()
# ---------------------------------------------------------------------------

@dataclass
class BlockResult:
    """Structured result from a block's execution.

    The block returns metrics and state updates; aspects handle health
    tracking, dashboard events, and error escalation based on these.
    """
    metrics: dict[str, Any] = field(default_factory=dict)
    state_updates: dict[str, Any] = field(default_factory=dict)
    routing: RoutingHint = RoutingHint.CONTINUE
    diagnosis: str = ""         # optional self-diagnosis text


# ---------------------------------------------------------------------------
# PipelineBlock — the fenced phase abstraction
# ---------------------------------------------------------------------------

class PipelineBlock(ABC):
    """A single fenced phase in the pipeline.

    The block ONLY contains business logic and declares its own
    validation rules via ``input_specs`` / ``output_specs``.
    Cross-cutting consequences are handled by aspects.
    """

    name: str = ""
    input_specs: list[ParamSpec] = []
    output_specs: list[ParamSpec] = []
    needs_corpus: bool = False
    criticality: BlockCriticality = BlockCriticality.BEST_EFFORT
    is_looped: bool = False

    @abstractmethod
    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Execute the block's business logic.

        Args:
            ctx: The block context with all dependencies.

        Returns:
            A BlockResult with metrics, state updates, and routing hint.
        """
        ...


# ---------------------------------------------------------------------------
# Aspect — cross-cutting concern applied to every block
# ---------------------------------------------------------------------------

class Aspect(ABC):
    """A cross-cutting concern applied uniformly to every pipeline block.

    Aligned with ADK's BasePlugin pattern:
    - ``before()`` returns ``Optional[BlockResult]``.
      First non-None return **short-circuits** (block won't execute).
    - ``after()`` may modify the result.
    - ``on_error()`` may return an override BlockResult.
    """

    name: str = ""

    async def before(
        self, block: PipelineBlock, ctx: BlockContext,
    ) -> Optional[BlockResult]:
        """Return None to continue, or a BlockResult to short-circuit."""
        return None

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        """Called after block.execute() succeeds.  May modify result."""
        pass

    async def on_error(
        self, block: PipelineBlock, ctx: BlockContext, error: Exception,
    ) -> Optional[BlockResult]:
        """Return None to let runner handle it, or a BlockResult to override."""
        return None


# ---------------------------------------------------------------------------
# PipelineRunner -- stateful aspect-application engine
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Applies aspects uniformly around each pipeline block's execution.

    The runner is NOT a sequencer -- ADK manages the flow.
    The runner provides ``run_block()`` which ADK callbacks invoke.
    """

    def __init__(
        self,
        blocks: list[PipelineBlock],
        aspects: list[Aspect],
    ) -> None:
        self.blocks = blocks
        self.aspects = aspects
        self._block_map = {b.name: b for b in blocks}
        self.consecutive_failures: int = 0

    def get_block(self, name: str) -> Optional[PipelineBlock]:
        """Look up a registered block by name."""
        return self._block_map.get(name)

    async def run_block(
        self,
        block_name: str,
        ctx: BlockContext,
    ) -> BlockResult:
        """Run a named block with all aspects applied.

        Aspect execution order:
          before: aspects[0] .. aspects[N] (first non-None short-circuits)
          after:  aspects[N] .. aspects[0]
          error:  aspects[N] .. aspects[0]
        """
        block = self._block_map.get(block_name)
        if block is None:
            logger.error("Unknown block: '%s'", block_name)
            return BlockResult(
                metrics={"error": f"Unknown block: {block_name}"},
                routing=RoutingHint.ABORT,
            )

        # --- before (in order) ---
        for aspect in self.aspects:
            try:
                short_circuit = await aspect.before(block, ctx)
                if short_circuit is not None:
                    logger.info(
                        "Aspect '%s' short-circuited block '%s'",
                        aspect.name, block.name,
                    )
                    # Run after() for aspects that already ran before()
                    for done in reversed(self.aspects):
                        try:
                            await done.after(block, ctx, short_circuit)
                        except Exception as ae:
                            logger.warning(
                                "Aspect '%s' after() during short-circuit: %s",
                                done.name, ae,
                            )
                        if done is aspect:
                            break
                    return short_circuit
            except Exception as exc:
                logger.warning(
                    "Aspect '%s' before() failed for '%s': %s",
                    aspect.name, block.name, exc,
                )

        # --- execute ---
        result: BlockResult
        try:
            result = await block.execute(ctx)
            self.consecutive_failures = 0
        except Exception as exc:
            self.consecutive_failures += 1

            # --- on_error (reversed) ---
            override: Optional[BlockResult] = None
            for aspect in reversed(self.aspects):
                try:
                    ar = await aspect.on_error(block, ctx, exc)
                    if ar is not None and override is None:
                        override = ar
                except Exception as ae:
                    logger.warning(
                        "Aspect '%s' on_error() failed for '%s': %s",
                        aspect.name, block.name, ae,
                    )

            if override is not None:
                result = override
            else:
                result = BlockResult(
                    metrics={"error": str(exc), "block_failed": True},
                    routing=RoutingHint.CONTINUE,
                    diagnosis=f"Block '{block.name}' failed: {exc}",
                )

            # after() even on error (cleanup)
            for aspect in reversed(self.aspects):
                try:
                    await aspect.after(block, ctx, result)
                except Exception as ae:
                    logger.warning(
                        "Aspect '%s' after() post-error: %s",
                        aspect.name, ae,
                    )
            return result

        # --- after (reversed) ---
        for aspect in reversed(self.aspects):
            try:
                await aspect.after(block, ctx, result)
            except Exception as exc:
                logger.warning(
                    "Aspect '%s' after() failed for '%s': %s",
                    aspect.name, block.name, exc,
                )

        return result

    def apply_state_updates(
        self, ctx: BlockContext, result: BlockResult,
    ) -> None:
        """Write a block's declared state updates back to session state."""
        for key, value in result.state_updates.items():
            ctx.state[key] = value
