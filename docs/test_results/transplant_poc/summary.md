# Context Transplant Proof-of-Concept — Summary

**Date:** 2026-04-23 22:02 UTC
**Status:** Reconstructed from user-provided spec (canonical script not on this VM).
**Models:** bee=gemini-2.5-flash, architect=gemini-2.5-flash, judge=gemini-2.5-pro

## Setup

- Corpus: 41 unique insulin/bodybuilding pharmacology findings from Phase 1A
- Query: insulin timing protocol for 90kg male strength athlete (6-week plan)
- Phase 1 (bee): Gemini Flash produces structured reasoning trace over corpus
- Phase 2 (architect + transplant): Gemini Flash answers query with bee trace + raw corpus in context
- Phase 3 (control): Gemini Flash answers query with raw corpus only, no bee trace
- Judge: Gemini Pro scores A (Phase 2) vs B (Phase 3) on specificity, citation fidelity, safety coverage, gap honesty

## Token economics

| Phase | Prompt tokens | Completion tokens | Latency |
|---|---:|---:|---:|
| 1 (bee)       | 1211 | 237 | 30257.0 ms |
| 2 (architect) | 1412 | 880 | 20655.0 ms |
| 3 (control)   | 1118 | 854 | 16559.0 ms |

## Judge verdict

```json
{
  "A": {
    "specificity": 5,
    "citation_fidelity": 5,
    "safety_coverage": 4,
    "gap_honesty": 5,
    "total": 19
  },
  "B": {
    "specificity": 5,
    "citation_fidelity": 5,
    "safety_coverage": 5,
    "gap_honesty": 5,
    "total": 20
  },
  "winner": "B",
  "rationale": "Both answers are of exceptionally high quality, providing specific, actionable, and well-cited protocols. They both score perfectly on specificity, citation fidelity, and gap honesty, with each identifying a different but equally valid limitation in the corpus. The deciding factor is safety coverage. Answer B is marginally superior because it includes a critical safety warning about the potential interaction between Metformin and insulin, a detail that Answer A omits. In pharmacology, a more comprehensive safety profile makes for a clinically superior answer."
}
```

## Interpretation

The transplanted-bee configuration (Phase 2) **did not beat** the no-transplant control (Phase 3) in this first POC run — they tied 19-vs-20, with the control winning by 1 point on safety-coverage. Both answers were rated "exceptionally high quality" by the Gemini-Pro judge. Three plausible causes worth investigating before concluding anything about the transplant technique itself:

1. **Hidden-reasoning leakage.** Gemini 2.5 Flash returned only 237 completion tokens of bee trace (816 chars) despite consuming 7,207 total tokens in Phase 1 — almost all of the actual reasoning happened in hidden thinking tokens that never reached the architect. The transplanted trace was essentially a short summary, not the full chain-of-thought we intended to move. Switching to Gemini 2.0 Flash (non-thinking) or OpenRouter with thinking tokens exposed should produce a fuller trace.
2. **Corpus too small to benefit from pre-processing.** 41 findings / 3.2 K characters fits trivially inside the architect's context; there's no summarisation burden to offload to a bee. The technique should only show wins on corpora large enough that the architect would otherwise drown (~50–500 K tokens).
3. **Query not hard enough.** "Design a 6-week insulin protocol" is well inside Flash's single-shot capability. A harder multi-hop query (e.g., "find cross-domain contraindications between findings 3, 17, and 29 that aren't explicit in any single finding") would stress-test whether a bee trace usefully pre-wires the retrieval step.

This run therefore serves as a **methodology smoke test** (the three-phase harness works end-to-end) and a **negative baseline** (transplant ≠ automatic win on trivial corpora). Recommended follow-ups: (a) re-run with Gemini 2.0 Flash or a model whose reasoning tokens are fully emitted, (b) scale the corpus to ≥100 K tokens, (c) replace the query with a multi-hop retrieval task.

## Reconstruction notes

The canonical script at `/home/ubuntu/context_transplant_poc.py` was not present on this VM when the task was picked up (cross-VM filesystem gap — this is a child Devin session). The script used here (`context_transplant_poc.py` in the repo root of the run) was reconstructed from the natural-language spec the user provided. Swap in the canonical version when available; the three-phase structure is preserved so downstream comparisons remain valid.

## Files

- `corpus.md` — shared corpus (41 findings, 3228 chars)
- `phase1_bee.json` — bee reasoning trace + metadata
- `phase2_architect.json` — architect answer with transplant
- `phase3_control.json` — architect answer without transplant (control)
- `judge.json` — LLM-as-judge scorecard
