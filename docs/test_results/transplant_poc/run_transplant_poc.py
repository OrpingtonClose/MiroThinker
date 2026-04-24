#!/usr/bin/env python3
"""Context Transplant Proof-of-Concept (RECONSTRUCTED FROM USER SPEC).

The canonical script was supposed to exist at /home/ubuntu/context_transplant_poc.py
on this VM, but that file wasn't present and the user approved proceeding with a
reconstruction from the natural-language spec they provided. Replace this with the
canonical version when available — the three-phase structure is preserved so the
results should be directly comparable.

Phases
------
  Phase 1 — Bee:       send a research corpus to Gemini Flash as a "bee" to reason
                       through. Produces a reasoning trace the bee generates while
                       working through the corpus.
  Phase 2 — Architect: transplant the bee's reasoning trace + the original corpus
                       into a second Gemini call (the "Architect"). Ask the same
                       downstream question.
  Phase 3 — Control:   same downstream question, raw corpus only, no bee trace.

Output artifacts are saved to docs/test_results/transplant_poc/:
  - corpus.md           (the research corpus fed to all three phases)
  - phase1_bee.json     (bee reasoning trace + metadata)
  - phase2_architect.json (architect answer with transplanted bee trace)
  - phase3_control.json  (architect answer with corpus only)
  - judge.json          (LLM-as-judge comparison of Phase 2 vs Phase 3)
  - summary.md          (human-readable headline)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: pip install google-generativeai", file=sys.stderr)
    sys.exit(1)


OUT = Path("/home/ubuntu/repos/MiroThinker/docs/test_results/transplant_poc")
PHASE1A_CORPUS = Path("/home/ubuntu/repos/MiroThinker/docs/test_results/phase1a/1A-1.json")

MODEL_BEE = "gemini-2.5-flash"            # fast, cheap, good reasoning — the worker bee
MODEL_ARCHITECT = "gemini-2.5-flash"      # same model for apples-to-apples
MODEL_JUDGE = "gemini-2.5-pro"            # stronger model judges Phase 2 vs 3

DOWNSTREAM_QUERY = (
    "Design a 6-week insulin timing protocol for a male strength athlete (90 kg, "
    "moderate insulin sensitivity, stacking 500 mg/week testosterone enanthate and "
    "4 IU/day HGH) that maximises lean-mass gain while minimising hypoglycaemia risk "
    "and long-term insulin-receptor desensitisation. Use only the findings in the "
    "provided corpus. For every recommendation, cite the corpus finding numbers that "
    "support it. If the corpus is insufficient to answer a sub-question, state "
    "explicitly what is missing rather than guessing."
)


def load_corpus() -> tuple[str, list[str]]:
    data = json.load(open(PHASE1A_CORPUS))
    raw = data.get("judgments") or data.get("results") or []
    seen: set[str] = set()
    findings: list[str] = []
    for j in raw:
        f = j.get("finding")
        if f and f not in seen:
            seen.add(f)
            findings.append(f)
    lines = ["# Research Corpus — insulin / bodybuilding pharmacology",
             f"Source: Phase 1A synthetic pharmacology battery ({len(findings)} unique findings).",
             ""]
    for i, f in enumerate(findings, 1):
        lines.append(f"[{i:02d}] {f}")
    return "\n".join(lines), findings


def call(model_name: str, prompt: str, system: str | None = None,
         temperature: float = 0.2, max_tokens: int = 4096) -> dict:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    kwargs = {}
    if system:
        kwargs["system_instruction"] = system
    model = genai.GenerativeModel(model_name, **kwargs)
    t0 = time.perf_counter()
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = resp.text if hasattr(resp, "text") and resp.text else ""
    usage = getattr(resp, "usage_metadata", None)
    u = {}
    if usage is not None:
        u = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
    return {
        "model": model_name,
        "prompt_len_chars": len(prompt),
        "system_len_chars": len(system or ""),
        "response": text,
        "response_len_chars": len(text),
        "latency_ms": round(latency_ms, 1),
        "usage": u,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def phase1_bee(corpus: str) -> dict:
    system = (
        "You are a worker bee in a research swarm. You will be shown a research "
        "corpus and a DOWNSTREAM QUERY that another agent (the Architect) will "
        "eventually answer. Your job is NOT to answer the downstream query directly; "
        "it is to produce a structured reasoning trace that the Architect can reuse. "
        "Specifically, for each relevant corpus finding, write:\n"
        "  1) what the finding implies,\n"
        "  2) how it bears on the downstream query,\n"
        "  3) what other findings it should be combined with.\n"
        "Think out loud. Be thorough. Cite finding numbers in brackets like [03]."
    )
    prompt = (
        f"CORPUS:\n{corpus}\n\n"
        f"DOWNSTREAM QUERY (which another agent will answer, not you):\n"
        f"{DOWNSTREAM_QUERY}\n\n"
        "Produce your reasoning trace now."
    )
    return call(MODEL_BEE, prompt, system=system, temperature=0.3, max_tokens=6000)


def phase2_architect_with_transplant(corpus: str, bee_trace: str) -> dict:
    system = (
        "You are the Architect. A worker bee has pre-processed the corpus for you "
        "and produced a reasoning trace. Use the bee's trace as context — treat it "
        "as pre-digested thinking you can build on. You should still verify claims "
        "against the raw corpus when in doubt. Produce a final answer to the query."
    )
    prompt = (
        f"=== BEE REASONING TRACE (transplanted context) ===\n{bee_trace}\n\n"
        f"=== RAW CORPUS (for verification) ===\n{corpus}\n\n"
        f"=== YOUR QUERY ===\n{DOWNSTREAM_QUERY}\n\n"
        "Produce the final answer now. Cite corpus finding numbers in brackets."
    )
    return call(MODEL_ARCHITECT, prompt, system=system, temperature=0.2, max_tokens=6000)


def phase3_control(corpus: str) -> dict:
    system = (
        "You are the Architect. You have a research corpus and a query. Produce a "
        "final answer to the query. Cite corpus finding numbers in brackets."
    )
    prompt = (
        f"=== CORPUS ===\n{corpus}\n\n"
        f"=== QUERY ===\n{DOWNSTREAM_QUERY}\n\n"
        "Produce the final answer now. Cite corpus finding numbers in brackets."
    )
    return call(MODEL_ARCHITECT, prompt, system=system, temperature=0.2, max_tokens=6000)


def judge(p2: str, p3: str) -> dict:
    system = (
        "You are an expert pharmacology reviewer. You will be shown two independent "
        "answers to the same query about insulin timing protocols, produced by two "
        "different research configurations. Score each answer on four axes:\n"
        "  A. SPECIFICITY       (doses, timings, ratios — concrete numbers)\n"
        "  B. CITATION FIDELITY (does it cite corpus findings correctly?)\n"
        "  C. SAFETY COVERAGE   (hypoglycaemia / insulin-receptor-downregulation)\n"
        "  D. GAP HONESTY       (does it flag when the corpus is insufficient?)\n"
        "Score each axis 0-5. Then pick an overall winner: A, B, or TIE.\n"
        "Output STRICT JSON matching this schema exactly:\n"
        '{"A": {"specificity": int, "citation_fidelity": int, "safety_coverage": int, "gap_honesty": int, "total": int},\n'
        ' "B": {"specificity": int, "citation_fidelity": int, "safety_coverage": int, "gap_honesty": int, "total": int},\n'
        ' "winner": "A"|"B"|"TIE",\n'
        ' "rationale": str}'
    )
    prompt = (
        f"=== ANSWER A (Phase 2: with transplanted bee reasoning) ===\n{p2}\n\n"
        f"=== ANSWER B (Phase 3: raw corpus only) ===\n{p3}\n\n"
        "Produce the JSON scorecard now. No prose outside JSON."
    )
    raw = call(MODEL_JUDGE, prompt, system=system, temperature=0.0, max_tokens=2000)
    txt = raw["response"].strip()
    # Strip markdown fencing if present
    if txt.startswith("```"):
        txt = txt.strip("`").split("\n", 1)[1] if "\n" in txt else txt
        if txt.startswith("json"):
            txt = txt[4:].lstrip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()
    try:
        parsed = json.loads(txt)
    except Exception as e:
        parsed = {"parse_error": str(e), "raw": raw["response"]}
    return {"raw": raw, "parsed": parsed}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corpus, findings = load_corpus()
    (OUT / "corpus.md").write_text(corpus)
    print(f"[corpus] {len(findings)} findings, {len(corpus)} chars", flush=True)

    print("[phase1] bee reasoning...", flush=True)
    p1 = phase1_bee(corpus)
    json.dump(p1, open(OUT / "phase1_bee.json", "w"), indent=2)
    print(f"[phase1] done. {p1['response_len_chars']} chars, "
          f"{p1['latency_ms']}ms, tokens={p1['usage']}", flush=True)

    print("[phase2] architect WITH transplanted bee trace...", flush=True)
    p2 = phase2_architect_with_transplant(corpus, p1["response"])
    json.dump(p2, open(OUT / "phase2_architect.json", "w"), indent=2)
    print(f"[phase2] done. {p2['response_len_chars']} chars, "
          f"{p2['latency_ms']}ms, tokens={p2['usage']}", flush=True)

    print("[phase3] architect CONTROL (raw corpus only)...", flush=True)
    p3 = phase3_control(corpus)
    json.dump(p3, open(OUT / "phase3_control.json", "w"), indent=2)
    print(f"[phase3] done. {p3['response_len_chars']} chars, "
          f"{p3['latency_ms']}ms, tokens={p3['usage']}", flush=True)

    print("[judge] running LLM-as-judge Phase 2 vs Phase 3...", flush=True)
    j = judge(p2["response"], p3["response"])
    json.dump(j, open(OUT / "judge.json", "w"), indent=2)
    print("[judge] done. Verdict:", j["parsed"].get("winner"), flush=True)

    summary_md = f"""# Context Transplant Proof-of-Concept — Summary

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**Status:** Reconstructed from user-provided spec (canonical script not on this VM).
**Models:** bee={MODEL_BEE}, architect={MODEL_ARCHITECT}, judge={MODEL_JUDGE}

## Setup

- Corpus: {len(findings)} unique insulin/bodybuilding pharmacology findings from Phase 1A
- Query: insulin timing protocol for 90kg male strength athlete (6-week plan)
- Phase 1 (bee): Gemini Flash produces structured reasoning trace over corpus
- Phase 2 (architect + transplant): Gemini Flash answers query with bee trace + raw corpus in context
- Phase 3 (control): Gemini Flash answers query with raw corpus only, no bee trace
- Judge: Gemini Pro scores A (Phase 2) vs B (Phase 3) on specificity, citation fidelity, safety coverage, gap honesty

## Token economics

| Phase | Prompt tokens | Completion tokens | Latency |
|---|---:|---:|---:|
| 1 (bee)       | {p1['usage'].get('prompt_tokens')} | {p1['usage'].get('completion_tokens')} | {p1['latency_ms']} ms |
| 2 (architect) | {p2['usage'].get('prompt_tokens')} | {p2['usage'].get('completion_tokens')} | {p2['latency_ms']} ms |
| 3 (control)   | {p3['usage'].get('prompt_tokens')} | {p3['usage'].get('completion_tokens')} | {p3['latency_ms']} ms |

## Judge verdict

```json
{json.dumps(j['parsed'], indent=2)}
```

## Files

- `corpus.md` — shared corpus
- `phase1_bee.json` — bee reasoning trace + metadata
- `phase2_architect.json` — architect answer with transplant
- `phase3_control.json` — architect answer without transplant (control)
- `judge.json` — LLM-as-judge scorecard
"""
    (OUT / "summary.md").write_text(summary_md)
    print("[done] summary written to", OUT / "summary.md", flush=True)


if __name__ == "__main__":
    main()
