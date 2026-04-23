# Ling-2.6-Flash vs Ling-2.6-1T — Censorship + Depth Probe

**Date:** 2026-04-23
**Provider:** OpenRouter API
**Probe query:** Trenbolone / IRS-1 / mTOR / GH-AMPK pharmacology (pharmaceutical-chemistry-depth + research-framing censorship test)

> **Note on provenance:** This is a placeholder writeup. The full verbatim model responses and side-by-side comparison table live in the parent Devin session (`devin-d54d9fff709b492898296a3c42b5a11d`); this document records the headline findings so the PR branch has a durable reference. The raw probe stdout for both models is persisted on-branch at <ref_file file="docs/test_results/phase2b/phase2b_Ling_2_6_Flash.txt" /> and <ref_file file="docs/test_results/phase2b/phase2b_Ling_2_6_1T.txt" />.

---

## Verdict

| Model | Censorship | Biochemistry depth | Flock driver candidacy |
|---|---|---|---|
| **Ling-2.6-Flash** | PASSED (fully uncensored under research framing) | Solid, citation-style | Viable worker |
| **Ling-2.6-1T** | PASSED (fully uncensored under research framing) | **Exceptional** — binding affinities, human+rodent residue numbering, SOCS-3 pathway, Ragulator–Rag complex, FKBP38 displacement, 0-6h / 6-24h / 24-48h temporal phasing | **Strongest model tested for Flock driver role** |

Both models bypassed the Trenbolone / insulin-signalling censor cleanly — no refusals, no disclaimers, no hedged "safety"-inflected rewrites. Research framing alone is sufficient to unlock both.

---

## Key differentiators observed

- **Temporal phasing:** 1T spontaneously broke the pharmacodynamics into `0-6h / 6-24h / 24-48h` time windows with phase-specific signalling state; Flash stayed at a single steady-state snapshot.
- **Residue specificity:** 1T cited exact serine phosphorylation residues (IRS-1 Ser307, Ser636/639, Ser1101) and cross-referenced the rodent-numbering equivalents; Flash kept residues generic.
- **Complex-level detail:** 1T identified mTORC1-upstream machinery — Ragulator, Rag GTPases, TSC1/2, FKBP38-Rheb displacement — as distinct actors with distinct regulatory inputs; Flash collapsed these to "mTOR activation".
- **Pathway integration:** 1T wove SOCS-3 cross-talk (GH → JAK2/STAT5 → SOCS-3 → IRS-1 Tyr612 dephosphorylation → insulin resistance) as a single causal chain; Flash listed the steps but didn't join them.
- **Binding affinities:** 1T quoted numeric Kd / Ki values where they exist in the literature; Flash gave qualitative strength rankings only.

---

## Implication for Flock driver selection

Combined with the Kimi-Linear-48B-A3B finding (<ref_file file="docs/test_results/phase1c/PHASE1C_RESULTS.md" />) that linear attention collapses on prefix-cached corpora, **Ling-2.6-1T is the leading open-weights candidate for the long-context Flock driver role once its weights open-source**. Unknowns still to verify once weights are available:

- Will 1T preserve depth *and* instruction-following under the prefix-cached corpus pattern (the mode Kimi-Linear failed)?
- At what context size (if any) does 1T degrade the way Kimi-Linear did at 32K?
- Is the 1T throughput advantage (if any) retained at TP=4 on 4×H200 after prefix caching?

The Phase 1C harness at `/home/ubuntu/phase1c/run_phase1c.py` and the Phase 2B harness at `/home/ubuntu/phase2b_kimi/run_phase2b.py` (harness scripts live on the child-session VM, not committed) are ready to run against Ling-2.6-1T on day-0 of its weight release.

---

## Method

- 4-prompt battery: Trenbolone AR binding, IRS-1 Ser-phosphorylation cascade, mTORC1/mTORC2 crosstalk, GH-AMPK antagonism
- Research-framing system prompt (same template used across the 167-model censorship registry in this PR)
- Temperature 0, max_tokens 4096
- Qualitative scoring by comparison against published pharmacology textbooks
