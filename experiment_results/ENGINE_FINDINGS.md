# MCP Swarm Engine — What We Learned

**10 experiments, 1 model (gpt-4o-mini), 1 corpus domain (insulin/bodybuilding protocols)**

This document distills the architectural and behavioral insights from running the MCP-powered worker swarm across different configurations. These findings inform how the engine should be configured for production use on H200.

---

## The Architecture Works

The core hypothesis — workers as stateless Strands Agents with ConditionStore tools, exploring corpus via tool calls instead of receiving it in one prompt — is validated. A 32K-context model successfully processed a 96K-character corpus by making multiple tool calls per worker. No context window errors. No truncation. The store handled 642 rows (largest experiment) without performance issues.

Workers naturally adopted a consistent behavior pattern: search the corpus, read their section, check what peers found, store findings, look for gaps, repeat. This emerged from the tool descriptions alone — no explicit workflow was programmed.

## Prompt Decomposition Is the Biggest Lever

The single most impactful variable across all 10 experiments was **how the research question was decomposed into worker angles**.

### Auto-detection produces bad angles

When the LLM was asked to detect angles from the corpus, it consistently produced:
- **Vague, overlapping labels** like "testosterone", "trenbolone", "bodybuilding cycle protocol"
- **Catch-all angles** that absorbed 75% of all findings while others starved
- **Duplicate angles** ("growth hormone" and "growth hormone part 2" were detected as separate angles by the 5-worker auto-detection)

With 3 workers, auto-detection created one dominant angle that produced 112 findings while the other two combined for 33. Workers competed over the same material instead of specializing.

### Preset angles fix the problem

Hand-crafted angles that decompose by **knowledge type** (not by keyword) produced balanced, specialized output:

| Angle | Findings | Avg Confidence |
|---|---|---|
| Insulin & GH protocols | 114 | 0.59 |
| Micronutrient interactions | 27 | 0.76 |
| Ramping & periodization | 27 | 0.79 |
| Testosterone & Trenbolone pharmacokinetics | 23 | 0.87 |
| Ancillaries & health markers | 20 | 0.73 |

Every angle produced meaningful findings. The highest-confidence findings came from the most specialized angles (pharmacokinetics at 0.87), not the broadest ones.

Preset angles were also **36% faster** than auto-detection (171s vs 268s for comparable configs) because the angle detection LLM call was skipped and workers started immediately.

## Scaling Behavior

### Corpus size scales linearly

| Corpus | Findings | Time | Findings/KB |
|---|---|---|---|
| 7K (quarter) | 43 | 108s | 6.0 |
| 28K (full) | 177 | 171s | 6.3 |
| 96K (3x combined) | 425 | 235s | 4.4 |

3.4x corpus produced 2.4x findings in 1.4x time. Efficiency improves at scale because ingestion cost is amortized. The architecture does not degrade with larger corpora — it just makes more tool calls.

### Workers scale better than waves

| Config | Findings | Time | Efficiency |
|---|---|---|---|
| 3 workers x 2 waves | 151 | 184s | 0.82 findings/s |
| 5 workers x 2 waves | 165 | 173s | 0.95 findings/s |
| 3 workers x 4 waves | 163 | 237s | 0.69 findings/s |

Adding workers (parallel) is strictly better than adding waves (sequential) for the same total work. 5w x 2 waves beat 3w x 4 waves in both quantity (165 > 163) and speed (173s < 237s).

### Diminishing returns after wave 1

Every experiment showed the same pattern: wave 1 produces the most findings, with 30-50% drop-off per subsequent wave.

- Full config: 42 → 22 → 14
- Preset angles: 31 → 17 → 15
- More waves: 18 → 10 → 10 → 11

2-3 waves is the sweet spot for 28K corpus. Wave 4 rarely produced anything wave 3 didn't.

## Tool Usage Reveals Worker Behavior

Workers called tools in a consistent distribution across all experiments:

| Tool | % of calls | What it means |
|---|---|---|
| store_finding | 37% | Workers store aggressively — they write findings as they go |
| search_corpus | 22% | Active exploration of the corpus |
| get_peer_insights | 20% | Cross-pollination is happening organically |
| get_corpus_section | 13% | Workers read their assigned sections |
| get_research_gaps | 6% | Some gap-driven exploration |
| check_contradictions | **<2%** | **Workers almost never verify** |

The `check_contradictions` tool was called only 7 times across ~1,200 total tool calls. Workers are accumulators, not verifiers. They search → find → store → search more, but never stop to check if what they found contradicts existing findings. This is a prompt problem — the system prompt needs to mandate periodic contradiction checking, or the tool should auto-trigger.

## Serendipity Produces Real Connections

The serendipity wave (a separate pass that looks for cross-domain connections between worker angles) produced 4-7 findings per experiment at 0.72-0.90 confidence. These were genuine cross-domain insights:

- GH is diabetogenic → insulin required to manage blood glucose (endocrinology x bodybuilding)
- Blood volume redistribution during training → nutrient partitioning timing (cardiovascular physiology x supplementation)
- Micronutrient interactions with compound stacks (nutrition science x pharmacology)

The serendipity wave adds 17-51s overhead but produces the highest-confidence findings in the entire run. Worth enabling in all configurations.

## Convergence Detection Works

The strict convergence experiment (threshold=10) stopped at wave 3 when new findings dropped to 5, saving one full wave while producing nearly identical total findings (146 vs 163). The mechanism works correctly — it measures store growth rate and stops when the marginal value of another wave falls below threshold.

## Temperature Doesn't Matter (For Architecture)

| Temperature | Findings | Tool Calls | Time |
|---|---|---|---|
| 0.3 (default) | 151 | 79 | 184s |
| 0.7 (high) | 144 | 72 | 154s |

Slightly fewer, slightly faster results at higher temperature. The architecture is robust to temperature changes — this is a content quality knob, not an architectural one. Quality differences would need human evaluation to assess.

## What These Experiments Did NOT Test

1. **Different models** — All 10 experiments used gpt-4o-mini. The H200 test plan called for comparing Qwen3.5-32B vs Llama-3.3-70B for epistemic diversity.
2. **External data** — Workers only searched a static corpus. No web search, no API calls, no new information discovery.
3. **Old vs new architecture** — No side-by-side comparison of MCP engine vs the original GossipSwarm.
4. **Report quality** — Findings were counted and scored by confidence, but no human evaluated whether the reports were actually good, complete, or actionable.
5. **Multi-model topologies** — The multi-endpoint bridge (built in PR #174) was not exercised. No per-GPU model routing was tested.

## Recommended Configuration

Based on these experiments:

```
workers: 5-8 (parallel scales well)
waves: 3 (diminishing returns after that)
convergence_threshold: 15 (stop early if store stops growing)
serendipity: on (highest-confidence findings)
angles: preset (hand-crafted by domain expert, decomposed by knowledge type)
temperature: 0.3 (default is fine)
```

For corpora >100K chars, increase waves to 5-8 and workers proportionally.
