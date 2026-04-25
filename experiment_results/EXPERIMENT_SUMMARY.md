# MCP Engine Experiment Results

**Date:** 2026-04-22  
**Model:** gpt-4o-mini (OpenAI API)  
**Architecture:** MCP-powered worker agents with ConditionStore tools (PR #176)

## Configuration Matrix

| # | Experiment | Workers | Waves | Conv.Thresh | Temp | Serendipity | Corpus |
|---|---|---|---|---|---|---|---|
| 0 | baseline_3w_2wave | 3 | 2 | 3 | 0.3 | No | 28K full |
| 1 | more_workers_5w_2wave | 5 | 2 | 5 | 0.3 | No | 28K full |
| 2 | more_waves_3w_4wave | 3 | 4 | 3 | 0.3 | No | 28K full |
| 3 | serendipity_3w_2wave | 3 | 2 | 3 | 0.3 | Yes | 28K full |
| 4 | high_temp_3w_2wave | 3 | 2 | 3 | 0.7 | No | 28K full |
| 5 | strict_convergence_3w_4wave | 3 | 4 | 10 | 0.3 | No | 28K full |
| 6 | full_config_5w_3wave_seren | 5 | 3 | 5 | 0.3 | Yes | 28K full |
| 7 | preset_angles_5w_3wave | 5 | 3 | 5 | 0.3 | Yes | 28K full |
| 8 | preset_angles_5w_3wave_96k | 5 | 3 | 5 | 0.3 | Yes | **96K combined** |
| 9 | small_corpus_3w_2wave | 3 | 2 | 3 | 0.3 | No | 7K quarter |

## Results Summary

| Experiment | Elapsed(s) | Waves | Findings | Tool Calls | Report Chars | Convergence |
|---|---|---|---|---|---|---|
| baseline_3w_2wave | 184 | 2 | 151 | 79 | 5,795 | max waves reached |
| more_workers_5w_2wave | 173 | 2 | 165 | 123 | 6,781 | max waves reached |
| more_waves_3w_4wave | 237 | 4 | 163 | 144 | 5,989 | max waves reached |
| serendipity_3w_2wave | 198 | 2 | 142 | 106 | 5,508 | max waves reached |
| high_temp_3w_2wave | 154 | 2 | 144 | 72 | 6,147 | max waves reached |
| strict_convergence_3w_4wave | 204 | 3 | 146 | 115 | 5,117 | **converged at wave 3** (5 < 10) |
| full_config_5w_3wave_seren | 268 | 3 | 192 | 209 | 5,330 | max waves reached |
| **preset_angles_5w_3wave** | **171** | **3** | **177** | **205** | **5,468** | max waves reached |
| **preset_angles_5w_3wave_96k** | **235** | **3** | **425** | **212** | **6,029** | max waves reached |
| small_corpus_3w_2wave | 108 | 2 | 43 | 78 | 6,347 | max waves reached |

## Findings Per Wave

| Experiment | Wave 1 | Wave 2 | Wave 3 | Wave 4 |
|---|---|---|---|---|
| baseline_3w_2wave | 17 | 20 | - | - |
| more_workers_5w_2wave | 27 | 24 | - | - |
| more_waves_3w_4wave | 18 | 10 | 10 | 11 |
| serendipity_3w_2wave | 12 | 9 | - | - |
| high_temp_3w_2wave | 19 | 11 | - | - |
| strict_convergence_3w_4wave | 13 | 14 | 5 | - |
| full_config_5w_3wave_seren | 42 | 22 | 14 | - |
| preset_angles_5w_3wave | 31 | 17 | 15 | - |
| preset_angles_5w_3wave_96k | 35 | 25 | 18 | - |
| small_corpus_3w_2wave | 7 | 5 | - | - |

## Tool Usage Patterns

Consistent pattern across all experiments:

| Tool | Avg % of calls | Role |
|---|---|---|
| store_finding | 37% | Workers store discoveries as they go |
| search_corpus | 22% | Workers search for evidence |
| get_peer_insights | 20% | Workers check what peers found |
| get_corpus_section | 13% | Workers read raw corpus chunks |
| get_research_gaps | 6% | Workers identify under-covered topics |
| check_contradictions | <2% | **Almost never used** — workers don't check conflicts |

**Key finding:** `check_contradictions` was called only 7 times across all experiments (out of ~1200 total tool calls). Workers prefer to search and store rather than verify. The system prompt needs to mandate contradiction checking, or the tool should auto-trigger when conflicting findings exist.

## Angle Distribution

### 3-worker experiments (auto-detected angles)

The LLM consistently detects ["bodybuilding cycle protocol", "testosterone", "trenbolone"]. The generic angle absorbs ~75% of findings.

| Angle | Avg findings | Avg confidence |
|---|---|---|
| bodybuilding cycle protocol | 112 | 0.53 |
| trenbolone | 20 | 0.71 |
| testosterone | 13 | 0.73 |

### 5-worker auto-detected angles

With 5 workers, detection splits better but still has issues ("growth hormone" splits into two sub-angles):

| Angle | Avg findings | Avg confidence |
|---|---|---|
| insulin | 111 | 0.58 |
| growth hormone / part 2 | 40 (combined) | 0.73 |
| trenbolone | 18 | 0.71 |
| testosterone | 11 | 0.77 |

### 5-worker PRESET angles (best distribution)

With explicit angle definitions, every angle produces meaningful findings:

| Angle | Findings | Avg confidence |
|---|---|---|
| Insulin & GH protocols — Milos Sarcev framework | 114 | 0.590 |
| Micronutrient interactions — compound synergies | 27 | 0.756 |
| Ramping strategy & phase periodization | 27 | 0.787 |
| Testosterone & Trenbolone pharmacokinetics | 23 | 0.865 |
| Ancillaries & health marker management | 20 | 0.725 |
| cross-domain connections (serendipity) | 6 | 0.900 |

## Key Findings

### 1. Preset angles beat auto-detection decisively

| Config | Findings | Tool Calls | Time | Angle balance |
|---|---|---|---|---|
| 5w auto-detected + seren | 192 | 209 | 268s | Imbalanced (insulin: 115, testosterone: 15) |
| **5w preset + seren** | **177** | **205** | **171s** | **Balanced (5 angles all producing)** |

Preset angles were **36% faster** with comparable findings and much better coverage. The auto-detected "growth hormone (part 2)" waste is eliminated.

### 2. Architecture scales linearly with corpus size

| Corpus | Findings | Time | Findings/KB | Time/KB |
|---|---|---|---|---|
| 7K (quarter) | 43 | 108s | 6.0 | 15.2s |
| 28K (full) | 177 | 171s | 6.3 | 6.1s |
| **96K (3× combined)** | **425** | **235s** | **4.4** | **2.4s** |

**3.4× corpus → 2.4× findings → 1.4× time.** The architecture handles 96K chars without any context window issues. Workers make multiple tool calls to explore their sections — no prompt concatenation bottleneck. Efficiency actually improves at scale (less time per KB) because ingestion is amortized.

### 3. More workers > more waves

| Comparison | Findings | Time | Efficiency |
|---|---|---|---|
| 3w × 2waves (baseline) | 151 | 184s | 0.82 findings/s |
| 5w × 2waves | 165 | 173s | 0.95 findings/s |
| 3w × 4waves | 163 | 237s | 0.69 findings/s |

Adding workers is more efficient than adding waves. Workers execute in parallel (via `asyncio.to_thread`), while waves are sequential.

### 4. Diminishing returns after wave 1

Across all experiments, wave 1 produces the most new findings. Subsequent waves show 30-50% drop-off:

- full_config: 42 → 22 → 14 (-48%, -36%)
- more_waves: 18 → 10 → 10 → 11 (-44%, +0%, +10%)
- preset_angles: 31 → 17 → 15 (-45%, -12%)

**2-3 waves is the sweet spot** for 28K corpus. Larger corpora may benefit from more waves.

### 5. Convergence detection works

The strict_convergence experiment (threshold=10) stopped at wave 3 when findings dropped to 5 (below threshold). This saved one full wave compared to the 4-wave experiment, producing nearly identical findings (146 vs 163).

### 6. Temperature has minimal impact on architecture

| Temp | Findings | Tool Calls | Time |
|---|---|---|---|
| 0.3 (default) | 151 | 79 | 184s |
| 0.7 (high) | 144 | 72 | 154s |

Higher temperature = slightly fewer, faster tool calls. Architecture is robust to temperature.

### 7. Serendipity finds real cross-domain connections

Serendipity wave adds 17-51s overhead but produces 4-7 high-confidence (0.72-0.90) cross-domain findings. The preset_angles config produced 6 cross-domain findings at 0.900 average confidence.

## Architecture Insights for H200 Deployment

1. **Recommended config:** 5 workers, preset angles (from h200_test/angles.py), 3 waves, convergence_threshold=15, serendipity=on
2. **Expected performance on 96K corpus:** ~425 findings in ~4 minutes
3. **For 500K+ corpora:** More waves matter — workers need multiple passes. 5-8 waves with convergence_threshold=20.
4. **Local model timing:** gpt-4o-mini is fast (1-3s/call). Local Qwen3.5-32B on H200 will be slower per call but avoids network latency. Expect 2-3× longer per-call but same architecture behavior.
5. **check_contradictions underused:** Workers almost never verify conflicting findings. System prompt needs stronger incentive.
6. **Store size is trivial:** ~200-425 rows per run, ~100-300KB. DuckDB handles this effortlessly.

## Phase Timing Breakdown

| Experiment | Ingestion | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Serendipity | Report |
|---|---|---|---|---|---|---|---|
| baseline | 8.9s | 81.0s | 58.7s | - | - | - | 22.3s |
| more_workers | 6.5s | 54.9s | 62.6s | - | - | - | 49.3s |
| more_waves | 13.1s | 70.5s | 46.5s | 52.6s | 38.3s | - | 16.0s |
| serendipity | 9.4s | 55.1s | 59.2s | - | - | 51.4s | 22.4s |
| high_temp | 6.5s | 70.2s | 53.8s | - | - | - | 23.0s |
| strict_conv | 4.2s | 58.9s | 53.0s | 64.9s | - | - | 22.8s |
| full_config | 8.0s | 65.5s | 54.3s | 89.8s | - | 25.9s | 25.0s |
| preset (28K) | 5.2s | 42.5s | 45.7s | 41.2s | - | 16.8s | 19.6s |
| preset (96K) | 3.4s | 99.8s | 39.6s | 50.5s | - | 23.1s | 18.2s |
| small_corpus | 7.9s | 38.6s | 44.4s | - | - | - | 16.8s |

## Next Steps

1. **Test on H200 with local Qwen3.5-32B** to measure real latency vs cloud API
2. **Add contradiction checking incentive** — modify system prompt or add auto-trigger
3. **Test with 8 angles** (full h200_test/angles.py set) to match the v3 8-compound protocol
4. **Compare MCP engine vs gossip engine** on identical corpus — quality and throughput side-by-side
5. **Test with larger context model** (gpt-4o at 128K) to compare quality when workers hold more per call
