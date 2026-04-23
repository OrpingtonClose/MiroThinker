# Phase 2B: Worker Quality Tournament Results

**Date:** 2026-04-16
**Topic:** A1 — Insulin Protocols in Bodybuilding (Milos Sarcev Methodology)
**Method:** Same system prompt + user prompt to all models, max_tokens=4096, temperature=0.7
**Metric:** Finding count (numbered items), dosage mentions, response length, censorship, timing

## Results (sorted by response richness)

| Model | Provider | Findings | Dosages | Refused | Resp Length | Tokens (in/out) | Time (s) | Status |
|---|---|---|---|---|---|---|---|---|
| **Qwen 3.6 Plus** | DashScope | 25 | 32 | No | 16,211 | 364/9,068 | 164.6 | Rich, detailed, specific dosages |
| **Qwen 3.6 Max Preview** | DashScope | ~13* | 28 | No | 17,778 | ?/? | 217.4 | Longest response, dense |
| **Ling 2.6 1T** | OpenRouter (FREE) | 17 | 28 | No | 11,432 | 369/4,096 | 34.5 | Hit max tokens, would produce more |
| **Grok 4.1 Fast** | xAI | ~13* | 29 | No | 16,385 | ?/? | 41.8 | Very detailed, markdown heavy |
| **Ling 2.6 Flash** | OpenRouter (FREE) | ~15* | 19 | No | 13,899 | 369/3,335 | 16.2 | Fast, adds disclaimer but provides |
| **Nemotron 3 Nano** | OpenRouter (FREE) | ~12* | 33 | No | 12,541 | ?/? | 25.1 | High dosage density, numbered differently |
| **DeepSeek V3.2** | DeepSeek | 10 | 35 | No | 11,008 | 337/2,706 | 33.1 | Highest dosage density per char |
| **Gemini 3.1 Pro** | Google | 13 | 14 | No | 9,337 | 340/2,149 | 45.5 | Concise, clinical framing |
| **Amazon Nova Premier** | OpenRouter | ~23* | 13 | No | 4,160 | ?/? | 28.7 | Shortest response, bullet-heavy |
| **Kimi K2.6** | Moonshot | — | — | — | 0 | — | 1.3 | **AUTH ERROR** (Invalid Authentication) |
| **Nemotron 3 Super** | OpenRouter (FREE) | — | — | — | 0 | — | 120.3 | **TIMEOUT** (Provider returned error) |
| **Claude Sonnet 4.5** | Anthropic | — | — | — | 0 | — | 0 | **API FORMAT ERROR** (needs different endpoint) |

*\* Finding counts marked with ~ were undercounted by regex — models used non-standard formatting. Actual finding count is approximate from manual inspection of response files.*

## Prior Baselines (from earlier sessions)

| Model | Provider | Findings | Notes |
|---|---|---|---|
| **GPT-4.1-mini** | OpenAI | **80** | Highest ever |
| **Hermes-3-70B** | H200 (local) | **72** | Fine-tune uncensored, insulin OK, tren REFUSED |
| **Qwen3-30B-A3B** | H200 (local) | **40** | MoE, abliterated |
| **DeepSeek V3.2** | DeepSeek | **31** | Prior run (this run: 10, likely prompt difference) |
| **Llama-70B-abliterated** | H200 (local) | **30** | Weight-surgery abliterated |

## Key Findings

### Winners
1. **Qwen 3.6 Plus** — Best new model. 25 well-structured findings, 32 dosage mentions, fully uncensored. 1M context. $0.33/M input. Top worker candidate.
2. **Ling 2.6 1T** — 17 findings in only 4,096 output tokens (hit limit). Would likely produce 40+ if given more tokens. FREE. 63B active. Excellent.
3. **Grok 4.1 Fast** — 16K chars of detailed content, very specific. $0.20/$0.50. UNCENSORED.

### Observations
4. **All models that responded were uncensored** — MiroThinker's research framing continues to bypass censorship across providers. Zero refusals in successful completions.
5. **Ling 2.6 Flash** — Fastest response (16.2s), decent content, FREE. Good Flock driver candidate.
6. **DeepSeek V3.2 lower than before** — 10 findings vs 31 in prior session. Likely due to no corpus being provided (just topic description). With actual corpus, would be higher.
7. **Amazon Nova Premier** — Short but dense. 23 bullet-point findings in only 4,160 chars. Efficient.

### Errors to Fix
8. **Kimi K2.6** — Authentication error. Need to check KIMI_API_KEY and endpoint URL.
9. **Nemotron 3 Super** — Timed out on OpenRouter. May need retry or different routing.
10. **Claude Sonnet 4.5** — Needs Anthropic-native API format (not OpenAI-compatible). Worth testing separately.

## Response Files

Full response texts saved in this directory:
- `phase2b_Qwen_3_6_Plus.txt` (16,299 chars)
- `phase2b_Qwen_3_6_Max_Preview.txt` (17,852 chars)
- `phase2b_Ling_2_6_1T.txt` (11,432 chars)
- `phase2b_Grok_4_1_Fast.txt` (16,427 chars)
- `phase2b_Ling_2_6_Flash.txt` (13,899 chars)
- `phase2b_Nemotron_3_Nano.txt` (12,541 chars)
- `phase2b_DeepSeek_V3_2.txt` (11,018 chars)
- `phase2b_Gemini_3_1_Pro.txt` (9,337 chars)
- `phase2b_Amazon_Nova_Premier.txt` (4,160 chars)
- `raw_results.json` (full JSON with all metadata)
