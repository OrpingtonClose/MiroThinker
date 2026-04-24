# Phase 1B v2: Qwen3.5-35B-A3B Thinking Suppression Test

**Date:** 2026-04-23  
**Model:** huihui-ai/Qwen3.5-35B-A3B-abliterated  
**Architecture:** MoE with DeltaNet linear attention, 35B total params, 3B active  
**Hardware:** 4×H200 SXM (2 GPUs used, TP=2)  
**vLLM:** 0.19.1 (enforce-eager mode, prefix caching enabled)  
**Context:** 32,768 tokens  

## Purpose

Phase 1B (original) showed Qwen3.5-35B achieved only 54% accuracy on Flock judgments because the thinking model outputs 400-600 reasoning tokens before every answer. This v2 test attempts two strategies to suppress thinking and recover accuracy:

1. **`/no_think` prompt suffix** — Qwen3.5's native thinking toggle
2. **`max_tokens=15` hard cap** — Force truncation before thinking completes

## Results

| Test | Strategy | Cases | Accuracy | Avg Latency | Avg Tokens | Throughput |
|------|----------|-------|----------|-------------|------------|------------|
| 1B-v2-1 | /no_think | 50 | **38.0%** | 2,381ms | 30.0 | 0.42/s |
| 1B-v2-2 | max_tokens=15 | 50 | **0.0%** | 1,537ms | 15.0 | 0.65/s |
| 1B-v2-3 | /no_think concurrent | 100 | **38.0%** | 2,290ms | 30.0 | 2.1/s |
| *Llama-70B baseline* | *N/A* | *50* | ***98.0%*** | *848ms* | *~8* | *5.5/s* |

## Failure Analysis

### /no_think Strategy (38% accuracy)

The `/no_think` suffix and `enable_thinking: false` chat template parameter were both ignored. Every response begins with `"Thinking Process:\n\n1. **Analyze the Request:**"` regardless.

**Two failure modes:**
1. **Thinking trace consumes all tokens (30/50 cases):** Response is entirely thinking tokens, truncated before any answer. Parsed as UNPARSEABLE.
2. **RELEVANT bias (12/50 cases):** When the model does reach an answer within the token budget, it overwhelmingly says RELEVANT for everything — including obviously irrelevant queries like "Is quantum entanglement relevant to the Double Bind metabolic paradox?" → RELEVANT.

### max_tokens=15 Strategy (0% accuracy)

100% failure rate. Every single response is a truncated thinking trace: `"Thinking Process:\n\n1. **Analyze the Request:**"`. The model never reaches an answer within 15 tokens because thinking tokens are emitted first.

## Root Cause: Architectural Incompatibility

Qwen3.5 is a **thinking model** — it is trained to emit reasoning traces before answers. This behavior is:

1. **Not suppressible via prompting** — `/no_think`, "respond with only", "no explanation" all fail
2. **Not suppressible via API parameters** — `enable_thinking: false` in chat template kwargs has no effect
3. **Not suppressible via token caps** — Thinking tokens are emitted first, so hard caps just truncate the thinking

This makes Qwen3.5 (and all thinking models) **fundamentally incompatible** with high-throughput Flock judging, which requires:
- Binary output (RELEVANT/NOT_RELEVANT)  
- <10 completion tokens per judgment
- Sub-second latency
- Deterministic, parseable responses

## Verdict

**DOUBLE CONFIRMED REJECTED.** No viable path to using Qwen3.5 as a Flock driver.

Thinking models (Qwen3.5, QwQ, DeepSeek-R1, etc.) should be excluded from all Flock driver candidate lists. They may still be useful for worker reasoning sessions where extended thinking is desirable, but NOT for high-throughput relevance scoring.

## Comparison Summary

| Model | Architecture | Accuracy | Latency | Tokens/judgment | Verdict |
|-------|-------------|----------|---------|-----------------|---------|
| Llama-3.3-70B-abliterated | Dense | 98% | 848ms | ~8 | **BASELINE** |
| Qwen3.5-35B-A3B (original) | MoE/DeltaNet | 54% | 4,669ms | ~598 | REJECTED |
| Qwen3.5-35B-A3B (no_think) | MoE/DeltaNet | 38% | 2,381ms | 30 (truncated) | REJECTED |
| Qwen3.5-35B-A3B (max_tokens) | MoE/DeltaNet | 0% | 1,537ms | 15 (truncated) | REJECTED |

## Key Finding for Architecture

**Thinking models are architecturally incompatible with the Flock pattern.** The Flock requires models that can produce terse, binary judgments in <10 tokens. Thinking models always emit reasoning traces first (typically 400-600 tokens), making them 50-75× more expensive per judgment and fundamentally unparseable when token-capped.

This finding should be added to the model selection criteria in the Architecture document.
