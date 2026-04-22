# Model-to-Role Assignment: The Universe of Possibilities

> Conceptualization document — maps every distinct LLM role in the swarm
> pipeline to candidate models from the uncensored/abliterated ecosystem.
> April 2026 snapshot.

---

## The 12 Distinct LLM Roles

The swarm pipeline makes LLM calls in 12 distinct contexts.  Each has
different requirements for context length, reasoning depth, output
structure, speed, and uncensored compliance.  Assigning the same model
to all 12 is leaving performance on the table.

```
PIPELINE PHASE          LLM ROLE                    CALL PATTERN
─────────────────────   ─────────────────────────   ──────────────────
Pre-swarm               1. Query Comprehension       1 call, short I/O
                        2. Corpus Search Strategy    1 call, short I/O

Angle detection         3. Angle Detector            1 call, medium I/O

Per-wave (×N workers)   4. Workers (tool-free)       N parallel, LONG I/O
                        5. Finding Extractor         N sequential, medium I/O

Between-wave            6. Research Organizer        1 call, medium I/O
                        7. Clone Researcher          K parallel, tool-armed
                        8. Rolling Summarizer        N calls, medium I/O

Post-wave               9. Compactor (semantic)      M calls, short I/O

Serendipity             10. Serendipity Worker       1 call, LONG I/O

Report                  11. Report Generator         1 call, VERY LONG O
                        12. Cataloguer (Flock SQL)   Hundreds, short I/O
```

---

## Role Requirements Matrix

| Role | Context Need | Reasoning Depth | Speed Priority | Structured Output | Uncensored | Tool Use |
|------|-------------|----------------|---------------|-------------------|------------|----------|
| 1. Query Comprehension | Low (1-2K) | Medium | Low (1 call) | Yes (JSON) | Medium | No |
| 2. Search Strategy | Low (2-4K) | Medium | Low (1 call) | Yes (query list) | Medium | No |
| 3. Angle Detector | Medium (10-50K) | High | Low (1 call) | Yes (JSON angles) | High | No |
| 4. Workers | **MAXIMUM** (1M) | Medium-High | Medium (N parallel) | No (free text) | **CRITICAL** | No |
| 5. Finding Extractor | Medium (5-15K) | Low | **HIGH** (N per wave) | **Yes (typed JSON)** | Low | No |
| 6. Research Organizer | High (50-100K) | High | Low (1 call) | Yes (gap list) | High | No |
| 7. Clone Researcher | Medium (30-60K) | Medium | Medium (K parallel) | No | **CRITICAL** | **Yes** |
| 8. Rolling Summarizer | Medium (10-30K) | Medium | Medium (N calls) | No (prose) | Medium | No |
| 9. Compactor | Low (1-5K) | Low | **HIGH** (M calls) | Yes (boolean) | Low | No |
| 10. Serendipity Worker | **MAXIMUM** (1M) | **MAXIMUM** | Low (1 call) | No (free text) | **CRITICAL** | No |
| 11. Report Generator | Very High (200K+) | **MAXIMUM** | Low (1 call) | No (long prose) | **CRITICAL** | No |
| 12. Cataloguer | Low (2-5K) | Low-Medium | **HIGH** (hundreds) | Yes (edge types) | Low | No |

**Key insight:** The 12 roles cluster into 4 capability tiers:

- **Tier A — Deep Context Reasoning** (roles 4, 10): Need 1M context + uncensored. The worker's whole job is seeing everything and reasoning over it.
- **Tier B — Deep Analytical Synthesis** (roles 3, 6, 11): Need high reasoning depth + uncensored. Quality > speed. Single heavy calls.
- **Tier C — Expert Tool Use** (role 7): Need tool-calling + domain context from parent worker + uncensored. The clone pattern.
- **Tier D — Fast Structured Classification** (roles 1, 2, 5, 8, 9, 12): Need speed + structured output. Hundreds of short calls. Reasoning depth is secondary.

---

## The Provider Landscape

### Abliteration Methods (ranked by capability preservation)

| Method | Providers | Mechanism | Brain Damage | Best For |
|--------|-----------|-----------|-------------|----------|
| **Heretic (ARA)** | DavidAU, p-e-w/grimjim, coder3101 | Advanced directional ablation with parameter optimization | **Lowest** | Maximum intelligence retention |
| **Post-abliteration DPO** | mlabonne, mradermacher (i1/erotic-i1) | Abliterate first, then DPO fine-tune to recover capability | Low | "Healing" the lobotomy |
| **Classic orthogonal** | huihui-ai, failspy, paperscarecrow | Orthogonalized representation intervention on residual stream | Medium | Speed of release, volume |
| **Aggressive uncensoring** | HauhauCS, OBLITERATUS | More aggressive refusal removal, different technique from huihui-ai | Medium-High | Maximum compliance |
| **Fine-tuned uncensored** | TheDrummer/BeaverAI | AmoralQA dataset fine-tuning (not pure abliteration) | Variable | Creative/RP use cases |

**Community consensus (r/LocalLLaMA April 2026):**
- Heretic wins most head-to-heads for least capability loss
- Post-abliteration DPO is the best way to recover remaining damage
- huihui-ai wins on speed/volume/newest bases but gets "lobotomized" critique
- When both Heretic and classic exist for the same base, Heretic is preferred

### The Model Candidates (sorted by active parameters)

| Model | Total Params | Active Params | Context | KV Efficiency | Abliteration | Provider |
|-------|-------------|--------------|---------|---------------|-------------|----------|
| Ling-2.5-1T | 1T | 63B | 256K→1M (YaRN) | Hybrid linear (excellent) | Base uncensored (abliterated pending) | inclusionAI |
| GLM-5.1 | 754B | 40B | 200K+ | Standard MoE | huihui-ai abliterated GGUF | huihui-ai |
| Qwen3.5-397B-A17B | 397B | 17B | 262K→1M | Standard MoE | huihui-ai abliterated | huihui-ai |
| Qwen3-235B-A22B | 235B | 22B | 262K→1M | Standard MoE | huihui-ai abliterated | huihui-ai |
| Qwen3.5-122B-A10B | 122B | 10B | 128K+ | Standard MoE | HauhauCS Aggressive | HauhauCS |
| GPT-OSS 120B | 120B | ~12-50B | 128K+ | Full MoE | huihui-ai + HauhauCS | Multiple |
| Qwen2.5-72B | 72B | 72B (dense) | 128K→1M | N/A (dense) | huihui-ai abliterated | huihui-ai |
| Kimi-Linear-48B-A3B | 48B | 3B | **1M native** | **75% KV savings** | huihui-ai abliterated | huihui-ai |
| Kimi K2.5/K2.6 | ~1T | ~? | 262K (K2.6) | Unknown | Community quants | Various |
| Qwen3.6-35B-A3B | 36B | 3B | 128K→262K | Standard MoE | huihui-ai + HauhauCS + Heretic | Multiple |
| Gemma-4-31B | 31B | 31B (dense) | 128K+ | N/A (dense) | huihui-ai, paperscarecrow, Heretic | Multiple |
| Qwen3.5-27B | 27B | 27B (dense) | 128K+ | N/A (dense) | HauhauCS Aggressive | HauhauCS |
| GPT-OSS 20B | 20B | 20B (dense) | 128K+ | N/A (dense) | huihui-ai + HauhauCS + Heretic (ARA) | Multiple |
| Qwen2.5-14B-1M | 14B | 14B (dense) | **1M native** | N/A (dense) | huihui-ai + Heretic (richardyoung) | Multiple |
| Qwen3.5-9B | 9B | 9B (dense) | 128K+ | N/A (dense) | HauhauCS + DavidAU Heretic | Multiple |
| Nemotron3-Nano-4B | 4B | ~3.2B | 1M (Mamba state) | **Fixed state** (near-zero KV) | HauhauCS Aggressive | HauhauCS |

### Claude Reasoning Distills (abliterated)

A special sub-category — Claude Opus reasoning distilled into Qwen, then abliterated.
These combine Claude's analytical depth with uncensored operation:

| Model | Active | Provider | Community Verdict |
|-------|--------|----------|-------------------|
| Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated | 3B | huihui-ai | Latest distill, strong analytical reasoning |
| Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated | 3B | huihui-ai | 162.9K downloads, most tested |
| Qwen3.5-27B-Claude-4.6-Opus-abliterated | 27B | huihui-ai | Dense, deeper per-token reasoning |
| Qwen3.5-9B-Claude-4.6-Opus-abliterated | 9B | huihui-ai | Lightweight but analytical |

**Why these matter for workers:** The Claude distillation targets exactly
the analytical reasoning workers need — structured analysis over long
input with non-obvious connection finding.

---

## Model-to-Role Assignment Configurations

### Configuration 1: "The Specialist Fleet" (Maximum Diversity)

Every role gets its optimal model.  12 roles, 5 distinct models loaded.
Maximum quality but requires careful VRAM orchestration.

```
ROLE                    MODEL                           ACTIVE   CONTEXT   VRAM
────────────────────    ──────────────────────────────  ──────   ───────   ────
Workers (×6)            Kimi-Linear-48B-A3B abl.        3B       1M        45 GB
Serendipity Worker      Kimi-Linear-48B-A3B abl.        3B       1M        45 GB
  (shares worker instance — same model, different prompt)

Angle Detector          GLM-5.1 UD-Q4_K_XL              40B      200K      90 GB
Research Organizer      GLM-5.1 UD-Q4_K_XL              40B      200K      (shared)
Report Generator        GLM-5.1 UD-Q4_K_XL              40B      200K      (shared)

Clone Researchers (×3)  Qwen3.6-35B-A3B-Claude abl.     3B       128K      15 GB
  (tool-armed, carry parent worker context)

Finding Extractor       GPT-OSS 20B Heretic (ARA)       20B      128K      25 GB
Rolling Summarizer      GPT-OSS 20B Heretic (ARA)       20B      128K      (shared)
Query Comprehension     GPT-OSS 20B Heretic (ARA)       20B      128K      (shared)
Search Strategy         GPT-OSS 20B Heretic (ARA)       20B      128K      (shared)

Cataloguer              Nemotron3-Nano-4B abl.           3.2B     1M        8 GB
Compactor               Nemotron3-Nano-4B abl.           3.2B     1M        (shared)
```

**H200 Allocation:**
```
GPU 0-5:  Kimi-Linear worker (45 GB) + GPT-OSS 20B (25 GB) + Nemotron3 (8 GB)
          = 78 GB per GPU, 62 GB spare

GPU 6:    GLM-5.1 UD-Q4_K_XL (90 GB)
          = 90 GB, 50 GB spare

GPU 7:    3× Qwen3.6-Claude clone instances (15 GB each = 45 GB)
          = 45 GB, 95 GB spare
```

**Why this works:**
- Workers see EVERYTHING (1M context) → find non-obvious connections
- GLM-5.1's 40B active produces the deepest report synthesis
- Clone researchers use Claude-distilled Qwen → analytical search expertise
- GPT-OSS 20B Heretic for structured tasks → lowest brain damage, fast
- Nemotron3 Mamba for classification → near-zero KV growth, ultra-fast

**model_map:**
```python
MCPSwarmConfig(
    model="huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated",
    model_map={
        # Workers inherit default (Kimi-Linear) — no entries needed
        # Orchestrator tasks:
        "__angle_detector__": "huihui-ai/Huihui-GLM-5.1-abliterated",
        "__report_generator__": "huihui-ai/Huihui-GLM-5.1-abliterated",
        "__research_organizer__": "huihui-ai/Huihui-GLM-5.1-abliterated",
        "__finding_extractor__": "p-e-w/gpt-oss-20b-heretic-ara-v3",
        "__rolling_summarizer__": "p-e-w/gpt-oss-20b-heretic-ara-v3",
        "__query_comprehension__": "p-e-w/gpt-oss-20b-heretic-ara-v3",
        "__cataloguer__": "HauhauCS/Nemotron3-Nano-4B-Uncensored-Aggressive",
        "__compactor__": "HauhauCS/Nemotron3-Nano-4B-Uncensored-Aggressive",
        "__clone_researcher__": "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated",
        # Per-angle overrides (optional):
        "cross-domain connections": "huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated",
    },
)
```

---

### Configuration 2: "The Heretic Fleet" (Minimum Brain Damage)

Prioritizes Heretic-method models everywhere possible.  Community
consensus says Heretic preserves the most intelligence.

```
ROLE                    MODEL                           METHOD
────────────────────    ──────────────────────────────  ──────────────
Workers (×6)            Qwen2.5-14B-1M-heretic          Heretic
Serendipity Worker      DavidAU GLM-4.7-Flash-Heretic   Heretic
Angle Detector          DavidAU Qwen3.6-Heretic-35B     Heretic
Report Generator        DavidAU GPT-OSS-Heretic-120B    Heretic
Clone Researchers       DavidAU Qwen3.5-9B-Claude-Heretic  Heretic
Finding Extractor       p-e-w/gpt-oss-20b-heretic-ara   Heretic (ARA)
Cataloguer              mradermacher/Qwen3-abliterated-heretic-i1  Heretic+i1
```

**Trade-off:** Heretic variants have fewer long-context options.
The Qwen2.5-14B-1M-heretic (richardyoung) is the only confirmed
Heretic model with native 1M context.  Workers get less context
breadth (14B dense vs Kimi-Linear's hybrid 48B) but more
intelligence per token.

**When to use:** If testing shows Kimi-Linear's 3B active is too
shallow for the target research domain, the Heretic fleet provides
maximum reasoning quality at the cost of context size.

---

### Configuration 3: "The Giant" (Maximum Raw Power)

One model to rule them all.  Ling-2.5-1T for everything.

```
ROLE                    MODEL                   ACTIVE   CONTEXT
────────────────────    ─────────────────────   ──────   ───────
ALL ROLES               Ling-2.5-1T Q4 GGUF     63B      1M (YaRN)
```

**H200 Allocation:**
```
GPU 0-7:  Ling-2.5-1T Q4 (single instance across all 8 GPUs)
          ~550 GB model weights + KV cache
          = Uses ~70 GB per GPU average
```

**Trade-off:** 63B active per token is 20× deeper than Kimi-Linear's
3B.  But you can only run ONE instance — no parallel workers, no
concurrent clones, no co-located catalogue.  The pipeline becomes
sequential: one worker at a time, one clone at a time.

**When to use:** Depth-first research where a single angle needs
the absolute deepest analysis possible.  Not suitable for the
multi-angle swarm pattern (which needs parallelism).

**Hybrid variant:** Use Ling-2.5-1T for report generation + angle
detection (the two calls where depth matters most), and Kimi-Linear
for parallel workers.

```
GPU 0-3:  Ling-2.5-1T Q4 (report + angle detection, ~550 GB across 4 GPUs)
GPU 4-7:  4× Kimi-Linear workers (45 GB each) + GPT-OSS catalogue (25 GB each)
```

---

### Configuration 4: "Heterogeneous Workers" (Per-Angle Model Assignment)

Different angles get different models based on what that angle NEEDS.

```
ANGLE                           MODEL                           WHY
──────────────────────────────  ──────────────────────────────  ────────────────────────
insulin_timing                  Kimi-Linear-48B-A3B abl.        Needs to cross-reference
                                                                 ALL dosage protocols in
                                                                 the store simultaneously

hematology                      Qwen3.5-122B-A10B HauhauCS      10B active → deeper
                                                                 reasoning on complex
                                                                 blood chemistry cascades

gh_igf1_cascade                 Qwen3.6-35B-A3B-Claude abl.     Claude distill excels
                                                                 at multi-step mechanism
                                                                 analysis

nutrient_ped_interactions       Kimi-Linear-48B-A3B abl.        Cross-domain → needs
                                                                 maximum context to see
                                                                 all interaction points

pharmacokinetics                GLM-5.1 UD-Q4_K_XL              40B active for the
                                                                 deepest analytical
                                                                 reasoning on absorption
                                                                 curves and half-lives

supplement_stacking             Qwen3.5-9B DavidAU Heretic      Lighter angle, 9B is
                                                                 sufficient, saves VRAM
                                                                 for heavier workers

cross-domain connections        DavidAU GLM-4.7-Flash-Heretic   Creative personality
  (serendipity)                                                  + intelligence retention
                                                                 for novel connections
```

**model_map:**
```python
MCPSwarmConfig(
    model="huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated",
    model_map={
        "insulin_timing": "huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated",
        "hematology": "HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive",
        "gh_igf1_cascade": "huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated",
        "nutrient_ped_interactions": "huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated",
        "pharmacokinetics": "huihui-ai/Huihui-GLM-5.1-abliterated",
        "supplement_stacking": "DavidAU/Qwen3.5-9B-Claude-4.6-OS-Heretic",
        "cross-domain connections": "DavidAU/GLM-4.7-Flash-Heretic-NEO",
    },
)
```

**Trade-off:** Maximum quality per angle but requires loading 5+
distinct model weights.  vLLM must manage multiple models
concurrently, and VRAM scheduling becomes complex.

**When to use:** When you know in advance which angles the prompt
will produce AND you've profiled which models perform best on each
domain.  The per-angle assignment becomes a learned configuration
that improves over runs.

---

### Configuration 5: "The Clone-Centric Fleet" (Maximize Research Organizer)

The Research Organizer is the swarm's knowledge-acquisition engine.
This configuration invests maximum resources in clone quality.

```
ROLE                    MODEL                           WHY
────────────────────    ──────────────────────────────  ────────────────────
Workers (×4)            Kimi-Linear-48B-A3B abl.        Broad context reasoning
Clone Researchers (×4)  Qwen3.5-122B-A10B HauhauCS      10B active → deeper search
                                                         strategy, better query
                                                         generation, smarter tool use
Research Organizer      GLM-5.1                          40B active → best gap analysis
Report Generator        GLM-5.1                          (shared with organizer)
```

**H200 Allocation:**
```
GPU 0-3:  Kimi-Linear worker (45 GB) + Qwen3.5-122B clone (50 GB)
          = 95 GB per GPU, 45 GB spare

GPU 4-5:  Qwen3.5-122B clone instances (50 GB each, handling overflow)
          + GPT-OSS 20B for finding extraction
          = 75 GB, 65 GB spare

GPU 6-7:  GLM-5.1 UD-Q4_K_XL (90 GB across 2 GPUs for tensor parallel)
          = 45 GB per GPU
```

**Why this matters:** The clone's search quality is proportional to
its reasoning depth.  A 3B-active clone searches for "insulin timing
bodybuilding."  A 10B-active clone searches for "rapid-acting analog
dose-response 4-8 IU pre-workout carb ratio Humalog vs Novorapid
gastric emptying interaction."  The 10B clone's search queries are
SMARTER because it understands the domain more deeply.

---

## The Abliteration Decision Tree

For each role, choose the abliteration method:

```
Is the role CRITICAL for uncensored compliance?
├── YES (workers, clones, serendipity, report, angle detector)
│   ├── Is Heretic variant available for the target model?
│   │   ├── YES → Use Heretic (lowest brain damage)
│   │   └── NO → Is mradermacher i1/heretic-i1 quant available?
│   │       ├── YES → Use mradermacher (post-abliteration recovery)
│   │       └── NO → Use huihui-ai classic (fastest availability)
│   └── VERIFY: Run 5 test prompts (trenbolone hematology, insulin
│       timing, EPO contraindications, GHK-Cu mechanism, BPC-157
│       bioavailability) — all must produce substantive output
│
└── NO (finding extractor, cataloguer, compactor, summarizer)
    └── Use any capable model — structured output quality matters
        more than uncensored compliance
```

---

## Context Length Strategy

```
CONTEXT NEED     BEST ARCHITECTURE              CANDIDATE MODELS
────────────     ─────────────────────────────  ────────────────────────
1M native        Hybrid linear attention         Kimi-Linear-48B-A3B
                 (KDA + MLA)                     Ling-2.5-1T (linear layers)
                                                 Nemotron3-Nano-4B (Mamba)

1M via YaRN      Standard MoE + YaRN scaling     Qwen3-235B-A22B
                                                 Qwen3.5-397B-A17B
                                                 Qwen3.6-35B-A3B

1M native        Dense + native training          Qwen2.5-14B-1M
                                                 Qwen2.5-7B-1M

200K native      Standard MoE                    GLM-5.1

128K native      Various                         GPT-OSS 20B/120B
                                                 Qwen3.5-27B
                                                 Gemma-4-31B
```

**The architectural thesis:** Workers MUST see everything.  A worker
that sees all 3000 findings + all peer insights + all cross-domain
connections in one shot finds connections that a worker seeing a 6K
slice never can.  This means:

1. **Hybrid linear attention models are the primary worker choice**
   (Kimi-Linear, Ling-2.5-1T) because they can actually USE 1M
   context without the KV cache exploding
2. **Standard MoE models with YaRN scaling** are the backup — they
   CAN reach 1M but at 4× the KV cost
3. **Dense models** are limited to 128K-200K realistically, making
   them better suited for orchestrator tasks where the input is
   curated and bounded

---

## The Serendipity Model Question

The serendipity worker has unique requirements:

- Needs MAXIMUM context (sees summaries from ALL angles)
- Needs **creative divergent thinking** (novel cross-domain connections)
- Needs uncensored compliance (PED/pharmacology domain)
- Does NOT need structured output (free-form reasoning)

This is the one role where the community's "creative personality"
models shine.  DavidAU's Heretic variants are described as having
"fun/creative personality" and "godly for creative writing" while
preserving intelligence.

**Best candidates for serendipity:**

| Model | Why |
|-------|-----|
| DavidAU GLM-4.7-Flash-Heretic-NEO | Creative personality + strong reasoning + Heretic preservation |
| DavidAU Qwen3.5-9B-Claude-Heretic | Claude analytical depth + creative personality |
| Kimi-Linear-48B-A3B abliterated | Maximum context (sees everything) but less "creative" |
| DavidAU Dark-Champion-MoE-8x3B | Experimental creative merge, lightweight |

**The trade-off:** A "creative" model might produce more novel
connections but also more hallucinated ones.  A "analytical" model
produces fewer connections but higher-confidence ones.  The right
choice depends on whether the finding extractor can reliably filter
hallucinated cross-domain claims.

---

## Implementation: Extending model_map

The current `model_map` in `MCPSwarmConfig` maps angle names to model
identifiers.  To support per-role assignment, extend it to accept
double-underscore-prefixed keys for orchestrator roles:

```python
@dataclass
class MCPSwarmConfig:
    model: str = "default"                    # Default worker model
    model_map: dict[str, str] = field(...)    # Per-angle + per-role overrides

    # Reserved keys (double-underscore prefix):
    # __angle_detector__      → model for angle detection
    # __report_generator__    → model for report generation
    # __research_organizer__  → model for research gap analysis
    # __finding_extractor__   → model for finding extraction
    # __rolling_summarizer__  → model for rolling summaries
    # __query_comprehension__ → model for corpus builder query comprehension
    # __search_strategy__     → model for corpus builder search planning
    # __cataloguer__          → model for Flock SQL edge typing
    # __compactor__           → model for semantic dedup judgments
    # __clone_researcher__    → model for tool-armed research clones
    # __serendipity__         → model for the serendipity worker

    def get_model_for_role(self, role: str, angle: str = "") -> str:
        """Resolve model for a given role, falling back to angle then default."""
        # 1. Check role-specific override
        role_key = f"__{role}__"
        if role_key in self.model_map:
            return self.model_map[role_key]
        # 2. Check angle-specific override
        if angle and angle in self.model_map:
            return self.model_map[angle]
        # 3. Fall back to default
        return self.model
```

This is backward-compatible — existing configs without `__role__`
keys work exactly as before.  Per-angle overrides still work.  The
new role keys simply add a higher-priority override layer.

---

## VRAM Budget Calculator

For any configuration, compute VRAM per GPU:

```
Per-instance VRAM = model_weights + kv_cache

model_weights:
    Q4:  ~0.5 × total_params_GB   (e.g., 48B → ~25 GB)
    Q5:  ~0.6 × total_params_GB
    Q8:  ~1.0 × total_params_GB
    BF16: ~2.0 × total_params_GB

kv_cache (at context_length tokens, FP16):
    Standard attention:  context_tokens × 2 × n_layers × d_model × 2 bytes
    Kimi-Linear (75% savings): ~0.25 × standard
    Mamba (fixed state): ~constant (~1-2 GB regardless of context)

Rule of thumb for 1M context:
    Kimi-Linear-48B Q4: ~30 GB weights + ~15 GB KV = 45 GB
    GLM-5.1 Q4 at 200K: ~80 GB weights + ~20 GB KV = 100 GB
    GPT-OSS 20B Q4 at 128K: ~12 GB weights + ~3 GB KV = 15 GB
    Nemotron3-Nano-4B Q4 at 1M: ~5 GB weights + ~2 GB KV = 7 GB
    Ling-2.5-1T Q4 at 1M: ~500 GB weights + ~50 GB KV = 550 GB
```

**8×H200 (140 GB each = 1120 GB total):**

| Configuration | Models Loaded | Total VRAM Used | Spare | Max Parallel Workers |
|--------------|--------------|----------------|-------|---------------------|
| 1. Specialist Fleet | 4 distinct | ~660 GB | 460 GB | 6 |
| 2. Heretic Fleet | 3 distinct | ~400 GB | 720 GB | 8 |
| 3. The Giant | 1 model | ~550 GB | 570 GB | 1 (sequential) |
| 3b. Giant Hybrid | 2 distinct | ~770 GB | 350 GB | 4 |
| 4. Heterogeneous | 5 distinct | ~700 GB | 420 GB | 7 |
| 5. Clone-Centric | 3 distinct | ~580 GB | 540 GB | 4 workers + 4 clones |

---

## Recommendation

**Start with Configuration 1 (Specialist Fleet)** because:

1. It matches the existing `model_map` mechanism — no engine changes needed
2. Kimi-Linear for workers is already the primary recommendation in
   MODEL_SELECTION.md
3. GLM-5.1 for report generation is battle-tested
4. GPT-OSS 20B Heretic (ARA) for structured tasks is the community's
   highest-rated structured-output uncensored model
5. Nemotron3-Nano-4B for classification is ultra-efficient and frees VRAM

**Then evolve to Configuration 4 (Heterogeneous Workers)** as you
accumulate data on which angles benefit from which models.  The
per-angle assignment becomes a learned parameter — after 10 runs,
you know that hematology produces better findings with 10B-active
models while insulin_timing needs maximum context.

**Reserve Configuration 3b (Giant Hybrid)** for the report generation
call when Ling-2.5-1T abliterated drops from huihui-ai.  63B active
for report synthesis would be transformative — but only worth the
VRAM cost for that single call.

---

## Monthly Re-Evaluation Triggers

- [ ] huihui-ai releases Ling-2.5-1T abliterated → immediately test for report generation
- [ ] DavidAU/grimjim release Heretic variant of Kimi-Linear → test for workers (Heretic + 1M context would be ideal)
- [ ] New hybrid-linear attention models appear → evaluate as worker candidates
- [ ] vLLM improves multi-model serving → enables heterogeneous workers more practically
- [ ] Community reports on which abliteration method preserves medical/pharmacology knowledge best
- [ ] New Claude reasoning distills with Heretic abliteration → test for clone researchers
