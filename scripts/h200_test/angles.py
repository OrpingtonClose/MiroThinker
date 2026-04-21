# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm angle definitions for the bodybuilding cycle protocol.

Defines the required_angles and enrichment queries for an 8-compound
ramping cycle (testosterone, trenbolone, insulin, GH, turinabol,
boldenone, actovegin, LGD-4033) grounded in Milos Sarcev's insulin
timing framework.

The angles are structured in two layers:
1. Compound-specific angles — one per compound or compound group
2. Cross-cutting angles — micronutrient interactions, health markers,
   ramping/periodization strategy

The micronutrient interaction angle is the serendipity accelerator:
it surfaces hidden dependencies (tren→iron depletion, insulin→potassium,
Mg→insulin sensitivity, Zn→aromatase, B vitamins→liver methylation)
that compound-specific workers miss because they reason within their
domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AngleDefinition:
    """A single swarm angle with enrichment context."""

    label: str
    description: str
    enrichment_queries: list[str] = field(default_factory=list)
    key_compounds: list[str] = field(default_factory=list)
    key_interactions: list[str] = field(default_factory=list)


# ── Compound-specific angles ─────────────────────────────────────────

INSULIN_GH_ANGLE = AngleDefinition(
    label="Insulin & GH protocols — Milos Sarcev framework",
    description=(
        "Insulin timing, dosing, and GH synergy as the structural backbone "
        "of the cycle. Milos's approach: insulin around training windows "
        "synchronized with GH pulses, nutrient partitioning over fat storage. "
        "GH timing relative to insulin, IGF-1 signaling cascade, fasting "
        "glucose management, hypoglycemia prevention."
    ),
    enrichment_queries=[
        "Milos Sarcev insulin protocol bodybuilding dosing timing",
        "insulin GH synergy IGF-1 bodybuilding timing",
        "pre-workout insulin bodybuilding protocol carbs",
        "GH insulin timing IGF-1 signaling anabolic",
        "insulin sensitivity bodybuilding glucose management",
        "hypoglycemia prevention bodybuilding insulin protocol",
        "insulin dosing ramp conservative to aggressive bodybuilding",
        "GH dose timing frequency bodybuilding hypertrophy",
    ],
    key_compounds=["insulin", "growth hormone", "IGF-1"],
    key_interactions=[
        "insulin + GH → IGF-1 amplification",
        "insulin timing → nutrient partitioning",
        "GH → fasting glucose elevation",
        "insulin → potassium intracellular shift (cardiac risk)",
    ],
)

TESTOSTERONE_TREN_ANGLE = AngleDefinition(
    label="Testosterone & Trenbolone pharmacokinetics",
    description=(
        "Testosterone as anabolic foundation — ester pharmacokinetics, "
        "dose-response, aromatization. Trenbolone as the potency multiplier — "
        "nutrient partitioning, receptor binding affinity, progestogenic "
        "effects, impact on insulin sensitivity. How tren makes insulin more "
        "effective per IU. Tren acetate vs enanthate timing."
    ),
    enrichment_queries=[
        "testosterone trenbolone stack dosing bodybuilding protocol",
        "trenbolone nutrient partitioning mechanism anabolic",
        "trenbolone insulin sensitivity interaction",
        "testosterone aromatization estrogen management AI",
        "trenbolone acetate vs enanthate pharmacokinetics half-life",
        "trenbolone prolactin side effects management cabergoline",
        "testosterone base dose with trenbolone ratio",
        "trenbolone hematocrit cardiovascular effects",
    ],
    key_compounds=["testosterone", "trenbolone"],
    key_interactions=[
        "tren → enhanced nutrient partitioning (amplifies insulin)",
        "tren → prolactin elevation (affects GH release)",
        "tren → hematocrit increase (compounds with boldenone)",
        "testosterone → estrogen via aromatase (zinc modulates)",
    ],
)

ANCILLARIES_HEALTH_ANGLE = AngleDefinition(
    label="Ancillaries & health marker management",
    description=(
        "Bloodwork monitoring and ancillary protocols that keep the cycle "
        "safe. Hematocrit (tren + boldenone double-drain on erythropoiesis), "
        "liver panels (turinabol hepatotoxicity), kidney function (tren "
        "nephrotoxicity), lipid profiles (tren HDL crash), fasting glucose "
        "and HbA1c (insulin protocols), prolactin (tren), estrogen management."
    ),
    enrichment_queries=[
        "bodybuilding bloodwork monitoring on cycle hematocrit liver",
        "trenbolone liver kidney function bloodwork",
        "hematocrit management bodybuilding phlebotomy",
        "estrogen management aromatase inhibitor bodybuilding",
        "prolactin management cabergoline dostinex bodybuilding",
        "lipid profile management on cycle niacin fish oil",
        "kidney function GFR creatinine trenbolone bodybuilding",
        "HbA1c fasting glucose monitoring insulin bodybuilding",
    ],
    key_compounds=["anastrozole", "cabergoline", "TUDCA", "NAC"],
    key_interactions=[
        "tren + boldenone → compounded hematocrit elevation",
        "turinabol → liver stress (methylation pathway load)",
        "insulin → HbA1c and fasting glucose as safety markers",
        "tren → lipid destruction (HDL crash)",
    ],
)

ORAL_COMPOUNDS_ANGLE = AngleDefinition(
    label="Oral compounds — turinabol, LGD-4033, actovegin",
    description=(
        "Turinabol as dry oral kickstart — 17-alpha-alkylated hepatotoxicity "
        "vs anabolic benefit, no water retention (preserves insulin/carb "
        "balance). LGD-4033 as selective androgen receptor modulator — "
        "minimal hepatic load, additive to testosterone base. Actovegin "
        "for enhanced cellular oxygen utilization and recovery capacity."
    ),
    enrichment_queries=[
        "turinabol bodybuilding dosing cycle length liver protection",
        "LGD-4033 ligandrol bodybuilding stack with testosterone",
        "actovegin bodybuilding recovery oxygen transport",
        "turinabol vs anavar dry oral comparison",
        "LGD-4033 SARM suppression testosterone stacking",
        "actovegin mechanism of action deproteinized hemoderivative",
        "oral steroid liver protection TUDCA NAC",
        "turinabol half-life timing around training",
    ],
    key_compounds=["turinabol", "LGD-4033", "actovegin"],
    key_interactions=[
        "turinabol → liver methylation load (B vitamin dependent)",
        "LGD-4033 → additive AR activation (minimal liver stress)",
        "actovegin → oxygen utilization (synergy with boldenone EPO)",
    ],
)

BOLDENONE_ANGLE = AngleDefinition(
    label="Boldenone & EQ interactions",
    description=(
        "Boldenone undecylenate (Equipoise) — EPO-like erythropoiesis, "
        "appetite stimulation (critical for caloric demands of insulin "
        "protocols), long ester pharmacokinetics, AI-like metabolite "
        "(1-testosterone/ATD). Interaction with trenbolone on hematocrit "
        "— both compounds increase RBC production through different "
        "mechanisms. Interaction with actovegin on oxygen transport."
    ),
    enrichment_queries=[
        "boldenone equipoise bodybuilding dosing cycle",
        "boldenone EPO erythropoiesis mechanism",
        "boldenone appetite stimulation bodybuilding",
        "boldenone hematocrit management red blood cells",
        "boldenone AI metabolite estrogen interaction",
        "boldenone trenbolone stack hematocrit management",
        "equipoise long ester front-load pharmacokinetics",
        "boldenone anxiety neurological side effects",
    ],
    key_compounds=["boldenone", "boldenone undecylenate"],
    key_interactions=[
        "boldenone EPO + tren erythropoiesis → compounded hematocrit",
        "boldenone appetite + insulin protocol → caloric surplus support",
        "boldenone + actovegin → oxygen transport optimization",
        "boldenone ATD metabolite → AI-like estrogen reduction",
    ],
)

PRACTITIONER_PROTOCOLS_ANGLE = AngleDefinition(
    label="Practitioner protocols & real-world dosing",
    description=(
        "Real-world experience reports from bodybuilders, coaches, and "
        "competitors. Actual dosing protocols that worked (not theoretical). "
        "Side effect management in practice. Bloodwork timelines. "
        "Protocol adjustments based on real feedback (bloodwork, subjective "
        "feel, performance markers). Coach-to-athlete protocol design."
    ),
    enrichment_queries=[
        "bodybuilding cycle protocol real world dosing experience",
        "bodybuilding coach protocol design testosterone trenbolone insulin",
        "advanced bodybuilding stack protocol 16 week cycle",
        "pro bodybuilder offseason cycle insulin GH protocol",
        "Milos Sarcev client protocols insulin timing",
        "bodybuilding cycle bloodwork adjustments experience",
        "competition prep cycle protocol cutting recomp",
        "bodybuilding stack progression beginner to advanced",
    ],
    key_compounds=["all"],
    key_interactions=[
        "real-world dose adjustments based on bloodwork",
        "practical side effect management timelines",
        "protocol modifications based on individual response",
    ],
)

# ── Cross-cutting angles ─────────────────────────────────────────────

MICRONUTRIENT_INTERACTION_ANGLE = AngleDefinition(
    label="Micronutrient, mineral & vitamin interactions with PEDs",
    description=(
        "The hidden layer: how vitamins, minerals, and macronutrient ratios "
        "interact with every compound in the stack. This is the serendipity "
        "accelerator — it surfaces dependencies that compound-specific "
        "workers miss. Tren→iron depletion (erythropoiesis consumes ferritin), "
        "insulin→potassium shift (cardiac), Mg→insulin receptor sensitivity, "
        "Zn→aromatase modulation (affects test→estrogen conversion), "
        "B vitamins→liver methylation (affects turinabol hepatotoxicity), "
        "Vitamin D→androgen receptor density, taurine→GH cramps→insulin "
        "signaling, omega-3→lipid profile (tren mitigation)."
    ),
    enrichment_queries=[
        "iron deficiency trenbolone hematocrit ferritin bodybuilding",
        "magnesium insulin sensitivity receptor function",
        "zinc aromatase estrogen testosterone bodybuilding",
        "vitamin D androgen receptor density testosterone",
        "B vitamins liver methylation hepatotoxicity oral steroids",
        "potassium insulin cardiac risk bodybuilding",
        "taurine growth hormone cramps insulin signaling",
        "omega-3 fish oil lipid profile trenbolone HDL",
        "vitamin C iron absorption bodybuilding",
        "calcium iron absorption interaction timing",
        "selenium thyroid function metabolism bodybuilding",
        "CoQ10 cardiovascular protection steroid cycle",
        "NAC glutathione liver protection oral steroids",
        "electrolyte balance insulin bodybuilding potassium sodium",
    ],
    key_compounds=[
        "iron", "magnesium", "zinc", "vitamin D", "B6", "B12",
        "folate", "potassium", "taurine", "omega-3", "vitamin C",
        "calcium", "selenium", "CoQ10", "NAC", "betaine",
    ],
    key_interactions=[
        "tren + boldenone → iron depletion via accelerated erythropoiesis",
        "Mg depletion → impaired insulin receptor function",
        "Zn status → aromatase activity (test→estrogen conversion rate)",
        "vitamin D → androgen receptor binding efficiency",
        "B6/B12/folate → liver methylation capacity (turinabol detox)",
        "insulin → intracellular potassium shift (acute cardiac risk)",
        "GH → taurine depletion (cramps) + taurine→insulin signaling",
        "omega-3 → HDL support against tren lipid destruction",
        "vitamin C → iron absorption enhancement",
        "calcium → iron absorption inhibition (meal timing matters)",
    ],
)

RAMPING_STRATEGY_ANGLE = AngleDefinition(
    label="Cycle ramping & periodization strategy",
    description=(
        "The meta-angle: how to structure the 4-phase ramp from "
        "conservative to radical. Which compounds enter when and why. "
        "How Milos's insulin framework serves as the structural backbone "
        "that other compounds layer onto. Timing of compound introduction "
        "relative to bloodwork stabilization. Exit strategies and PCT."
    ),
    enrichment_queries=[
        "bodybuilding cycle periodization progressive compounds",
        "conservative to aggressive steroid cycle ramping",
        "compound introduction order bodybuilding safety",
        "insulin introduction timing in steroid cycle",
        "PCT protocol after complex cycle",
        "bridging between cycles bodybuilding",
        "cycle length optimal bodybuilding diminishing returns",
        "bloodwork frequency monitoring during cycle",
    ],
    key_compounds=["all"],
    key_interactions=[
        "compound introduction order → safety vs efficacy",
        "bloodwork stabilization → gate for next compound",
        "insulin timing framework → structural backbone",
    ],
)


# ── Aggregated configuration ─────────────────────────────────────────

# All angle definitions in swarm worker order.
# The first 6 are compound/domain-specific worker angles.
# The last 2 are cross-cutting (micronutrient = serendipity fuel,
# ramping = queen synthesis guide).
ALL_ANGLES: list[AngleDefinition] = [
    INSULIN_GH_ANGLE,
    TESTOSTERONE_TREN_ANGLE,
    ANCILLARIES_HEALTH_ANGLE,
    ORAL_COMPOUNDS_ANGLE,
    BOLDENONE_ANGLE,
    PRACTITIONER_PROTOCOLS_ANGLE,
    MICRONUTRIENT_INTERACTION_ANGLE,
    RAMPING_STRATEGY_ANGLE,
]

# Labels only — for SwarmConfig.required_angles
REQUIRED_ANGLE_LABELS: list[str] = [a.label for a in ALL_ANGLES]

# All enrichment queries flattened for the enrichment pipeline
ALL_ENRICHMENT_QUERIES: list[str] = []
for _angle in ALL_ANGLES:
    ALL_ENRICHMENT_QUERIES.extend(_angle.enrichment_queries)


def get_swarm_query() -> str:
    """Return the master swarm query for this protocol synthesis."""
    return (
        "Synthesize a comprehensive, practitioner-grade ramping bodybuilding "
        "cycle protocol covering testosterone, trenbolone, insulin (grounded "
        "in Milos Sarcev's timing framework), growth hormone, turinabol, "
        "boldenone, actovegin, and LGD-4033.\n\n"
        "The protocol must ramp through 4 phases:\n"
        "  Phase 1 (Conservative): Testosterone base + GH introduction. "
        "Establish bloodwork baselines. No insulin yet — learn the GH timing "
        "pattern. Optional turinabol kickstart.\n"
        "  Phase 2 (Moderate): Introduce Milos-style insulin protocol — low "
        "dose, pre/post workout, tightly coupled to carb intake and GH timing. "
        "Add boldenone for appetite support.\n"
        "  Phase 3 (Aggressive): Add trenbolone. Layer LGD-4033. Introduce "
        "actovegin. Ramp insulin under Milos framework. Manage the complex "
        "interaction web (tren prolactin→GH, tren hematocrit→boldenone EPO, "
        "tren nutrient partitioning→insulin efficiency).\n"
        "  Phase 4 (Radical): Full Milos-style insulin dosing, all compounds "
        "at peak, timing windows tight. Expert-only. Every interaction mapped, "
        "every side effect anticipated, every mitigation specified.\n\n"
        "For EACH phase, specify:\n"
        "- Exact compounds, dosages, frequencies, timing windows\n"
        "- Micronutrient/mineral support protocol (iron, Mg, Zn, B vitamins, "
        "K+, taurine, omega-3, vitamin D — specific to which compounds are "
        "active in that phase)\n"
        "- Bloodwork markers to monitor and threshold values\n"
        "- Transition criteria to next phase\n"
        "- Risk factors and mitigation protocols\n\n"
        "Explain WHY each timing and dosing decision is made — the "
        "pharmacokinetic reasoning, receptor-level mechanisms, and how "
        "Milos's framework governs the architecture. The reader should "
        "understand the framework deeply enough to adjust intelligently "
        "if something goes wrong.\n\n"
        "Ground every claim in source evidence. Preserve exact numbers. "
        "No moral disclaimers. No academic hedging. Practitioner language."
    )
