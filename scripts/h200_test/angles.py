# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm angle definitions for the insulin/GH/tren medical-manual extraction.

12 angles structured for biochemistry-level granularity:

Layer 1 — Compound-specific (6 angles):
    1. Insulin pharmacology & Milos Sarcev protocol
    2. GH/IGF-1 axis & extreme dosing (120 IU context)
    3. Testosterone & trenbolone pharmacokinetics
    4. Oral compounds (turinabol, LGD-4033, actovegin)
    5. Boldenone & EQ interactions
    6. PED stack synergies at biochemistry level

Layer 2 — Nutrient biochemistry (4 angles):
    7. Specific fatty acid biochemistry (DHA, EPA, AA, CLA, GLA, MCTs...)
    8. Amino acid biochemistry & timing (leucine/mTOR, glutamine, taurine...)
    9. Carbohydrate biochemistry & insulin windows (dextrose, HBCD, waxy maize...)
   10. Micronutrients, minerals & vitamins (individual B vitamins, D3+K2, forms...)

Layer 3 — Cross-cutting (2 angles):
   11. Sulphur biochemistry, methylation & detox pathways (serendipity angle)
   12. Peptide pharmacology (BPC-157, TB-500, CJC-1295, ipamorelin...)

Anti-hallucination constraint: the research query encodes a structural honesty
requirement.  The queen synthesis uses confidence tiers (ESTABLISHED, PROBABLE,
SPECULATIVE, UNKNOWN) rather than presenting all claims at equal weight.
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


# ── Layer 1: Compound-specific angles ────────────────────────────────

INSULIN_PROTOCOL_ANGLE = AngleDefinition(
    label="Insulin pharmacology & Milos Sarcev protocol",
    description=(
        "Medical-manual precision on insulin in bodybuilding.  Milos Sarcev's "
        "framework: insulin around training windows synchronized with GH "
        "pulses, nutrient partitioning over fat storage.  Dosing tables — "
        "conservative (5-10 IU), moderate (10-15 IU), aggressive (15-25+ IU) "
        "with exact carbohydrate gram-per-IU ratios.  Timing protocols: "
        "pre-workout, intra-workout, post-workout windows.  Glucometer "
        "monitoring protocol — when to measure, what thresholds trigger "
        "intervention.  Hypoglycemia emergency protocol — exact glucose "
        "thresholds, fast-acting carb quantities, when to call emergency "
        "services.  Insulin types: Humalog (lispro) vs NovoRapid (aspart) "
        "vs regular (Humulin R) — onset, peak, duration curves and why "
        "each matters for training window design.  Fasting glucose and "
        "HbA1c management over multi-week protocols.  The Milos approach "
        "to progressive insulin introduction — never start aggressive."
    ),
    enrichment_queries=[
        "Milos Sarcev insulin protocol bodybuilding exact dosing timing",
        "insulin GH synergy IGF-1 bodybuilding timing protocols",
        "pre-workout insulin bodybuilding protocol carbs per IU ratio",
        "insulin sensitivity bodybuilding glucose monitoring glucometer",
        "hypoglycemia prevention emergency protocol bodybuilding insulin",
        "insulin dosing ramp conservative to aggressive bodybuilding weekly",
        "Humalog vs NovoRapid vs Humulin R bodybuilding onset peak duration",
        "insulin types rapid acting short acting bodybuilding pharmacokinetics",
        "Milos Sarcev insulin carbohydrate loading protocol exact grams",
        "insulin bodybuilding fasting glucose HbA1c long-term monitoring",
        "insulin nutrient partitioning mechanism GLUT4 translocation",
        "insulin potassium intracellular shift cardiac risk bodybuilding",
    ],
    key_compounds=["insulin lispro", "insulin aspart", "regular insulin", "IGF-1"],
    key_interactions=[
        "insulin + GH → IGF-1 amplification (hepatic and local)",
        "insulin timing → GLUT4 translocation → nutrient partitioning",
        "GH → fasting glucose elevation (counter-regulatory)",
        "insulin → intracellular potassium shift (acute cardiac risk)",
        "insulin + tren → amplified nutrient partitioning",
        "insulin type (rapid vs short) → training window design",
    ],
)

GH_IGF1_EXTREME_ANGLE = AngleDefinition(
    label="GH/IGF-1 axis & extreme dosing — 120 IU context",
    description=(
        "Dedicated analysis of growth hormone at extreme doses.  How can "
        "Andrei sustain 120 IU GH?  What organ strain does this create?  "
        "GH pharmacology at supraphysiological doses: pulsatile vs "
        "continuous secretion patterns, receptor downregulation, GH "
        "resistance.  IGF-1 cascade — hepatic IGF-1 vs autocrine/paracrine "
        "local IGF-1, IGF-1 LR3 (long-acting analog).  Acromegalic "
        "features: visceral organ growth (GH gut/palumboism), bone "
        "remodeling, connective tissue thickening, cardiomegaly risk.  "
        "GH timing relative to insulin — the competitive glucose "
        "dynamic (GH raises blood glucose, insulin lowers it).  "
        "GH fragments: HGH frag 176-191 (AOD-9604) for fat metabolism "
        "without IGF-1 elevation.  Dose-response curves: where does "
        "additional GH stop producing proportional benefit?  Diminishing "
        "returns and the ceiling effect.  Water retention management.  "
        "GH-induced insulin resistance at high doses — how this changes "
        "the insulin protocol requirements."
    ),
    enrichment_queries=[
        "growth hormone extreme high dose 100+ IU bodybuilding effects",
        "GH gut palumboism visceral organ growth mechanism IGF-1",
        "IGF-1 LR3 vs IGF-1 bodybuilding differences long acting",
        "growth hormone insulin resistance high dose mechanism",
        "growth hormone acromegaly features bodybuilding cardiomegaly",
        "GH dose response curve diminishing returns bodybuilding",
        "growth hormone timing relative to insulin fasting glucose",
        "HGH fragment 176-191 AOD-9604 fat metabolism without IGF-1",
        "growth hormone receptor downregulation continuous vs pulsatile",
        "GH water retention management bodybuilding protocol",
        "growth hormone carpal tunnel neuropathy high dose side effects",
        "Andrei Deiu GH dose bodybuilding protocol extreme",
    ],
    key_compounds=["growth hormone", "IGF-1", "IGF-1 LR3", "AOD-9604"],
    key_interactions=[
        "GH → hepatic IGF-1 production (dose-dependent ceiling)",
        "GH → insulin resistance (glucose competition with insulin)",
        "GH + insulin → IGF-1 amplification but glucose tug-of-war",
        "GH → visceral organ growth at extreme doses (GH gut)",
        "GH → cardiomegaly risk (direct myocardial IGF-1 receptors)",
        "GH → connective tissue thickening (collagen synthesis)",
        "GH → water retention via sodium reabsorption",
    ],
)

TESTOSTERONE_TREN_ANGLE = AngleDefinition(
    label="Testosterone & trenbolone pharmacokinetics",
    description=(
        "Testosterone as anabolic foundation — ester pharmacokinetics "
        "(cypionate, enanthate, propionate, suspension) with exact "
        "half-lives and peak timing.  Dose-response for hypertrophy.  "
        "Aromatization to estradiol — rate depends on body fat, dose, "
        "and individual aromatase activity.  AI dosing (anastrozole, "
        "letrozole) and why crashing estrogen is dangerous.  "
        "Trenbolone as potency multiplier — 19-nor pharmacology, "
        "5x AR binding affinity vs testosterone, non-aromatizable "
        "(no estradiol conversion → removes estrogenic MMP suppression "
        "in connective tissue).  Nutrient partitioning mechanism: "
        "tren upregulates IGF-1 locally in muscle, reduces cortisol "
        "receptor sensitivity, enhances nitrogen retention.  How tren "
        "makes insulin more effective per IU via enhanced GLUT4 "
        "translocation and reduced lipogenesis.  Progestogenic effects "
        "— prolactin elevation affecting GH release, management with "
        "cabergoline.  Tren acetate vs enanthate — acetate for control "
        "(short half-life, quick clearance if sides hit), enanthate "
        "for stable levels.  Cardiovascular impact: hematocrit elevation, "
        "HDL destruction, LV hypertrophy, arterial stiffness."
    ),
    enrichment_queries=[
        "testosterone ester pharmacokinetics half-life cypionate enanthate",
        "trenbolone nutrient partitioning mechanism IGF-1 upregulation",
        "trenbolone insulin sensitivity GLUT4 translocation interaction",
        "testosterone aromatization rate estrogen management AI dosing",
        "trenbolone acetate vs enanthate half-life pharmacokinetics",
        "trenbolone prolactin elevation GH suppression cabergoline",
        "trenbolone cardiovascular hematocrit HDL LV hypertrophy",
        "testosterone dose response curve hypertrophy anabolic threshold",
        "trenbolone 19-nor pharmacology androgen receptor binding affinity",
        "trenbolone non-aromatizable estrogen MMP connective tissue",
        "testosterone base dose with trenbolone optimal ratio",
        "trenbolone nephrotoxicity kidney function GFR impact",
    ],
    key_compounds=["testosterone", "trenbolone", "anastrozole", "cabergoline"],
    key_interactions=[
        "tren → enhanced nutrient partitioning (amplifies insulin per IU)",
        "tren → prolactin elevation (suppresses endogenous GH pulsatility)",
        "tren → hematocrit increase (compounds with boldenone EPO effect)",
        "testosterone → estradiol via aromatase (zinc modulates activity)",
        "tren non-aromatizable → no estradiol → removes MMP suppression",
        "tren → cortisol receptor desensitization (anti-catabolic)",
        "tren + insulin → synergistic GLUT4 + nutrient partitioning",
    ],
)

ORAL_COMPOUNDS_ANGLE = AngleDefinition(
    label="Oral compounds — turinabol, LGD-4033, actovegin",
    description=(
        "Turinabol as dry oral kickstart — 17-alpha-alkylated hepatotoxicity "
        "vs anabolic benefit, no water retention (preserves insulin/carb "
        "balance).  Exact dosing: 40-60mg/day, 4-6 week duration, split "
        "dosing for stable levels.  Liver methylation pathway load — "
        "requires B vitamin cofactors (B6, B12, folate, TMG) for "
        "phase II conjugation.  LGD-4033 as selective androgen receptor "
        "modulator — minimal hepatic load, additive to testosterone base, "
        "dose 5-10mg/day.  Actovegin for enhanced cellular oxygen "
        "utilization — deproteinized hemoderivative mechanism, synergy "
        "with boldenone on oxygen transport."
    ),
    enrichment_queries=[
        "turinabol bodybuilding dosing 40-60mg cycle length liver",
        "LGD-4033 ligandrol 5-10mg bodybuilding stack testosterone",
        "actovegin bodybuilding recovery oxygen cellular utilization",
        "turinabol hepatotoxicity methylation pathway B vitamins",
        "LGD-4033 SARM selective androgen receptor minimal liver",
        "actovegin mechanism deproteinized hemoderivative",
        "oral steroid liver protection TUDCA NAC dosing protocol",
        "turinabol half-life split dosing timing around training",
    ],
    key_compounds=["turinabol", "LGD-4033", "actovegin"],
    key_interactions=[
        "turinabol → liver methylation load (B6/B12/folate/TMG dependent)",
        "LGD-4033 → additive AR activation (minimal liver stress)",
        "actovegin → oxygen utilization (synergy with boldenone EPO)",
        "turinabol → no aromatization (preserves insulin/carb balance)",
    ],
)

BOLDENONE_ANGLE = AngleDefinition(
    label="Boldenone & EQ interactions",
    description=(
        "Boldenone undecylenate (Equipoise) — EPO-like erythropoiesis, "
        "appetite stimulation (critical for caloric demands of insulin "
        "protocols), long ester pharmacokinetics (14-day half-life), "
        "AI-like metabolite (1-testosterone/ATD reduces estrogen).  "
        "Interaction with trenbolone on hematocrit — both compounds "
        "increase RBC production through different mechanisms (boldenone "
        "via EPO stimulation, tren via direct marrow stimulation).  "
        "Interaction with actovegin on oxygen transport.  Anxiety and "
        "neurological side effects at higher doses.  Front-loading "
        "strategy due to long ester."
    ),
    enrichment_queries=[
        "boldenone equipoise bodybuilding dosing 400-800mg cycle",
        "boldenone EPO erythropoiesis mechanism stimulation",
        "boldenone appetite stimulation caloric surplus insulin protocol",
        "boldenone hematocrit management phlebotomy threshold",
        "boldenone AI metabolite 1-testosterone ATD estrogen",
        "boldenone trenbolone stack compounded hematocrit management",
        "equipoise long ester front-load pharmacokinetics 14 day",
        "boldenone anxiety neurological side effects higher doses",
    ],
    key_compounds=["boldenone", "boldenone undecylenate"],
    key_interactions=[
        "boldenone EPO + tren erythropoiesis → compounded hematocrit",
        "boldenone appetite + insulin protocol → caloric surplus support",
        "boldenone + actovegin → oxygen transport optimization",
        "boldenone ATD metabolite → AI-like estrogen reduction",
    ],
)

PED_SYNERGIES_ANGLE = AngleDefinition(
    label="PED stack synergies at biochemistry level",
    description=(
        "Every compound-compound interaction mapped at the receptor, "
        "pathway, and enzyme level.  Not just 'tren + insulin is good' "
        "but WHY at the molecular level.  Synergistic interactions: "
        "tren + insulin (GLUT4 upregulation × nutrient partitioning), "
        "GH + insulin (IGF-1 amplification via hepatic + local production), "
        "testosterone + GH (AR density × IGF-1 receptor sensitivity).  "
        "Antagonistic interactions: tren prolactin → suppresses GH "
        "pulsatility (offset by exogenous GH), high-dose GH → insulin "
        "resistance (requires higher insulin doses).  Conditional "
        "interactions: boldenone + tren hematocrit compounding (monitor, "
        "manage, but both serve important functions).  Real-world "
        "practitioner protocol adjustments — how coaches titrate "
        "compounds based on bloodwork feedback.  Protocol design as "
        "a system: which compounds enter when and why, transition "
        "criteria between phases, exit strategies and PCT."
    ),
    enrichment_queries=[
        "trenbolone insulin synergy GLUT4 nutrient partitioning mechanism",
        "GH insulin IGF-1 amplification hepatic local production",
        "testosterone GH synergy androgen receptor IGF-1 receptor",
        "trenbolone prolactin GH pulsatility suppression mechanism",
        "high dose GH insulin resistance mechanism compensation",
        "boldenone trenbolone hematocrit compounding management",
        "PED compound interaction matrix receptor pathway level",
        "bodybuilding coach protocol titration bloodwork adjustment",
        "Milos Sarcev client protocol design system phase transition",
        "advanced bodybuilding stack 16 week cycle real world dosing",
        "pro bodybuilder offseason cycle insulin GH protocol experience",
        "PCT protocol after complex multi-compound cycle",
    ],
    key_compounds=["all"],
    key_interactions=[
        "tren + insulin → synergistic nutrient partitioning (GLUT4 × IGF-1)",
        "GH + insulin → IGF-1 amplification but glucose competition",
        "tren prolactin → suppresses endogenous GH (offset by exogenous)",
        "GH resistance at extreme doses → insulin dose escalation needed",
        "boldenone + tren → compounded hematocrit (manage, don't avoid)",
        "testosterone → estradiol → bone density + joint lubrication",
        "phase design: compound introduction order gates on bloodwork",
    ],
)


# ── Layer 2: Nutrient biochemistry angles ────────────────────────────

FATTY_ACID_ANGLE = AngleDefinition(
    label="Specific fatty acid biochemistry & PED interactions",
    description=(
        "NOT 'omega-3' as a category — individual fatty acids with distinct "
        "biochemical roles in PED-augmented physiology.  "
        "DHA (docosahexaenoic acid): neuroprotection under tren "
        "neurotoxicity, cell membrane fluidity, anti-inflammatory via "
        "resolvins and protectins.  "
        "EPA (eicosapentaenoic acid): anti-inflammatory prostaglandin "
        "pathway (PGE3), HDL support against tren lipid destruction, "
        "triglyceride reduction.  "
        "ALA (alpha-linolenic acid): conversion rates to EPA/DHA are "
        "poor (5-15%) — supplementing ALA alone is insufficient.  "
        "Arachidonic acid (AA): prostaglandin PGF2-alpha and PGE2 "
        "production — the muscle growth signal.  AA is a DIRECT driver "
        "of exercise-induced muscle protein synthesis via COX-2 pathway.  "
        "Synergy with insulin-mediated nutrient delivery.  "
        "CLA (conjugated linoleic acid): body composition effects, "
        "PPAR-gamma modulation, controversial efficacy.  "
        "GLA (gamma-linolenic acid): anti-inflammatory cascade via "
        "DGLA → PGE1, counterbalances AA-derived inflammation.  "
        "MCTs (medium-chain triglycerides): rapid oxidation, ketone "
        "body production (beta-hydroxybutyrate), GH interaction — "
        "ketones may potentiate GH release.  "
        "Palmitoleic acid (C16:1n-7): lipokine signaling, insulin "
        "sensitivity enhancement, emerging research."
    ),
    enrichment_queries=[
        "DHA docosahexaenoic acid neuroprotection neurotoxicity steroid",
        "EPA eicosapentaenoic acid anti-inflammatory prostaglandin PGE3",
        "EPA DHA HDL support trenbolone lipid destruction fish oil dose",
        "arachidonic acid prostaglandin PGF2-alpha muscle growth COX-2",
        "arachidonic acid muscle protein synthesis exercise bodybuilding",
        "CLA conjugated linoleic acid body composition PPAR-gamma",
        "GLA gamma-linolenic acid DGLA PGE1 anti-inflammatory",
        "MCT medium chain triglycerides ketone GH release interaction",
        "palmitoleic acid lipokine insulin sensitivity signaling",
        "omega-3 specific DHA EPA ratio bodybuilding dosing grams",
        "arachidonic acid supplement bodybuilding insulin synergy",
        "fatty acid cell membrane fluidity androgen receptor function",
    ],
    key_compounds=[
        "DHA", "EPA", "ALA", "arachidonic acid", "CLA",
        "GLA", "MCTs", "palmitoleic acid",
    ],
    key_interactions=[
        "DHA → neuroprotection (counters tren neurotoxicity)",
        "EPA → PGE3 anti-inflammatory (counters tren HDL destruction)",
        "AA → PGF2-alpha → muscle growth signal (synergy with insulin)",
        "MCTs → ketone bodies → may potentiate GH release",
        "palmitoleic acid → insulin sensitivity signaling lipokine",
        "EPA/DHA ratio matters: 2:1 EPA:DHA for inflammation focus",
        "GLA → DGLA → PGE1 (counterbalances AA-driven inflammation)",
    ],
)

AMINO_ACID_ANGLE = AngleDefinition(
    label="Amino acid biochemistry & timing with PED protocols",
    description=(
        "Individual amino acids with specific biochemical roles — not just "
        "'protein' but the distinct signaling molecules.  "
        "Leucine: direct mTOR activation via leucyl-tRNA synthetase → "
        "mTORC1 pathway.  This is the PRIMARY insulin synergy — insulin "
        "activates mTOR via PI3K/Akt, leucine activates via Rag GTPases.  "
        "Dual activation = maximum muscle protein synthesis.  Dose: 3-5g "
        "in the insulin window.  "
        "Glutamine: gut integrity under tren stress (tren damages gut "
        "epithelium), immune function support, conditionally essential "
        "under extreme training.  Dose: 10-20g/day split.  "
        "Taurine: GH-induced cramps (taurine depletion mechanism), bile "
        "acid conjugation (liver support under oral steroid load), "
        "insulin signaling modulation, osmoregulation.  Dose: 3-5g.  "
        "Glycine: collagen synthesis (connective tissue under heavy "
        "loading), sleep quality → GH release optimization, detox "
        "conjugation pathway.  Dose: 3-5g before bed.  "
        "NAC (N-acetyl cysteine): glutathione precursor → liver "
        "protection under oral steroid methylation load.  Rate-limiting "
        "step in glutathione synthesis.  Dose: 600-1200mg/day.  "
        "Arginine/Citrulline: NO pathway → vasodilation → blood flow → "
        "nutrient delivery in insulin windows.  Citrulline is superior "
        "(better bioavailability, converts to arginine in kidneys).  "
        "EAA ratios for insulin-window protein synthesis — the 2:1:1 "
        "leucine ratio and why it matters for mTOR signaling."
    ),
    enrichment_queries=[
        "leucine mTOR activation mTORC1 insulin synergy PI3K Akt pathway",
        "leucine dose muscle protein synthesis insulin window bodybuilding",
        "glutamine gut integrity trenbolone gut epithelium damage",
        "taurine GH cramps depletion mechanism growth hormone",
        "taurine bile acid conjugation liver support oral steroid",
        "taurine insulin signaling modulation pancreatic beta cells",
        "glycine collagen synthesis connective tissue heavy loading",
        "glycine sleep quality growth hormone release optimization",
        "NAC glutathione precursor liver protection oral steroid",
        "citrulline arginine NO vasodilation nutrient delivery insulin",
        "EAA essential amino acid ratio leucine mTOR bodybuilding",
        "amino acid timing relative to insulin injection window",
    ],
    key_compounds=[
        "leucine", "glutamine", "taurine", "glycine", "NAC",
        "citrulline", "arginine", "EAA blend",
    ],
    key_interactions=[
        "leucine + insulin → dual mTOR activation (Rag + PI3K/Akt)",
        "glutamine → gut integrity (counters tren gut damage)",
        "taurine → counters GH-induced cramps + bile acid support",
        "glycine → collagen synthesis + sleep→GH optimization",
        "NAC → glutathione → liver detox (rate-limiting precursor)",
        "citrulline → NO → blood flow → insulin window nutrient delivery",
        "EAA 2:1:1 leucine ratio → mTOR signaling optimization",
    ],
)

CARBOHYDRATE_ANGLE = AngleDefinition(
    label="Carbohydrate biochemistry & insulin windows",
    description=(
        "The molecular-level carbohydrate science behind Milos's insulin "
        "protocol.  Different carb types have radically different glycemic "
        "profiles and this MATTERS for insulin timing.  "
        "Dextrose (glucose): fastest glycemic response, direct muscle "
        "glycogen via GLUT4 (insulin-dependent), the classic Milos choice "
        "for post-injection carbs.  "
        "Highly Branched Cyclic Dextrin (HBCD/Cluster Dextrin): low "
        "osmolality, rapid gastric emptying, sustained glucose release — "
        "ideal for INTRA-workout with insulin (no osmotic gut distress).  "
        "Waxy maize starch: high molecular weight, moderate glycemic index, "
        "fast gastric emptying.  "
        "Maltodextrin: fast-absorbing, spikes insulin hard — useful for "
        "post-workout but can cause GI distress intra-workout.  "
        "Fructose: liver glycogen ONLY (not muscle) — GLUT5 transporter, "
        "independent of insulin.  Small amounts useful for liver glycogen "
        "replenishment but excess → lipogenesis.  Critical distinction.  "
        "Glycogen supercompensation: the window after depletion training "
        "where GLUT4 translocation is maximized — this is where insulin "
        "has its greatest anabolic effect.  "
        "Carb-per-IU ratios: Milos's rule (8-10g fast carbs per IU rapid "
        "insulin), adjustments based on insulin sensitivity, body weight, "
        "and training intensity.  Timing: carbs BEFORE insulin peaks, "
        "not after."
    ),
    enrichment_queries=[
        "dextrose glucose glycemic index muscle glycogen GLUT4 insulin",
        "HBCD highly branched cyclic dextrin cluster dextrin intra-workout",
        "waxy maize starch molecular weight glycemic bodybuilding",
        "maltodextrin glycemic spike insulin post-workout bodybuilding",
        "fructose liver glycogen GLUT5 not muscle lipogenesis distinction",
        "glycogen supercompensation GLUT4 translocation insulin window",
        "carbs per IU insulin ratio Milos Sarcev 8-10 grams protocol",
        "intra-workout carbohydrate insulin protocol HBCD vs dextrose",
        "carbohydrate timing relative to insulin injection peak curve",
        "glycemic index vs glycemic load bodybuilding insulin protocol",
        "post-workout carb loading glycogen synthesis rate insulin",
        "insulin resistance carbohydrate type selection management",
    ],
    key_compounds=[
        "dextrose", "HBCD", "waxy maize", "maltodextrin",
        "fructose", "amylopectin", "palatinose",
    ],
    key_interactions=[
        "dextrose → GLUT4 (insulin-dependent) → muscle glycogen directly",
        "fructose → GLUT5 (insulin-independent) → liver glycogen ONLY",
        "HBCD → low osmolality → no gut distress with intra-workout insulin",
        "glycogen depletion → GLUT4 upregulation → insulin sensitivity spike",
        "carb timing BEFORE insulin peak → prevents hypoglycemia",
        "8-10g fast carbs per IU rapid insulin (Milos baseline ratio)",
        "excess fructose → hepatic lipogenesis (defeats insulin protocol)",
    ],
)

MICRONUTRIENT_ANGLE = AngleDefinition(
    label="Micronutrients, minerals & vitamins — individual compounds",
    description=(
        "Individual micronutrients, not categories.  Forms matter — "
        "magnesium glycinate absorbs differently from oxide.  "
        "Zinc (30-50mg elemental as picolinate or glycinate): aromatase "
        "modulation — zinc status directly affects testosterone→estrogen "
        "conversion rate.  Also: immune function, insulin signaling.  "
        "Magnesium (400-600mg as glycinate or citrate, NOT oxide): "
        "insulin receptor function — Mg depletion directly impairs "
        "insulin sensitivity.  Also: muscle relaxation, sleep quality, "
        "hundreds of enzymatic reactions.  "
        "Vitamin D3 (5000-10000 IU/day): androgen receptor density — "
        "D3 status correlates with free testosterone.  Also: immune "
        "modulation, bone density, insulin sensitivity.  "
        "Vitamin K2-MK7 (100-200mcg): works with D3 for calcium "
        "direction — into bones, not arteries.  Critical on high-dose "
        "D3.  Arterial calcification prevention.  "
        "Individual B vitamins: B6 (pyridoxal-5-phosphate, prolactin "
        "management), B12 (methylcobalamin, neurological function under "
        "tren), folate (5-MTHF, methylation cycle), niacin/B3 (lipid "
        "profile, flush form for HDL), pantothenic acid/B5 (adrenal "
        "support, CoA synthesis), biotin/B7 (glucose metabolism).  "
        "Chromium (200-400mcg as picolinate): insulin sensitivity "
        "enhancement, GLUT4 potentiation.  "
        "Vanadium (vanadyl sulfate 10-20mg): insulin-mimetic effects, "
        "controversial but used in bodybuilding.  "
        "Selenium (200mcg as selenomethionine): thyroid function → "
        "T4→T3 conversion → metabolic rate → insulin sensitivity.  "
        "Also: glutathione peroxidase cofactor.  "
        "Boron (6-10mg): free testosterone increase by reducing SHBG, "
        "estrogen metabolism, bone density.  "
        "Iron: monitor ferritin under tren+boldenone (erythropoiesis "
        "consumes iron stores).  Do NOT supplement blindly — check "
        "ferritin first.  Excess iron is toxic.  "
        "Potassium: insulin shifts K+ intracellularly — monitor and "
        "replenish, especially on insulin days.  Cardiac risk."
    ),
    enrichment_queries=[
        "zinc aromatase modulation testosterone estrogen conversion rate",
        "magnesium glycinate insulin receptor sensitivity mechanism",
        "vitamin D3 androgen receptor density free testosterone",
        "vitamin K2 MK7 calcium direction arterial calcification D3",
        "vitamin B6 pyridoxal-5-phosphate prolactin management",
        "vitamin B12 methylcobalamin neurological trenbolone",
        "niacin B3 flush form HDL lipid profile trenbolone",
        "chromium picolinate insulin sensitivity GLUT4 potentiation",
        "vanadyl sulfate insulin mimetic bodybuilding mechanism",
        "selenium selenomethionine thyroid T4 T3 conversion metabolism",
        "boron free testosterone SHBG estrogen metabolism dosing",
        "iron ferritin depletion trenbolone boldenone erythropoiesis",
        "potassium intracellular shift insulin cardiac risk monitoring",
        "magnesium forms glycinate citrate oxide absorption comparison",
    ],
    key_compounds=[
        "zinc picolinate", "magnesium glycinate", "vitamin D3",
        "vitamin K2-MK7", "P5P (B6)", "methylcobalamin (B12)",
        "5-MTHF (folate)", "niacin", "chromium picolinate",
        "vanadyl sulfate", "selenomethionine", "boron", "iron",
        "potassium",
    ],
    key_interactions=[
        "Zn status → aromatase activity (test→estrogen conversion)",
        "Mg depletion → impaired insulin receptor function directly",
        "D3 → androgen receptor binding efficiency",
        "K2-MK7 + D3 → calcium into bones not arteries (critical)",
        "B6 P5P → prolactin reduction (adjunct to cabergoline)",
        "B12/folate → methylation capacity (turinabol detox support)",
        "niacin flush → HDL support against tren lipid destruction",
        "chromium → GLUT4 potentiation (amplifies insulin effect)",
        "selenium → T4→T3 conversion → metabolic rate",
        "iron: CHECK FERRITIN FIRST, tren+EQ deplete stores",
        "potassium: insulin shifts K+ intracellularly → cardiac risk",
    ],
)


# ── Layer 3: Cross-cutting angles ────────────────────────────────────

SULPHUR_METHYLATION_ANGLE = AngleDefinition(
    label="Sulphur biochemistry, methylation & detox pathways",
    description=(
        "The serendipity angle — sulphur-containing compounds and "
        "methylation pathways that connect to every part of the stack "
        "in non-obvious ways.  "
        "Glutathione synthesis pathway: cysteine (rate-limiting, via "
        "NAC) + glutamate + glycine → glutathione.  The body's master "
        "antioxidant and phase II conjugation engine.  Under oral "
        "steroid load, glutathione demand skyrockets.  "
        "Methylation cycle: SAMe (S-adenosyl methionine) → homocysteine "
        "→ methionine.  Requires B12, folate, TMG/betaine as cofactors.  "
        "Turinabol's 17-alpha-alkylation is cleared via methylation — "
        "a loaded methylation cycle means slower tbol clearance and "
        "higher liver stress.  TMG (trimethylglycine/betaine) donates "
        "methyl groups directly.  "
        "Sulphation pathways: PAPS-dependent sulphotransferases "
        "conjugate steroid hormones for excretion.  Sulphate "
        "availability (from MSM, taurine, cysteine) directly affects "
        "how fast the body clears tren/tbol metabolites.  "
        "MSM (methylsulfonylmethane): organic sulphur donor, joint "
        "support, anti-inflammatory, supports glutathione production.  "
        "Taurine: sulphur-containing amino acid — bile acid conjugation "
        "(taurocholate), osmoregulation, insulin signaling, antioxidant.  "
        "Garlic/allicin: organosulphur compounds — allicin → ajoene, "
        "lipid-lowering, antiplatelet (relevant under hematocrit "
        "elevation from tren/EQ).  DIM (diindolylmethane from "
        "cruciferous): estrogen metabolism — shifts estrogen toward "
        "2-hydroxy metabolites (less estrogenic) vs 16-hydroxy.  "
        "Connection to periodontal health: sulphation pathways clear "
        "both steroid metabolites AND bacterial endotoxins — shared "
        "pathway competition under PED load."
    ),
    enrichment_queries=[
        "glutathione synthesis cysteine NAC rate-limiting oral steroid",
        "methylation cycle SAMe homocysteine B12 folate TMG betaine",
        "turinabol 17-alpha-alkylation methylation clearance liver",
        "TMG trimethylglycine betaine methyl donor bodybuilding liver",
        "sulphation pathway PAPS steroid hormone excretion conjugation",
        "MSM methylsulfonylmethane sulphur donor joint anti-inflammatory",
        "taurine bile acid conjugation taurocholate liver support",
        "garlic allicin organosulphur lipid-lowering antiplatelet",
        "DIM diindolylmethane estrogen metabolism 2-hydroxy 16-hydroxy",
        "sulphur compounds glutathione production support pathway",
        "sulphation steroid metabolite clearance trenbolone excretion",
        "homocysteine elevation steroid cycle cardiovascular risk TMG",
    ],
    key_compounds=[
        "NAC", "glutathione", "SAMe", "TMG/betaine", "MSM",
        "taurine", "allicin", "DIM", "methionine", "cysteine",
    ],
    key_interactions=[
        "NAC → cysteine → glutathione (rate-limiting for liver detox)",
        "TMG → methyl groups → methylation cycle support (tbol clearance)",
        "sulphation pathway → steroid AND endotoxin clearance (shared!)",
        "MSM → organic sulphur → joint support + glutathione precursor",
        "taurine → taurocholate → bile acid → fat absorption + liver",
        "garlic allicin → antiplatelet (critical under high hematocrit)",
        "DIM → 2-OH estrogen shift (favorable estrogen metabolism)",
        "homocysteine → cardiovascular risk marker under PED load",
        "glutathione demand ↑↑ under oral steroid methylation load",
    ],
)

PEPTIDE_PHARMACOLOGY_ANGLE = AngleDefinition(
    label="Peptide pharmacology & PED stack interactions",
    description=(
        "Peptides as adjuncts to the main PED stack — each with specific "
        "pharmacology and interaction profiles.  "
        "BPC-157 (Body Protection Compound): gastric pentadecapeptide, "
        "gut healing (relevant under tren gut damage), tendon repair "
        "via VEGF upregulation, anti-inflammatory, accelerates wound "
        "healing.  May upregulate GH receptor sensitivity.  Dose: "
        "250-500mcg 2x/day, subcutaneous or oral.  "
        "TB-500 (Thymosin Beta-4): systemic tissue repair, "
        "angiogenesis, anti-fibrotic, cardiac tissue protection "
        "(relevant under cardiomegaly risk from GH).  Dose: 2-5mg "
        "2x/week.  "
        "CJC-1295 (with DAC): GHRH analog, sustained GH release over "
        "days.  On 120 IU exogenous GH: is this synergistic or "
        "redundant?  May help with endogenous GH recovery post-cycle.  "
        "Ipamorelin: GHRP, GH secretagogue that doesn't raise cortisol "
        "or prolactin (unlike GHRP-6, GHRP-2).  Cleaner GH pulse.  "
        "GHRP-6: potent GH release + significant appetite stimulation "
        "(ghrelin mimetic) — synergy with boldenone for caloric surplus.  "
        "Also raises cortisol and prolactin (contraindicated with tren "
        "prolactin issues).  "
        "AOD-9604 (GH fragment 176-191): fat metabolism without IGF-1 "
        "elevation.  Relevant for body composition without adding to "
        "the IGF-1 load from extreme GH dosing.  "
        "Epithalon (Epitalon): telomerase activation, telomere "
        "maintenance — speculative longevity peptide, potentially "
        "relevant for mitigating cellular damage from extreme PED use.  "
        "PT-141 (Bremelanotide): sexual function — melanocortin "
        "receptor agonist, relevant for managing tren/19-nor sexual "
        "side effects."
    ),
    enrichment_queries=[
        "BPC-157 gut healing tendon repair VEGF bodybuilding dosing",
        "BPC-157 GH receptor sensitivity upregulation mechanism",
        "TB-500 thymosin beta-4 tissue repair angiogenesis cardiac",
        "CJC-1295 DAC GHRH analog GH release sustained mechanism",
        "ipamorelin GH secretagogue no cortisol no prolactin clean",
        "GHRP-6 appetite stimulation ghrelin mimetic GH prolactin",
        "AOD-9604 GH fragment 176-191 fat metabolism without IGF-1",
        "epithalon epitalon telomerase telomere longevity peptide",
        "PT-141 bremelanotide melanocortin sexual function 19-nor",
        "peptide stack bodybuilding BPC-157 TB-500 combination",
        "CJC-1295 ipamorelin stack with exogenous GH synergy redundant",
        "GHRP-6 cortisol prolactin elevation trenbolone contraindication",
    ],
    key_compounds=[
        "BPC-157", "TB-500", "CJC-1295", "ipamorelin",
        "GHRP-6", "AOD-9604", "epithalon", "PT-141",
    ],
    key_interactions=[
        "BPC-157 → gut healing (counters tren gut damage directly)",
        "BPC-157 → VEGF → tendon repair (under heavy loading)",
        "TB-500 → cardiac tissue protection (counters GH cardiomegaly)",
        "CJC-1295 + exogenous GH → redundant or synergistic? (context)",
        "ipamorelin → clean GH pulse (no prolactin, unlike GHRP-6)",
        "GHRP-6 → appetite + GH but also prolactin (tren conflict)",
        "AOD-9604 → fat loss without IGF-1 load (GH gut mitigation)",
        "epithalon → telomere maintenance (speculative, long-term)",
    ],
)


# ── Aggregated configuration ─────────────────────────────────────────

# All angle definitions in swarm worker order.
# Layer 1: Compound-specific (6 angles)
# Layer 2: Nutrient biochemistry (4 angles)
# Layer 3: Cross-cutting (2 angles)
ALL_ANGLES: list[AngleDefinition] = [
    # Layer 1 — Compound-specific
    INSULIN_PROTOCOL_ANGLE,
    GH_IGF1_EXTREME_ANGLE,
    TESTOSTERONE_TREN_ANGLE,
    ORAL_COMPOUNDS_ANGLE,
    BOLDENONE_ANGLE,
    PED_SYNERGIES_ANGLE,
    # Layer 2 — Nutrient biochemistry
    FATTY_ACID_ANGLE,
    AMINO_ACID_ANGLE,
    CARBOHYDRATE_ANGLE,
    MICRONUTRIENT_ANGLE,
    # Layer 3 — Cross-cutting
    SULPHUR_METHYLATION_ANGLE,
    PEPTIDE_PHARMACOLOGY_ANGLE,
]

# Labels only — for SwarmConfig.required_angles
REQUIRED_ANGLE_LABELS: list[str] = [a.label for a in ALL_ANGLES]

# All enrichment queries flattened for the enrichment pipeline
ALL_ENRICHMENT_QUERIES: list[str] = []
for _angle in ALL_ANGLES:
    ALL_ENRICHMENT_QUERIES.extend(_angle.enrichment_queries)


# ---------------------------------------------------------------------------
# Confidence tier definitions for queen synthesis output
# ---------------------------------------------------------------------------

CONFIDENCE_TIERS = """
CONFIDENCE TIER DEFINITIONS — use these throughout the report:

  ESTABLISHED  — Multiple independent sources, consistent pharmacological
                 mechanism, widely reproduced in clinical or practitioner
                 settings.  Act on this with normal caution.

  PROBABLE     — Good mechanistic basis, limited but consistent clinical
                 evidence, corroborated by practitioner reports.  Reasonable
                 to act on but verify with bloodwork.

  SPECULATIVE  — Single source, theoretical mechanism only, or serendipity
                 finding.  Do NOT act on this without independent
                 verification from a medical professional.

  UNKNOWN      — Insufficient evidence to make any recommendation.
                 Explicitly flagged gap.  Honestly stating "we don't know"
                 is infinitely more valuable than a plausible-sounding
                 fabrication.  There are lives at stake.

For dosing recommendations specifically:
  - If a dose comes from 2+ independent sources → state the dose
  - If a dose comes from a single source → mark as SINGLE-SOURCE ESTIMATE
  - If no specific dose data exists → state UNKNOWN, do not extrapolate
"""


def get_swarm_query() -> str:
    """Return the master swarm query for the insulin/GH/tren extraction.

    This query encodes both the research scope AND the anti-hallucination
    constraint at the query level.  The queen synthesis must produce
    confidence-tiered output rather than presenting all claims at equal
    weight.
    """
    return (
        "Synthesize a comprehensive, medical-manual-grade protocol for a "
        "bodybuilding cycle covering insulin (grounded in Milos Sarcev's "
        "timing framework), growth hormone (including extreme dosing context "
        "— 120 IU), trenbolone, testosterone, turinabol, boldenone, "
        "actovegin, LGD-4033, and supporting peptides.\n\n"

        "CRITICAL CONSTRAINT — ANTI-HALLUCINATION:\n"
        "There are lives at stake.  Every recommendation in this report "
        "will be used by real people making real decisions about substances "
        "that can kill.  Therefore:\n"
        "  - If evidence is insufficient or contradictory, write "
        "'EVIDENCE INSUFFICIENT' rather than extrapolating.\n"
        "  - An honest gap is infinitely more valuable than a "
        "plausible-sounding fabrication.\n"
        "  - For every dosing recommendation, cite whether it comes from "
        "multiple independent sources, a single source, or extrapolation.\n"
        "  - If you don't know, say you don't know.  Do not fill gaps "
        "with plausible-sounding text.\n"
        "  - Mark every claim with a confidence tier: ESTABLISHED, "
        "PROBABLE, SPECULATIVE, or UNKNOWN.\n\n"

        "REQUIRED SECTIONS:\n\n"

        "1. INSULIN PROTOCOL (Milos Sarcev Framework):\n"
        "   - Medical-manual dosing tables: conservative, moderate, "
        "aggressive phases\n"
        "   - Exact carb-per-IU ratios by insulin type (lispro, aspart, "
        "regular)\n"
        "   - Glucometer monitoring protocol with threshold values\n"
        "   - Hypoglycemia emergency protocol with exact glucose "
        "thresholds and carb quantities\n"
        "   - Timing windows relative to training, GH injection, meals\n\n"

        "2. GROWTH HORMONE & IGF-1 AXIS:\n"
        "   - Dose-response analysis including extreme dosing (120 IU)\n"
        "   - Organ strain assessment: GH gut, cardiomegaly, acromegalic "
        "features\n"
        "   - GH-induced insulin resistance and protocol compensation\n"
        "   - IGF-1 cascade: hepatic vs local, IGF-1 LR3, fragments\n\n"

        "3. TRENBOLONE DYNAMICS:\n"
        "   - Nutrient partitioning mechanism at receptor level\n"
        "   - Insulin sensitivity interaction (how tren amplifies "
        "insulin per IU)\n"
        "   - Prolactin → GH suppression cascade and management\n"
        "   - Cardiovascular impact: hematocrit, lipids, LV hypertrophy\n\n"

        "4. NUTRIENT RECOMMENDATIONS — BIOCHEMISTRY LEVEL:\n"
        "   a) Specific fatty acids (NOT just 'omega-3'): DHA, EPA, "
        "arachidonic acid, CLA, GLA, MCTs, palmitoleic acid — each with "
        "mechanism, dose, timing, and PED interaction\n"
        "   b) Individual amino acids: leucine (mTOR/insulin synergy), "
        "glutamine (gut integrity), taurine (GH cramps + bile + insulin "
        "signaling), glycine (collagen + sleep/GH), NAC (glutathione), "
        "citrulline (NO/blood flow) — each with mechanism, dose, timing\n"
        "   c) Carbohydrate types: dextrose, HBCD, waxy maize, "
        "maltodextrin, fructose — glycemic profiles and WHY each matters "
        "for insulin window design, carb-per-IU protocols\n"
        "   d) Individual vitamins and minerals with FORMS: zinc "
        "(picolinate, aromatase), magnesium (glycinate, insulin receptor), "
        "D3+K2-MK7, individual B vitamins (P5P, methylcobalamin, "
        "5-MTHF, niacin), chromium, selenium, boron, iron (CHECK "
        "FERRITIN), potassium (cardiac risk with insulin)\n\n"

        "5. PED SYNERGIES AT BIOCHEMISTRY LEVEL:\n"
        "   - Every compound×compound interaction at receptor/pathway "
        "level\n"
        "   - Synergistic, antagonistic, and conditional interactions\n"
        "   - Protocol design as a system: phase transitions, bloodwork "
        "gates\n\n"

        "6. PEPTIDES:\n"
        "   - BPC-157, TB-500, CJC-1295, ipamorelin, GHRP-6, AOD-9604, "
        "epithalon, PT-141\n"
        "   - Each: mechanism, dose, timing, interaction with main stack\n"
        "   - Specifically: CJC-1295 + 120 IU GH — synergistic or "
        "redundant?\n\n"

        "7. SULPHUR BIOCHEMISTRY & METHYLATION:\n"
        "   - Glutathione synthesis pathway under oral steroid load\n"
        "   - Methylation cycle: SAMe, TMG, B12, folate — tbol "
        "clearance\n"
        "   - Sulphation pathways: steroid metabolite excretion\n"
        "   - Cross-connections: allicin (antiplatelet under high "
        "hematocrit), DIM (estrogen metabolism)\n\n"

        "8. HEALTH MONITORING & SAFETY:\n"
        "   - Complete bloodwork panel with exact threshold values\n"
        "   - Emergency protocols for each major risk\n"
        "   - When to stop, when to reduce, when to seek medical help\n\n"

        "9. SERENDIPITY FINDINGS:\n"
        "   - Non-obvious cross-domain connections discovered during "
        "research\n"
        "   - Mark each with confidence tier\n"
        "   - These may be the most valuable part of the report\n\n"

        "OUTPUT FORMAT:\n"
        "For EACH recommendation, include:\n"
        "  - WHAT: Exact compound, form, dose, timing\n"
        "  - WHY: Biochemical mechanism (receptor, pathway, enzyme)\n"
        "  - CONFIDENCE: ESTABLISHED / PROBABLE / SPECULATIVE / UNKNOWN\n"
        "  - SOURCE BASIS: Multiple independent / single source / "
        "extrapolation\n"
        "  - INTERACTION: Which stack compounds this interacts with "
        "and how\n\n"

        "Ground every claim in source evidence.  Preserve exact numbers.  "
        "No moral disclaimers.  No academic hedging.  Practitioner "
        "language.  If a number is uncertain, give the range and say "
        "it's uncertain.  Never fabricate precision you don't have."
    )
