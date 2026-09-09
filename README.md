# AI Environmental Scientist

Diagnoses land degradation from soil, climate, and land-use inputs, then
recommends interventions with quantified effects, time horizons, confidence
levels, and citations to published evidence.

Built for the Darukaa.Earth hackathon challenge.

**Live app:** _[fill in]_
**Repository:** https://github.com/SidharthaIITKGP/darukaa_hackathon

---

## The core idea

The centrepiece is not retrieval. It is a curated, signed, weighted **causal
graph** of environmental variables, with a Monte Carlo engine that propagates
interventions through it. Retrieval sits underneath, attaching evidence to
edges the graph already holds.

That inversion is the whole design. A prompted model can produce fluent
recommendations; what it cannot do is compute a downstream effect through a
chain of biophysical relationships and report the uncertainty that
accumulates along the way.

**43 nodes, 48 edges.** Nodes are environmental state variables (soil organic
carbon, infiltration rate, plant-available water, microbial biomass,
structural heterogeneity, pollinator abundance, species richness, erosion
rate, crop yield) and interventions (cover crops, alley cropping, contour
bunding, reduced tillage, hedgerows, and others).

Each edge carries a sign, an effect-size **distribution** rather than a point
value, a time lag, the site conditions under which it holds, an evidence
reference with a role, and an evidence strength.

### Why this is not GraphRAG

Microsoft-style GraphRAG builds an entity graph *from* text in order to
*retrieve* text. Here the graph is a hand-curated domain ontology whose
purpose is to **compute**. Documents point at the graph, not the reverse.

The distinction matters practically: GraphRAG's nodes are whatever an LLM
extracted, so you get noisy entities like "farmer" and "the study". A
domain-typed causal graph is small enough to curate by hand and every
traversal is a scientific argument you can print.

---

## What the system does

### Multi-variable reasoning, computed rather than prompted

The brief calls multi-variable reasoning the core differentiator and sets a
floor of three. Propagating a legume cover crop to crop yield traverses four
distinct mechanisms across seven state variables:

```
legume_cover_crop -> soil_organic_carbon -> aggregate_stability ->
  infiltration_rate -> plant_available_water -> canopy_cover ->
  structural_heterogeneity -> pollinator_abundance -> crop_yield

legume_cover_crop -> soil_organic_carbon -> microbial_biomass_carbon ->
  nutrient_cycling_rate -> nitrogen_availability -> crop_yield
```

Measured mean across the evaluation: **9.92 variables per recommendation.**

### Uncertainty, propagated

Effect sizes are stored as distributions, not numbers. Joshi 2023 reports
cover crops raising soil organic carbon by 7.3% with a 95% CI of 4.9 to 9.6%,
so the edge holds that interval and samples from it.

Effects compose **multiplicatively in log space** along a path, never
additively. Where several mechanisms converge on one variable they combine by
**noisy-OR** at the node, not across whole paths — combining across paths
double-counts shared trunks, which is a real bug this project hit and fixed.

Output reports a median, a 90% interval, and the probability of exceeding a
threshold.

### Evidence quality and magnitude confidence are separate

Two studies disagreeing about the size of an effect while agreeing on its
direction is **better** evidence than one study asserting a number. Collapsing
those into one confidence field penalises transparency, so the system reports
both axes:

```
soil organic carbon: +7.2% (90% CI +4.9% to +11.8%)
  evidence: high      (two converging meta-analyses)
  magnitude: moderate (sources disagree on size: Joshi 7.3%, McClelland 12%)
```

Where sources conflict, the interval is widened to span the disagreement. The
figures are never averaged.

### Evidence roles

A methodological critique is not corroboration, and a null result is not
agreement with a positive one. Every evidence reference carries a role:
`primary`, `corroborating`, `contradicting`, or `critique`. Only the first two
count towards evidential convergence.

This is load-bearing. Alley cropping's effect on species richness cites GCB
2025 (positive), Mupepele 2021 (no unequivocal effect), and Boinot 2022 (a
critique of Mupepele's methodology). Counting three citations as three
agreements would have made the weakest evidence in the graph look like the
strongest.

### Liebig's law as a gate, not a weighting

Nothing downstream can be realised until the binding constraint lifts, so
interventions are partitioned into two tiers: those that address the
diagnosed limiting factor, and everything else. Tier 1 ranks above tier 2
entirely.

The consequence is site-conditional advice from one graph:

| | 340mm semi-arid | 2100mm, 12% slope | 620mm flat |
|---|---|---|---|
| Binding constraint | plant-available water | erosion rate | soil organic carbon |
| Top recommendation | contour bunding | contour bunding | legume cover crop |
| Movement | +7.5% water | −10.0% erosion | +7.2% SOC |

Alley cropping ranks 2nd at 900mm and **23rd of 23** at 340mm, because trees
compete for the water that is already binding. The system says so:

> At 340mm rainfall, plant-available water is the binding constraint. Tree
> competition for water makes this counterproductive until water harvesting
> is established.

### Conversation

A LangGraph state machine. Clarifying questions are graph interrupts, not a
loop, checkpointed so a conversation resumes.

Questions are ordered by **value of information**, not by field order. For
each unknown field the system samples plausible values, re-runs the ranking,
and measures how much the top-3 set churns. It asks about the highest-churn
field first. On the semi-arid site, slope carries 22% churn because contour
bunding's precondition satisfaction is halved without it, so slope gets asked
about before anything else.

It also checks physical plausibility. Given 2.5% soil organic carbon under
180mm of rainfall:

> A soil organic carbon of 2.5% under 180mm annual rainfall is unusual, since
> carbon accrual at that level normally requires more biomass production than
> that rainfall supports. Is the site irrigated, does it waterlog seasonally,
> or has it had heavy organic amendment? Or was that figure from a different
> depth or a different plot?

When a value is revised, the system reports the **diff**, not just a new
answer:

> You revised annual rainfall from low to 340mm. The binding constraint moves
> from soil organic carbon to plant-available water.
>   - Contour bunding moved up from rank 6 to rank 1.
>   - Legume-based cover crop moved down from rank 1 to rank 2.

### The critic

Every draft is decomposed into atomic claims and verified through two gates.
Quantitative claims are first checked for **traceability** to a propagation
result, then for **entailment** against a retrieved passage. That split
matters: no passage in the corpus contains a propagated Monte Carlo figure, so
entailment alone cannot verify one.

Coverage is reported by category, not as a bare percentage:

```
GROUNDING: 92%  (34 of 37 claims)
  traceable to propagation:      10
  entailed by retrieved passage:  22
  unsupported, softened:           2
  no corpus evidence available:    3   <- corpus gap, not a failed check
```

---

## Evaluation against an LLM-only baseline

The brief prohibits generic LLM-only solutions, so the project measures
against one. Twelve sites spanning the degradation syndromes, both systems on
the **same model** (`groq/openai/gpt-oss-120b`), so the comparison is not
confounded by model capability.

| Metric | This system | LLM baseline |
|---|---|---|
| Grounding coverage | 77% | 3% |
| Variables per recommendation | 9.92 | 5.50 |
| Citation validity | 100% | 2% |
| Unverifiable citations (total) | 0 | 93 |
| Required caveats raised | 100% | 58% |
| Tradeoffs surfaced (mean) | 0.50 | 0.00 |
| Binding constraint correct | 8/8 | 2/8 |
| Top recommendation acceptable | 7/7 | 2/7 |
| Quantified claims (mean) | 8.33 | **23.33** |
| Latency (mean s) | 15.2 | **8.6** |

**Where the baseline wins.** It is faster, it writes better prose, and it
asserts roughly three times as many quantified claims. Most of those claims
cannot be checked, which is the argument this evaluation makes, but a reader
who does not check them receives more specific-looking advice from the
baseline.

**What citation validity does not measure.** It measures whether a cited work
is registered in `sources.yaml`, the fifteen documents this project ingested.
For this system that is close to a tautology, since the renderer raises rather
than print an unregistered citation. For the baseline it tests verifiability
against *this* registry, not existence in the world. A baseline citation may
name a real paper and still count as unverifiable here.

Full per-site detail: [`data/derived/eval_summary.md`](data/derived/eval_summary.md)

---

## Knowledge base

**12 documents, 1,171 pages reduced to 528, 542 chunks, 201 extracted tables.**

Assessment reports supply mechanism and tradeoffs. Meta-analyses supply
quantified effect sizes. The distinction is a retrieval metadata filter: when
the critic verifies a number it searches meta-analyses first, because
assessment reports deliberately avoid point estimates.

**Assessments.** IPCC SRCCL Ch6 (response options), IPCC AR6 WG3 Ch7 (AFOLU),
IPBES Land Degradation and Restoration SPM, FAO State of Knowledge of Soil
Biodiversity, FAO GSOCseq Technical Report, FAO Voluntary Guidelines for
Sustainable Soil Management.

**Meta-analyses.** Joshi 2023 (cover crops, SOC, 61 studies), McClelland 2021
(cover crops, SOC, temperate), GCB 2025 (agroforestry multifunctionality),
Mupepele 2021 (European agroforestry, null result), Boinot 2022 (critique),
Woodcock 2019 (pollinator diversity and yield).

**Structured.** Takola et al. 2023, an open-access database of 41
meta-analyses and 298 effect sizes (CC-BY-NC-ND).

`ingest_manifest.yaml` records which page ranges of each document were
ingested and why the rest were not. The largest reduction was the FAO soil
biodiversity report, 616 pages to 196: Chapters 1 and 2 are a photographic
catalogue of soil organisms, pages 369 to 476 are country survey responses,
and the remainder is references. None of it carries reasoning content, and
chunking it would have crowded out substantive passages at retrieval time.

### The citation registry

`sources.yaml` is the single source of truth for citations. Every causal edge
references a `source_id` registered there, validated at graph build time, and
the renderer raises rather than emit an unregistered citation.

**Fabricated references are structurally impossible, not merely discouraged.**

Each entry also records the source's `scope`. When a site falls outside it,
the recommendation carries an explicit extrapolation note — McClelland is
scoped to temperate climates, so applying it to semi-arid India is flagged
rather than silently accepted.

### Retrieval

Dense (`bge-small-en-v1.5`) and sparse (BM25) retrieval fused by **reciprocal
rank fusion** (k=60), then optionally reranked by a cross-encoder.

RRF rather than score normalisation because dense cosine and BM25 scores are
not on comparable scales, so rank-based fusion needs no calibration.

Reranking is opt-in per call site: off for conversational turns (19ms), on for
evidence binding where a passage becomes a citation (1.6s). A `RERANK_FLOOR`
of 0.30 distinguishes *evidence found* from *retrieval returned the least-bad
matches* — retrieval always returns k results, which is not the same as the
corpus supporting the claim.

---

## Modelling assumptions

Five constants shape the numbers. None is empirical. All are exposed in the
system's own output rather than hidden.

| Constant | Value | What it models |
|---|---|---|
| `TRANSMISSION` | 0.75 | Per-edge signal loss along a causal chain. Pure multiplication would assume perfect transmission at every hop, which overstates long chains. |
| `DELTA_REF` | 0.10 | The upstream change against which a published effect is assumed measured. An edge fires at full strength when its cause moves 10%, proportionally less below that. Without this, a 2% upstream change would trigger a nearly full published effect. |
| `PRIORITY_TOLERANCE` | 0.10 | Constraint-relief differences within 10% relative are treated as tied, and evidence quality decides. An admission about Monte Carlo precision, not a claim about biophysics. |
| `TIER_1_MIN_EFFECT` | 0.01 | Floor below which an intervention does not count as addressing the binding constraint. |
| `RERANK_FLOOR` | 0.30 | Cross-encoder score below which a retrieved passage is treated as unsupportive. Chosen from observed score distributions. |

---

## Known limitations

Stated because a reviewer will find them anyway, and because a system that
reports its own gaps is more useful than one that does not.

**Evidence is thin and unevenly distributed.** Seven of 48 edges carry
published effect sizes. The remaining 41 are own estimates at `MECHANISTIC`
strength, citing assessment reports for **direction only**, never for a
number. They are labelled as such throughout. The consequence is that
interventions in well-studied areas (cover crops, soil carbon) carry stronger
support than erosion control, and the ranking reflects that asymmetry rather
than concealing it.

**Downstream figures are model outputs, not measurements.** The cover
crop to soil organic carbon figure reproduces Joshi's published estimate. The
crop yield figure at the end of an eight-hop chain is a projection resting on
mechanistic estimates and the constants above. It is presented with its
interval and its assumptions, not as a measurement.

**A missing edge that would change the flagship answer.** Boundary trees carry
a negative water edge gated to semi-arid zones; cover crops carry none, though
they also transpire and compete. This likely overstates cover crops below
300mm. It was not added because no published effect size was available and an
own estimate would have been decisive — recorded in `src/graph/edges.py` under
"Known modelling gaps" rather than quietly fixed.

**Corpus scope.** Every meta-analysis was gathered on cropland, so scope
checking reads climate and crop but not land use. Grazing-land
recommendations rest on globally scoped FAO guidance with no scope check
available to fire.

**Entailment softness in the eval run.** Of 220 claims sent to entailment, 21
were model-judged; 199 fell back to the retrieval floor after provider rate
limits. Those 199 count as supported on retrieval alone, which is the
remaining softness in the 77% figure.

**One intervention below the variable floor.** `vetiver_grass_strips` has a
single outgoing edge, so it can produce a one-variable recommendation on
erosion-limited sites. Caught by the eval, not yet fixed.

**WOCAT not integrated.** The WOCAT Global SLM Database (2,000+
field-documented practices with context preconditions and observed impacts)
was identified as high-value. A harvester was built, but the public API uses
short-lived client-side tokens that rotate within minutes. Deferred rather
than partially shipped.

**Deliberately cut.** Landscape fragmentation metrics from land-cover rasters,
and a species-richness expectation model. Both would have strengthened
diagnosis; both were bonus-scope and traded for evaluation and packaging.

---

## Running it

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

python -m src.retrieval.index      # build indexes (committed, so optional)
python -m src.demo                 # three worked sites, non-interactive
python -m src.agents.graph --demo  # five-turn conversation transcript
streamlit run app.py               # web interface
python -m src.eval.run             # evaluation, baselines cached
python -m pytest tests/ -q         # 111 tests
```

Set `GROQ_API_KEY` in `.env`, and `DARUKAA_MODEL=groq/openai/gpt-oss-120b`.
Language models are used for three things only: parsing free-text input,
entailment checking, and prose phrasing. **Every number, ranking, and citation
comes from the graph and the registry.**

Source PDFs are excluded from the repository (87MB, and redistribution is a
licensing grey area). The derived chunks, tables, causal graph, and search
indexes are committed, so the system runs from a clone without them.

### Layout

```
src/ingest/      PDF parsing, table extraction, chunking
src/graph/       schemas, causal graph, propagation engine
src/retrieval/   hybrid search, reranking, evidence binding
src/agents/      LangGraph state machine, rendering
src/eval/        test sites, metrics, LLM baseline
data/corpus/     source documents (gitignored)
data/derived/    chunks, tables, graph, indexes, eval results
sources.yaml     citation registry
```

---

## Attribution

Knowledge base: IPCC, IPBES, and FAO reports, all publicly available.
Takola et al. 2023 meta-analysis database (CC-BY-NC-ND 4.0).
Live data: ISRIC SoilGrids v2.0, NASA POWER, GBIF.

GBIF occurrence counts are sampling-effort biased and are reported as
observed-versus-expected, never as absolute species richness.
