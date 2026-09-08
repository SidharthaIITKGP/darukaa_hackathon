# Darukaa.Earth — AI Environmental Scientist

Conversational system that diagnoses land degradation and recommends
interventions, each with a quantified effect, time horizon, confidence level,
and a citation to a real study.

Hackathon submission. Deadline: 2026-09-09 EOD. Optimise for correctness of
reasoning over feature count.

## Environment

- Python 3.12, venv at `.venv/` managed by **uv**
- Install with `uv pip install <pkg>`. **Never** run bare `pip install`, and
  **never** create a new venv — one already exists.
- Run scripts with the venv active.

## The core idea (read this before writing any code)

The centrepiece is **not RAG**. It is a curated, signed, weighted **causal
graph** of environmental variables, plus a Monte Carlo engine that propagates
interventions through it. RAG sits *underneath*, attaching evidence to edges
the graph already has.

This is **not GraphRAG**. Microsoft-style GraphRAG builds an entity graph
*from* text in order to retrieve text. Here the graph is a hand-curated
domain ontology whose purpose is to *compute*. Documents point at the graph,
not the reverse. Do not implement entity extraction over the corpus as a
retrieval strategy, and do not describe this system as GraphRAG in any
comment, docstring, or README text.

Nodes are of two kinds:
- **State variables** — soil_organic_carbon, ph, infiltration_rate,
  plant_available_water, microbial_biomass_carbon, aggregate_stability,
  canopy_cover, structural_heterogeneity, pollinator_abundance,
  species_richness, erosion_rate, crop_yield
- **Interventions** — legume_cover_crop, alley_cropping, hedgerow_planting,
  contour_bunding, reduced_tillage, farmyard_manure, residue_retention

Edges carry: sign, effect-size distribution, time lag, context preconditions,
evidence refs, and evidence strength.

## Non-negotiable conventions

1. **Citations.** Every emitted claim references a `source_id` that exists in
   `sources.yaml`. If it isn't registered there, it must not appear in
   output. Never invent a citation, DOI, or page number.

2. **Effect sizes are distributions, never point values.** Store as a CI plus
   a distribution family. A published "7.3% (95% CI 4.9–9.6%)" becomes
   lognormal parameters, not the float 0.073.

3. **Compose effects multiplicatively in log space** along a causal path.
   Never sum percentage changes. Combine multiple converging paths with
   noisy-OR (`1 - prod(1 - contribution)`), never by addition — summing
   double-counts shared mechanisms.

4. **Effect-size metrics are not interchangeable.** LRR of 0.18 is ~19.7%
   (`exp(0.18) - 1`), not 18%. Hedges' d is not a percentage at all. Write
   one conversion function per metric and use it everywhere.

5. **Encode disagreement, do not reconcile it.** Joshi 2023 reports +7.3% for
   cover crops on SOC; McClelland 2021 reports +12%. Represent as a wider
   distribution with `confidence: moderate` and both source_ids. Do not
   average them and do not pick one.

6. **Flag extrapolation.** Each source in `sources.yaml` has a `scope`. When
   a site falls outside it (e.g. applying temperate cover-crop evidence to
   semi-arid India), the recommendation must carry an explicit note.

7. **Multi-variable reasoning is mandatory.** No recommendation may be
   justified by a single variable. The brief calls this the core
   differentiator and sets a floor of three environmental variables.

8. **Never let a live API call block the demo.** Cache-first, then live, then
   degrade to asking the user. `data/cache/` is committed for demo sites.

## Data layout

```
data/
  corpus/assessments/     6 PDFs — mechanism, tradeoffs. NOT effect sizes.
  corpus/meta_analyses/   6 PDFs — quantified effect sizes. Cite for numbers.
  structured/takola_2023/ CSVs — 298 effect sizes. Parsed to edges, NOT embedded.
  rasters/gsocseq/        Optional GeoTIFFs — point lookups at runtime.
  cache/                  Prefetched API responses. Committed.
  derived/                Generated. Gitignored. Rebuildable.
```

`ingest_manifest.yaml` defines which page ranges of each PDF to ingest —
1171 raw pages reduced to 528. Honour it; do not ingest whole PDFs.

`sources.yaml` is the citation registry and records per-source caveats.
**Read both files before writing ingestion or graph code.**

### Takola CSV caveats (all four matter)

- `Management_grouped` has only 5 coarse values and does **not** map to
  intervention nodes. Use `Management_raw` free text and map by hand.
- Rows are per-**estimate**, not per-intervention. Multiple rows share a
  `StudyID`. Pool within study; never treat each row as an independent edge.
- **Sign conventions vary by study.** Verify direction per row. Several
  `diversification` rows are negative where positive is expected.
- Metrics are mixed: LRR (164), Percentage change (106), HedgesD (23),
  CohensD (4), HedgesG (1).

## Layout for code

```
src/
  ingest/      PDF parsing, chunking, table extraction, embedding
  graph/       schemas, causal graph, propagation engine
  retrieval/   hybrid search, reranking
  agents/      LangGraph nodes and state machine
  eval/        test sites, scoring, LLM baseline comparison
tests/
```

## Style

- Pydantic v2 for every data contract. No bare dicts crossing module lines.
- Type hints everywhere. Fail loud on bad input; no silent `except: pass`.
- No new dependencies without asking.
- Prose in docstrings and comments: plain, no em-dashes, no "not X but Y"
  constructions.

## Working agreement

- Change only the files named in the current task. Do not refactor
  neighbouring code opportunistically.
- End every task by running the verification command given in the prompt and
  showing its actual output. Do not claim success without running it.
- If a spec seems wrong, say so before implementing it.
- Do not create README files, summary documents, or progress reports unless
  explicitly asked.

## Build stages (current position: Stage 1)

0. Knowledge base collected, organised, manifested — **done**
1. Ingestion → `chunks.jsonl`, `tables.jsonl`
2. Schemas — `SiteState`, `CausalEdge`
3. Causal graph — nodes and ~40-60 curated edges
4. Propagation engine — Monte Carlo, tradeoff detection
5. Retrieval — hybrid + RRF + cross-encoder rerank
6. Agent graph — LangGraph, clarifying questions, memory
7. Geo — fragmentation metrics (cuttable)
8. Eval — test sites, scoring, LLM-only baseline comparison
9. Interface and packaging

Never cut: the causal graph, the propagation engine, the critic loop, the
eval baseline comparison. Those four are what the rubric rewards.
