"""Run the twelve eval sites through both systems and write the comparison.

    python -m src.eval.run                 both systems
    python -m src.eval.run --no-baseline   graph system only
    python -m src.eval.run --model X       baseline model override

Outputs:
    data/derived/eval_results.json   full per-site detail
    data/derived/eval_summary.md     the comparison table and per-site notes

Baseline responses are cached under data/derived/baseline_cache/ keyed on
site_id and model, so a rerun costs nothing and the published numbers are
reproducible from the cache without an API key.

This module is slow by nature: twelve sites through the full diagnose,
propagate, bind, synthesise and critic chain, each running a cross-encoder
over retrieved passages, plus one model call per site the first time the
baseline is generated. It is not collected by pytest, and a test that wraps
it belongs behind a `slow` marker rather than in the default suite.

How the graph system is run
---------------------------
The eval calls the agent nodes in sequence rather than driving the compiled
LangGraph, for one reason: a clarifying question is a graph interrupt, and a
batch run has nobody to answer it. So gap analysis runs and its question is
recorded, and then the chain continues to the answer the system would give if
pressed for one now. Intake is skipped because these sites arrive as
structured SiteState objects rather than as free text, which is what intake
exists to parse. Acquire runs where a site has coordinates, cache first, so
the sparse coordinates-only site exercises the acquisition path offline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from src.agents import nodes
from src.agents.state import Claim, initial_state
from src.eval.baseline import (
    DEFAULT_BASELINE_MODEL,
    DEPRECATED_BASELINE_MODEL,
    DEPRECATION_DATE,
    BaselineUnavailable,
    baseline_call,
    parse_baseline,
)
from src.eval.metrics import (
    RecommendationRecord,
    Scores,
    SystemRun,
    failed_scores,
    score_baseline,
    score_response,
)
from src.eval.sites import EvalSite, eval_sites
from src.graph.nodes import STATE_VARIABLES
from src.graph.propagate import RankedIntervention

_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = _ROOT / "data" / "derived" / "eval_results.json"
SUMMARY_PATH = _ROOT / "data" / "derived" / "eval_summary.md"
ENV_PATH = _ROOT / ".env"


def load_env() -> None:
    """Read provider credentials from .env into the environment.

    Called at the entry point rather than at import, so importing this module
    for its scoring functions does not quietly change the process
    environment. Existing environment variables win, so an operator who
    exported a key deliberately is not overridden by a stale file. .env is
    gitignored and no value read from it is logged, printed or written to
    either output artifact.
    """
    load_dotenv(ENV_PATH, override=False)


def _variables_behind(item: RankedIntervention) -> list[str]:
    """Distinct state variables on the causal paths behind one recommendation.

    This is the headline metric's raw material. It counts nodes the
    propagation engine actually traversed for this intervention at this site,
    so it cannot be inflated by naming a variable in prose.
    """
    seen: list[str] = []
    for result in item.effects.values():
        for path in result.paths:
            for node in path.path:
                if node in STATE_VARIABLES and node not in seen:
                    seen.append(node)
    return seen


def run_system(site_case: EvalSite) -> SystemRun:
    """One site through the graph system, start to finish."""
    site = site_case.site
    started = time.perf_counter()
    state = initial_state(site)

    try:
        if site.lat is not None and site.lon is not None:
            state.update(nodes.acquire_node(state))
        gap = nodes.gap_analysis_node(state)
        state.update(gap)
        pending_question = state.get("pending_question")

        state.update(nodes.diagnose_node(state))
        state.update(nodes.plan_node(state))
        state.update(nodes.bind_node(state))
        state.update(nodes.synthesise_node(state))
        state.update(nodes.critic_node(state))
    except Exception as error:  # noqa: BLE001 - reported as a site failure
        return SystemRun(
            site_id=site.site_id,
            completed=False,
            error=f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            latency_s=time.perf_counter() - started,
        )

    ranked: list[RankedIntervention] = state.get("ranked") or []
    diagnosis = state.get("diagnosis") or (None, "")
    claims = [c if isinstance(c, Claim) else Claim.model_validate(c) for c in (state.get("claims") or [])]

    return SystemRun(
        site_id=site.site_id,
        completed=True,
        latency_s=time.perf_counter() - started,
        limiting_factor=diagnosis[0],
        limiting_why=diagnosis[1],
        pending_question=pending_question,
        ranked=[item.intervention for item in ranked],
        recommendations=[
            RecommendationRecord(
                intervention=item.intervention,
                variables=_variables_behind(item),
                tradeoffs=len(item.tradeoffs),
            )
            for item in ranked[: nodes.RENDER_TOP_N]
        ],
        claims=claims,
        grounding_coverage=state.get("grounding_coverage"),
        draft=state.get("draft") or "",
    )


class SiteResult:
    """Both systems' output for one site, plus their scores.

    A plain object rather than a Pydantic model because it holds two
    already-validated models and a parsed baseline, and exists only to carry
    them between the runner and the reporters in this module.
    """

    def __init__(
        self,
        case: EvalSite,
        system: SystemRun,
        system_scores: Scores,
        baseline_text: str | None,
        baseline_scores: Scores | None,
        baseline_error: str | None,
    ) -> None:
        self.case = case
        self.system = system
        self.system_scores = system_scores
        self.baseline_text = baseline_text
        self.baseline_scores = baseline_scores
        self.baseline_error = baseline_error


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _accuracy(scores: list[Scores], field: str) -> tuple[int, int]:
    judged = [getattr(s, field) for s in scores if getattr(s, field) is not None]
    return sum(1 for v in judged if v), len(judged)


# One row per metric. The note says why the two columns differ, which is the
# part of a comparison table that carries the argument.
_METRIC_ROWS: list[tuple[str, str, str]] = [
    (
        "grounding_coverage",
        "Grounding coverage",
        "Not the same computation on both sides. The system's is its critic's: the "
        "fraction of empirical claims traceable to a propagation result or entailed by "
        "a retrieved passage. The baseline has neither, so the nearest analogue is used, "
        "the fraction of its claims carrying a citation that resolves against "
        "sources.yaml. Read the two as answers to the same question, not as one number.",
    ),
    (
        "mean_variables_per_recommendation",
        "Variables per recommendation",
        "The brief's core differentiator, floor of three. The system counts distinct "
        "state variables on the causal paths the propagation engine actually traversed "
        "for that intervention at that site, across every scored objective. The baseline "
        "counts distinct environmental variables its prose names anywhere in the "
        "recommendation. The baseline's is the more generous count of the two, since "
        "naming a variable is cheaper than propagating through it.",
    ),
    (
        "citation_validity",
        "Citation validity",
        "Fraction of cited works registered in sources.yaml. The system prints citation "
        "strings read out of that registry, so this is 1.0 by construction. For the "
        "baseline it measures verifiability against this registry of fifteen documents, "
        "NOT whether the paper exists in the world.",
    ),
    (
        "fabricated_citations",
        "Unverifiable citations (total)",
        "Cited works that do not resolve to a registered source. Zero for the system is "
        "structural: the renderer raises rather than print an unregistered citation. For "
        "the baseline this counts citations this system cannot check, some of which are "
        "likely real papers.",
    ),
    (
        "required_flags_present",
        "Required flags raised",
        "Fraction of the caveats each site demanded that the response actually stated: "
        "extrapolation beyond a source's scope, an implausible input pairing, a "
        "diagnosis reached by elimination, a constraint nothing in the graph addresses.",
    ),
    (
        "tradeoffs_surfaced",
        "Tradeoffs surfaced (mean)",
        "The system counts tradeoffs the engine found by propagating to opposite-signed "
        "targets. The baseline counts mentions of the word, which is the generous "
        "reading of its prose.",
    ),
    (
        "quantified_claims",
        "Quantified claims (mean)",
        "Lines asserting a percentage, counted the same way on both sides.",
    ),
    (
        "claims_with_ci",
        "Quantified claims with an interval",
        "Fraction of those lines that also stated an uncertainty interval. Both systems "
        "were asked for intervals; only one of them cannot omit them.",
    ),
    (
        "latency_s",
        "Latency (mean seconds)",
        "Wall clock per site, mean across the twelve with the first site in brackets. "
        "Read the bracket, not the mean: the system's retrieval and cross-encoder results "
        "are cached per process, so the first site pays for work the other eleven reuse "
        "and several later sites finish in under a second. The mean therefore says the "
        "system is faster, and that is an artefact of running twelve sites in one "
        "process. On a cold single-site run, which is what a user experiences, the "
        "baseline is the faster of the two. The baseline's own figure times the "
        "generating call, carried in the cache, not the cache read.",
    ),
]


def _aggregate(scores: list[Scores], field: str) -> float:
    return _mean([float(getattr(s, field)) for s in scores])


def _format_metric(field: str, scores: list[Scores]) -> str:
    if not scores:
        return "not run"
    if field == "fabricated_citations":
        return f"{sum(s.fabricated_citations for s in scores)}"
    value = _aggregate(scores, field)
    if field in ("grounding_coverage", "citation_validity", "required_flags_present", "claims_with_ci"):
        return f"{value:.0%}"
    if field == "latency_s":
        # The first site's figure alongside the mean, because it is the only
        # one in the run that paid full price for model load and retrieval.
        return f"{value:.1f} (first site {scores[0].latency_s:.1f})"
    return f"{value:.2f}"


def _summary_markdown(
    results: list[SiteResult], model: str, include_baseline: bool
) -> str:
    system_scores = [r.system_scores for r in results]
    baseline_scores = [r.baseline_scores for r in results if r.baseline_scores is not None]

    lines: list[str] = []
    lines.append("# Evaluation: causal graph system vs LLM-only baseline")
    lines.append("")
    lines.append(
        "Twelve test sites spanning the degradation syndromes, each run through both "
        "systems. Generated by `python -m src.eval.run`."
    )
    lines.append("")
    lines.append(
        "Every metric below is computed from the response object or from the text the "
        "system emitted. No language model judges any output here."
    )
    lines.append("")

    lines.append("## Run provenance")
    lines.append("")
    lines.append(f"- Eval run: **{time.strftime('%Y-%m-%d', time.gmtime())}** (UTC)")
    lines.append(f"- Baseline model: **`{model}`**")
    lines.append(
        f"- `{DEPRECATED_BASELINE_MODEL}` was the obvious choice for this baseline until "
        f"Groq deprecated it on **{DEPRECATION_DATE}**, naming `gpt-oss-120b` as the "
        f"recommended replacement. The baseline runs on the replacement. The model ID is "
        f"recorded here because the numbers below are not reproducible without it: a "
        f"different model gives a different baseline."
    )
    lines.append(
        f"- The graph system's own three language-model touchpoints (intake top-up, "
        f"entailment, prose phrasing) were configured as `{nodes.model_name()}` and were "
        + (
            "available for this run."
            if nodes.llm_available()
            else (
                "**not available for this run**, so each fell back to its deterministic "
                "path. That matters most for the critic: with no entailment model, a "
                "claim whose retrieved passage clears the cross-encoder support floor is "
                "counted as supported on retrieval alone rather than on a judgement that "
                "the passage entails it. The grounding figure below should be read that "
                "way. Set DARUKAA_MODEL to run the critic with an entailment model."
            )
        )
    )
    lines.append(
        "- Baseline responses are cached under `data/derived/baseline_cache/`, keyed on "
        "site_id and model, so this table can be regenerated without an API key and "
        "without re-billing a single call."
    )
    lines.append("")

    failures = [r for r in results if not r.system.completed]
    baseline_failures = [r for r in results if include_baseline and r.baseline_scores is None]
    if failures:
        lines.append(
            f"**{len(failures)} of {len(results)} sites failed to complete in the graph "
            f"system** and are counted as failures rather than dropped. See the per-site "
            f"section."
        )
        lines.append("")
    if include_baseline and baseline_failures:
        lines.append(
            f"**The baseline produced no response for {len(baseline_failures)} of "
            f"{len(results)} sites.** Those sites are excluded from the baseline column "
            f"rather than scored as zero, which flatters the baseline; the per-site "
            f"section says which they were and why."
        )
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | This system | LLM baseline | Why the difference |")
    lines.append("| --- | --- | --- | --- |")
    for field, label, note in _METRIC_ROWS:
        baseline_cell = _format_metric(field, baseline_scores) if include_baseline else "not run"
        lines.append(
            f"| {label} | {_format_metric(field, system_scores)} | {baseline_cell} | {note} |"
        )

    for field, label in (
        ("limiting_factor_correct", "Binding constraint correct"),
        ("top_intervention_acceptable", "Top recommendation acceptable"),
    ):
        correct, judged = _accuracy(system_scores, field)
        system_cell = f"{correct}/{judged}" if judged else "not scored"
        if include_baseline and baseline_scores:
            b_correct, b_judged = _accuracy(baseline_scores, field)
            baseline_cell = f"{b_correct}/{b_judged}" if b_judged else "not scored"
        else:
            baseline_cell = "not run"
        lines.append(
            f"| {label} | {system_cell} | {baseline_cell} | Scored only on the sites where "
            f"a defensible answer exists. Four sites carry no expectation because the "
            f"input does not support one, and inventing certainty there to make scoring "
            f"easier would be the same error the eval is testing for. |"
        )
    lines.append("")

    completed = sum(1 for r in results if r.system.completed)
    lines.append(
        f"Sites completed: {completed}/{len(results)} (this system)"
        + (
            f", {len(baseline_scores)}/{len(results)} (baseline)"
            if include_baseline
            else ""
        )
    )
    lines.append("")

    lines.append("## Where the gap is largest and where it closes")
    lines.append("")
    lines.extend(_gap_lines(system_scores, baseline_scores, include_baseline))
    lines.append("")

    lines.append("## Where the baseline wins or ties")
    lines.append("")
    lines.extend(_win_lines(system_scores, baseline_scores, include_baseline))
    lines.append("")

    lines.append("## What citation validity does and does not measure")
    lines.append("")
    lines.append(
        "It measures whether a cited work is registered in `sources.yaml`, the fifteen "
        "documents this project ingested. For the graph system that is a tautology: the "
        "renderer reads citation strings out of the registry by source_id and raises "
        "rather than print one it cannot resolve. For the baseline it is a real test, but "
        "of verifiability against this registry rather than of existence in the world. A "
        "baseline citation may name a real, correct, well known paper and still count as "
        "unverifiable here, because nothing in this system can check it. That is the "
        "honest framing, and the number should not be read as a fabrication rate."
    )
    lines.append("")

    lines.append("## Per-site detail")
    lines.append("")
    for result in results:
        lines.extend(_site_section(result, include_baseline))
    return "\n".join(lines)


# Which direction is better for each metric. Needed to say who is ahead
# without reading the sign off the metric name.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "grounding_coverage": True,
    "mean_variables_per_recommendation": True,
    "citation_validity": True,
    "required_flags_present": True,
    "tradeoffs_surfaced": True,
    "quantified_claims": True,
    "claims_with_ci": True,
    "latency_s": False,
    "fabricated_citations": False,
}


def _gap_lines(
    system_scores: list[Scores], baseline_scores: list[Scores], include_baseline: bool
) -> list[str]:
    """Metrics ordered by how far apart the two systems are.

    The gap is relative, as a fraction of the larger of the two values, so
    metrics on different scales can be ranked against each other. A gap of
    zero means the two are level on that metric, which is as much a result as
    a large one.
    """
    if not include_baseline or not baseline_scores:
        return [
            "The baseline did not run, so there is no gap to rank. Rerun with a provider "
            "API key set to fill this section."
        ]

    gaps: list[tuple[float, str]] = []
    for field, label, _note in _METRIC_ROWS:
        system_value = _aggregate(system_scores, field)
        baseline_value = _aggregate(baseline_scores, field)
        scale = max(abs(system_value), abs(baseline_value))
        relative = abs(system_value - baseline_value) / scale if scale else 0.0
        if _HIGHER_IS_BETTER[field]:
            leader = "this system" if system_value > baseline_value else "the baseline"
        else:
            leader = "this system" if system_value < baseline_value else "the baseline"
        if abs(system_value - baseline_value) < 1e-9:
            leader = "level"
        gaps.append(
            (
                relative,
                f"- **{label}**: this system {system_value:.2f}, baseline "
                f"{baseline_value:.2f} ({relative:.0%} apart, {leader} ahead)",
            )
        )

    gaps.sort(key=lambda item: -item[0])
    lines = ["Largest gaps:", ""]
    lines.extend(text for _, text in gaps[:3])
    lines.append("")
    lines.append("Smallest gaps, or reversed:")
    lines.append("")
    lines.extend(text for _, text in reversed(gaps[-3:]))
    return lines


def _win_lines(
    system_scores: list[Scores], baseline_scores: list[Scores], include_baseline: bool
) -> list[str]:
    """The rows where the baseline is ahead or level, stated plainly.

    A comparison that shows a clean sweep is not credible. The baseline is
    one model call against a Monte Carlo plus retrieval plus a critic pass, so
    it is going to win on latency, and it writes fluent prose that names more
    variables than it reasons through. Both belong in the table.
    """
    if not include_baseline or not baseline_scores:
        return [
            "The baseline did not run, so nothing can be claimed about where it wins. "
            "It is faster than this system by roughly the cost of a Monte Carlo, a "
            "hybrid retrieval and a critic pass, and that gap is real whether or not it "
            "was measured today."
        ]

    lines: list[str] = []
    for field, label, _note in _METRIC_ROWS:
        system_value = _aggregate(system_scores, field)
        baseline_value = _aggregate(baseline_scores, field)
        better_high = _HIGHER_IS_BETTER[field]
        if better_high:
            baseline_ahead = baseline_value > system_value
            tied = abs(baseline_value - system_value) < 1e-9
        else:
            baseline_ahead = baseline_value < system_value
            tied = abs(baseline_value - system_value) < 1e-9
        if baseline_ahead:
            lines.append(
                f"- **{label}**: baseline {baseline_value:.2f} vs this system "
                f"{system_value:.2f}. The baseline is ahead."
            )
        elif tied:
            lines.append(f"- **{label}**: tied at {system_value:.2f}.")
    if not lines:
        lines.append(
            "- The baseline led on no metric in this run. That is a claim about these "
            "twelve sites and this parser, not a general result: the parser reads the "
            "baseline's prose with regexes, and every extraction it misses counts "
            "against the baseline."
        )

    # Three things the table gets wrong in this system's favour, or cannot
    # measure at all. They belong next to the wins rather than in a footnote,
    # because a comparison that only reports its own strengths is not
    # evidence.
    cold = system_scores[0].latency_s if system_scores else 0.0
    baseline_latency = _aggregate(baseline_scores, "latency_s")
    lines.append("")
    lines.append("Three caveats that run the same way:")
    lines.append("")
    lines.append(
        f"- **Latency, properly read, goes to the baseline.** The mean above favours this "
        f"system only because twelve sites share one process and one warm cache. The "
        f"first site, which is the honest figure for a single run, took "
        f"{cold:.1f}s against the baseline's {baseline_latency:.1f}s average call. A user "
        f"asking about one site waits longer for this system, and no amount of caching "
        f"changes that for the first question of a session."
    )
    lines.append(
        "- **The baseline writes better prose.** It is fluent, it adapts its structure to "
        "the site, and it reads like an expert wrote it. This system emits a fixed report "
        "layout with padded columns. Nothing in the table measures that, and it matters "
        "for whether anyone reads the output at all."
    )
    lines.append(
        "- **The baseline asserts far more.** It states around three times as many "
        "quantified claims per site. Most of them cannot be checked, which is the "
        "argument this eval makes, but a reader who does not check them receives more "
        "specific-looking advice from the baseline than from this system."
    )
    return lines


def _site_section(result: SiteResult, include_baseline: bool) -> list[str]:
    case = result.case
    scores = result.system_scores
    lines = [f"### {case.site.site_id}", ""]
    lines.append(f"{case.notes}")
    lines.append("")
    lines.append(
        f"- Expected binding constraint: `{case.expected_limiting_factor or 'none set'}`"
    )
    if case.expected_top_intervention_class:
        lines.append(
            "- Acceptable top recommendations: "
            + ", ".join(f"`{name}`" for name in case.expected_top_intervention_class)
        )
    else:
        lines.append("- Acceptable top recommendations: none set, see the note above")
    lines.append(
        "- Required flags: "
        + (", ".join(f"`{flag}`" for flag in case.must_flag) if case.must_flag else "none")
    )
    lines.append("")

    if not result.system.completed:
        lines.append("**This system: FAILED**")
        lines.append("")
        lines.append("```")
        lines.append((result.system.error or "").strip())
        lines.append("```")
        lines.append("")
    else:
        lines.append(
            f"- This system: constraint `{result.system.limiting_factor}`, top pick "
            f"`{result.system.ranked[0] if result.system.ranked else 'none'}`, "
            f"{scores.mean_variables_per_recommendation:.1f} variables per "
            f"recommendation, grounding {scores.grounding_coverage:.0%}, "
            f"{scores.quantified_claims} quantified claims, "
            f"{scores.claims_with_ci:.0%} with an interval, "
            f"flags {scores.required_flags_present:.0%}, {scores.latency_s:.1f}s"
        )
        if result.system.pending_question:
            lines.append(
                "- It asked a clarifying question first: "
                + result.system.pending_question.splitlines()[-1]
            )

    if include_baseline:
        if result.baseline_scores is None:
            lines.append(f"- Baseline: no response ({result.baseline_error})")
        else:
            b = result.baseline_scores
            lines.append(
                f"- Baseline: constraint `{_parsed_constraint(result)}`, top pick "
                f"`{_parsed_top(result)}`, "
                f"{b.mean_variables_per_recommendation:.1f} variables per "
                f"recommendation, grounding {b.grounding_coverage:.0%}, "
                f"{b.quantified_claims} quantified claims, "
                f"{b.claims_with_ci:.0%} with an interval, "
                f"flags {b.required_flags_present:.0%}, "
                f"{b.fabricated_citations} unverifiable citations, {b.latency_s:.1f}s"
            )
    lines.append("")
    return lines


def _parsed_constraint(result: SiteResult) -> str:
    if result.baseline_text is None:
        return "none"
    parsed = parse_baseline(result.baseline_text, result.case.site.site_id)
    return parsed.limiting_factor or "not stated in a recognised form"


def _parsed_top(result: SiteResult) -> str:
    if result.baseline_text is None:
        return "none"
    parsed = parse_baseline(result.baseline_text, result.case.site.site_id)
    return parsed.top_intervention or "not mapped to a graph intervention"


def _results_json(results: list[SiteResult], model: str, include_baseline: bool) -> dict:
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_model": model if include_baseline else None,
        "sites": [
            {
                "site_id": r.case.site.site_id,
                "site": json.loads(r.case.site.model_dump_json()),
                "expected_limiting_factor": r.case.expected_limiting_factor,
                "expected_top_intervention_class": r.case.expected_top_intervention_class,
                "must_flag": r.case.must_flag,
                "notes": r.case.notes,
                "system": {
                    "completed": r.system.completed,
                    "error": r.system.error,
                    "latency_s": r.system.latency_s,
                    "limiting_factor": r.system.limiting_factor,
                    "limiting_why": r.system.limiting_why,
                    "pending_question": r.system.pending_question,
                    "ranked": r.system.ranked,
                    "recommendations": [
                        json.loads(rec.model_dump_json()) for rec in r.system.recommendations
                    ],
                    "grounding_coverage": r.system.grounding_coverage,
                    "draft": r.system.draft,
                    "scores": json.loads(r.system_scores.model_dump_json()),
                },
                "baseline": (
                    {
                        "error": r.baseline_error,
                        "text": r.baseline_text,
                        "scores": (
                            json.loads(r.baseline_scores.model_dump_json())
                            if r.baseline_scores is not None
                            else None
                        ),
                    }
                    if include_baseline
                    else None
                ),
            }
            for r in results
        ],
    }


def run_eval(include_baseline: bool = True, model: str | None = None) -> list[SiteResult]:
    """Run every site through both systems and write both artifacts."""
    load_env()
    model = model or DEFAULT_BASELINE_MODEL
    cases = eval_sites()

    # Load the embedding model, the cross-encoder and the indexes once, so
    # the first site's latency is not the model load time of all twelve.
    from src.retrieval.search import warmup

    started = time.perf_counter()
    warmup()
    print(f"warmed retrieval models and indexes in {time.perf_counter() - started:.1f}s\n")

    results: list[SiteResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.site.site_id}")
        run = run_system(case)
        scores = score_response(case, run)
        status = "ok" if run.completed else "FAILED"
        print(f"  system: {status} in {run.latency_s:.1f}s")
        if not run.completed:
            print(f"  {(run.error or '').splitlines()[0]}")

        baseline_text: str | None = None
        baseline_scores: Scores | None = None
        baseline_error: str | None = None
        if include_baseline:
            try:
                call = baseline_call(case.site, model)
            except BaselineUnavailable as error:
                baseline_error = str(error)
                print(f"  baseline: unavailable ({error})")
            except Exception as error:  # noqa: BLE001 - reported as a site failure
                baseline_error = f"{type(error).__name__}: {error}"
                print(f"  baseline: FAILED ({baseline_error})")
            else:
                baseline_text = call.text
                parsed = parse_baseline(call.text, case.site.site_id, model)
                # The generating call's latency, carried in the cache, not the
                # cache read's. See BaselineCall.
                baseline_scores = score_baseline(case, parsed, call.latency_s)
                source = "cache" if call.cached else "model"
                print(f"  baseline: ok from {source}, {call.latency_s:.1f}s to generate")

        results.append(
            SiteResult(case, run, scores, baseline_text, baseline_scores, baseline_error)
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(_results_json(results, model, include_baseline), indent=2)
    )
    summary = _summary_markdown(results, model, include_baseline)
    SUMMARY_PATH.write_text(summary)

    print()
    print(summary.split("## Per-site detail")[0])
    print(f"wrote {RESULTS_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Darukaa.Earth evaluation harness")
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the LLM-only baseline"
    )
    parser.add_argument("--model", default=None, help=f"baseline model (default {DEFAULT_BASELINE_MODEL})")
    args = parser.parse_args(argv)

    results = run_eval(include_baseline=not args.no_baseline, model=args.model)
    failed = [r for r in results if not r.system.completed]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
