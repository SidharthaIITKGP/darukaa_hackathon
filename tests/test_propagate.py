"""Tests for the Monte Carlo propagation engine."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.graph.edges import EDGES, build_graph
from src.graph.propagate import (
    Confidence,
    find_tradeoffs,
    limiting_factor,
    propagate,
    rank_interventions,
)
from src.graph.schemas import (
    CausalEdge,
    Conditions,
    Distribution,
    EffectMetric,
    EvidenceRef,
    EvidenceStrength,
    Measurement,
    Provenance,
    SiteState,
)


@pytest.fixture(scope="module")
def graph() -> nx.MultiDiGraph:
    return build_graph()


def _fully_satisfying_site() -> SiteState:
    return SiteState(
        site_id="fully_satisfying",
        annual_rainfall_mm=Measurement(
            value=800, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        soil_organic_carbon_pct=Measurement(
            value=1.5, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        ph=Measurement(value=6.5, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE),
        clay_pct=Measurement(value=25, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE),
        slope_pct=Measurement(value=5, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )


def _deccan_site() -> SiteState:
    return SiteState(
        site_id="deccan_test",
        lat=17.85,
        lon=75.42,
        soil_organic_carbon_pct=Measurement(
            value=0.35, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=340, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(value=8.1, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )


def test_legume_cover_crop_soc_p50(graph: nx.MultiDiGraph) -> None:
    site = _fully_satisfying_site()
    result = propagate(
        graph, "legume_cover_crop", "soil_organic_carbon", site, n=20_000, rng=np.random.default_rng(0)
    )
    # Single published edge (Joshi pooled 0-15cm + McClelland), point-anchored
    # at 0.073. The no-till subgroup estimate was removed as a separate edge
    # (its condition can never be gated without a tillage-state node) and
    # folded into this edge's mechanism/evidence note instead.
    assert 0.015 < result.p50 < 0.08
    assert result.ci90[0] < result.ci90[1]


def test_first_hop_edge_reproduces_published_value(graph: nx.MultiDiGraph) -> None:
    """With TRANSMISSION skipped at the point of application and delta[intervention] = 0,
    a lone published edge out of the intervention must reproduce its own CI exactly:
    local * (1 + 0) * 1.0 == local. Verified in isolation (a single-edge graph) so
    parallel-edge mixture doesn't confound the reproduction check.

    NOTE: the target is the edge's `point` value (0.073), not the geometric mean of
    the CI (sqrt(0.049 * 0.13) = 0.0798). Distribution.sample() honours `point` as the
    lognormal's median when it is set (schemas.py), deriving sigma from the wider side
    of the interval so the samples still span the published CI."""
    published_edge = next(
        e
        for e in EDGES
        if e.source == "legume_cover_crop" and e.target == "soil_organic_carbon" and e.effect.point == 0.073
    )
    g = nx.MultiDiGraph()
    g.add_node("legume_cover_crop", kind="intervention")
    g.add_node("soil_organic_carbon", kind="state_variable")
    g.add_edge("legume_cover_crop", "soil_organic_carbon", edge=published_edge)

    site = _fully_satisfying_site()
    result = propagate(
        g, "legume_cover_crop", "soil_organic_carbon", site, n=20_000, rng=np.random.default_rng(0)
    )
    expected_median = published_edge.effect.point
    assert abs(result.p50 - expected_median) / expected_median < 0.05


def test_parallel_edges_mix_not_noisy_or() -> None:
    """Parallel edges are alternative estimates of one relationship, combined as a
    weighted mixture. A mixture's median sits between its inputs' medians; a
    noisy-OR would push it above both, which is what noisy-OR-over-alternative-
    estimates incoherently implies -- that adopting the intervention triggers both
    effects at once. Verified on a synthetic two-edge graph (the real graph's
    legume_cover_crop -> soil_organic_carbon parallel edge was removed: it was an
    unconditional firing of a conditional no-till subgroup estimate, see Section A
    of edges.py)."""
    g = nx.MultiDiGraph()
    g.add_node("a", kind="intervention")
    g.add_node("b", kind="state_variable")
    low_edge = CausalEdge(
        source="a",
        target="b",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.03, ci_high=0.08, point=0.05),
        lag_years=(0, 1),
        strength=EvidenceStrength.MECHANISTIC,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="test_source", role="primary")],
        mechanism="test",
    )
    high_edge = CausalEdge(
        source="a",
        target="b",
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.05, ci_high=0.10, point=0.073),
        lag_years=(0, 1),
        strength=EvidenceStrength.META_ANALYSIS,
        confidence=Confidence.MODERATE,
        evidence=[EvidenceRef(source_id="test_source", role="primary")],
        mechanism="test",
    )
    g.add_edge("a", "b", edge=low_edge)
    g.add_edge("a", "b", edge=high_edge)

    site = _fully_satisfying_site()
    result = propagate(g, "a", "b", site, n=20_000, rng=np.random.default_rng(0))
    assert len(result.paths) == 2
    individual_p50s = [p.p50 for p in result.paths]
    assert min(individual_p50s) < result.p50 < max(individual_p50s)


def test_multiplicative_composition_not_additive() -> None:
    g = nx.MultiDiGraph()
    g.add_node("a", kind="intervention")
    g.add_node("b", kind="state_variable")
    g.add_node("c", kind="state_variable")
    edge_kwargs = dict(
        sign="+",
        metric=EffectMetric.PERCENT_CHANGE,
        effect=Distribution(family="lognormal", ci_low=0.10, ci_high=0.10 + 1e-9, point=0.10),
        lag_years=(0, 0),
        strength=EvidenceStrength.EXPERT,
        confidence=Confidence.LOW,
        evidence=[EvidenceRef(source_id="test_source", role="primary")],
        mechanism="test",
    )
    g.add_edge("a", "b", edge=CausalEdge(source="a", target="b", **edge_kwargs))
    g.add_edge("b", "c", edge=CausalEdge(source="b", target="c", **edge_kwargs))

    site = SiteState(site_id="unit_test")
    result = propagate(g, "a", "c", site, n=1000, rng=np.random.default_rng(0))

    # Magnitude comes from node-wise forward_propagate. "a" is the
    # intervention: delta[a] = 0 (binary presence, not a proportional
    # change) and its first-hop edge a -> b skips TRANSMISSION and the
    # DELTA_REF scale (a published edge off the intervention already IS the
    # end-to-end effect):
    #   a -> b: transmitted = 0.10                                          = 0.10   -> delta[b] = 0.10
    #   b -> c: scale = clip(delta[b] / DELTA_REF, 0, 1.5) = clip(0.10/0.10) = 1.0
    #           transmitted = 0.10 * 1.0 * 0.75                             = 0.075  -> delta[c] = 0.075
    # Not 0.20 (additive) and not 0.21 (undamped multiplicative composition).
    assert abs(result.p50 - 0.075) < 0.001
    assert abs(result.p50 - 0.20) > 0.005
    assert abs(result.p50 - 0.21) > 0.005


def test_negative_edge_yields_negative_p50(graph: nx.MultiDiGraph) -> None:
    site = SiteState(
        site_id="low_rainfall",
        annual_rainfall_mm=Measurement(
            value=300, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )
    result = propagate(graph, "alley_cropping", "crop_yield", site, n=5000, rng=np.random.default_rng(0))
    assert result.p50 < 0


def test_unmet_rainfall_condition_shrinks_effect(graph: nx.MultiDiGraph) -> None:
    site_met = SiteState(
        site_id="slope_met",
        slope_pct=Measurement(value=10, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )
    site_unmet = SiteState(
        site_id="slope_unmet",
        slope_pct=Measurement(value=0, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )
    result_met = propagate(
        graph, "contour_bunding", "infiltration_rate", site_met, n=5000, rng=np.random.default_rng(0)
    )
    result_unmet = propagate(
        graph, "contour_bunding", "infiltration_rate", site_unmet, n=5000, rng=np.random.default_rng(0)
    )
    assert abs(result_unmet.p50) < abs(result_met.p50)


def test_mechanistic_only_path_gives_low_confidence(graph: nx.MultiDiGraph) -> None:
    site = _fully_satisfying_site()
    result = propagate(
        graph, "reduced_tillage", "microbial_biomass_carbon", site, n=5000, rng=np.random.default_rng(0)
    )
    assert result.magnitude_confidence == Confidence.LOW
    assert result.evidence_quality == Confidence.LOW


def test_alley_cropping_species_richness_evidence_quality_is_low(graph: nx.MultiDiGraph) -> None:
    """Three citations, but not three agreements. GCB 2025 is primary, Mupepele
    2021 finds no unequivocal effect (contradicting), and Boinot 2022 is a
    methodological critique of Mupepele rather than an independent measurement.
    So only one source agrees: no convergence bonus, no HIGH on evidence quality.
    The interval also spans zero (-10% to +45%), which floors evidence_quality at
    LOW -- an unresolved direction is weaker evidence than a narrow mechanistic
    estimate that at least commits to a sign.

    magnitude_confidence stays MODERATE: contested blocks HIGH but is not itself
    grounds for LOW, and that axis is about the size of the effect, not whether
    the relationship is established."""
    site = _fully_satisfying_site()
    result = propagate(
        graph, "alley_cropping", "species_richness", site, n=10_000, rng=np.random.default_rng(0)
    )
    assert result.ci90[0] < 0 < result.ci90[1]
    assert result.magnitude_confidence == Confidence.MODERATE
    assert result.evidence_quality == Confidence.LOW
    assert result.n_sources == 3
    assert result.n_agreeing_sources == 1
    assert result.convergence_factor == 1.0
    assert len(result.caveats) > 0


def test_legume_cover_crop_soc_evidence_quality_and_magnitude_confidence(
    graph: nx.MultiDiGraph,
) -> None:
    """The legume_cover_crop -> soil_organic_carbon edge is backed by two
    independent meta-analyses (Joshi 2023, McClelland 2020) that agree on
    direction but disagree on magnitude (7.3% vs 12%), encoded as contested.
    That is strong evidence with an honestly wide interval: evidence_quality
    should be HIGH (2+ sources, meta-analysis strength), and
    magnitude_confidence should be MODERATE, not LOW -- contested blocks HIGH
    but should not be double-penalised down to LOW."""
    site = _fully_satisfying_site()
    result = propagate(
        graph, "legume_cover_crop", "soil_organic_carbon", site, n=10_000, rng=np.random.default_rng(0)
    )
    assert result.evidence_quality == Confidence.HIGH
    assert result.magnitude_confidence == Confidence.MODERATE
    assert result.n_sources >= 2


def test_non_legume_cover_crop_evidence_quality_no_better_than_legume(
    graph: nx.MultiDiGraph,
) -> None:
    site = _fully_satisfying_site()
    legume = propagate(
        graph, "legume_cover_crop", "soil_organic_carbon", site, n=10_000, rng=np.random.default_rng(0)
    )
    non_legume = propagate(
        graph, "non_legume_cover_crop", "soil_organic_carbon", site, n=10_000, rng=np.random.default_rng(0)
    )
    quality_rank = {Confidence.LOW: 0, Confidence.MODERATE: 1, Confidence.HIGH: 2}
    assert quality_rank[non_legume.evidence_quality] <= quality_rank[legume.evidence_quality]


def test_same_seed_gives_identical_p50(graph: nx.MultiDiGraph) -> None:
    site = _fully_satisfying_site()
    r1 = propagate(
        graph, "legume_cover_crop", "soil_organic_carbon", site, n=5000, rng=np.random.default_rng(42)
    )
    r2 = propagate(
        graph, "legume_cover_crop", "soil_organic_carbon", site, n=5000, rng=np.random.default_rng(42)
    )
    assert r1.p50 == r2.p50


def test_legume_cover_crop_soc_reports_single_path(graph: nx.MultiDiGraph) -> None:
    """The no-till subgroup edge was removed (see edges.py Section A comment): its
    condition can never be gated without a tillage-state node, so it fired
    unconditionally and diluted the mixture. Only the pooled published edge remains."""
    site = _fully_satisfying_site()
    result = propagate(
        graph, "legume_cover_crop", "soil_organic_carbon", site, n=2000, rng=np.random.default_rng(0)
    )
    assert result.paths_found == 1


def test_find_tradeoffs_alley_cropping_low_rainfall(graph: nx.MultiDiGraph) -> None:
    site = SiteState(
        site_id="low_rainfall_400",
        annual_rainfall_mm=Measurement(
            value=400, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )
    tradeoffs = find_tradeoffs(graph, "alley_cropping", site, n=2000, seed=0)
    assert len(tradeoffs) >= 1


def test_limiting_factor_water(graph: nx.MultiDiGraph) -> None:
    site = _deccan_site()
    var, why = limiting_factor(site)
    assert var == "plant_available_water"
    assert why


def test_legume_cover_crop_crop_yield_deep_chain(graph: nx.MultiDiGraph) -> None:
    site = _fully_satisfying_site()
    result = propagate(
        graph, "legume_cover_crop", "crop_yield", site, n=5000, rng=np.random.default_rng(0)
    )
    assert result.paths_found >= 4
    touched = {node for pc in result.paths for node in pc.path}
    assert len(touched) >= 6


def test_rank_interventions_runs(graph: nx.MultiDiGraph) -> None:
    site = _deccan_site()
    ranked = rank_interventions(graph, site, n=1000, seed=0)
    assert len(ranked) == len(graph.nodes) or len(ranked) > 0
    assert ranked[0].score >= ranked[-1].score


# ============================== determinism ==============================
#
# The engine's numbers have to be reproducible across processes, not merely
# within one. An eval harness cannot compare a run against a baseline if the
# system's own figures move between invocations, so these are a prerequisite
# for the eval stage rather than a nicety.
#
# The bug these cover: nx.descendants returns a set, a subgraph built from one
# iterates its nodes in that set's order, and that order set the order
# predecessors were visited inside forward_propagate. Each predecessor draws
# from the rng, so the draw sequence, and therefore every figure, depended on
# PYTHONHASHSEED.


def test_propagate_is_identical_across_calls_with_the_same_seed(graph: nx.MultiDiGraph) -> None:
    site = _deccan_site()
    first = propagate(
        graph, "legume_cover_crop", "crop_yield", site, n=4000, rng=np.random.default_rng(0)
    )
    second = propagate(
        graph, "legume_cover_crop", "crop_yield", site, n=4000, rng=np.random.default_rng(0)
    )
    # Exact equality, not approximate. Same seed and same draw order means
    # the same floats, and anything less would hide the ordering bug.
    assert first.p50 == second.p50
    assert first.ci90 == second.ci90
    assert first.mean == second.mean
    assert first.p_positive == second.p_positive


def test_rank_interventions_is_identical_across_calls_with_the_same_seed(
    graph: nx.MultiDiGraph,
) -> None:
    site = _deccan_site()
    first = rank_interventions(graph, site, n=2000, seed=0)
    second = rank_interventions(graph, site, n=2000, seed=0)

    assert [r.intervention for r in first] == [r.intervention for r in second]
    assert [r.score for r in first] == [r.score for r in second]
    for a, b in zip(first, second):
        assert a.constraint_movement == b.constraint_movement
        assert {t: r.p50 for t, r in a.effects.items()} == {t: r.p50 for t, r in b.effects.items()}


@pytest.mark.slow
def test_demo_output_is_identical_across_python_hash_seeds() -> None:
    """The end-to-end guarantee, across processes.

    PYTHONHASHSEED has to be set before the interpreter starts, so this runs
    the demo in subprocesses rather than in-process.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    outputs = []
    for seed in ("1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [sys.executable, "-m", "src.demo"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1], "output differs between PYTHONHASHSEED 1 and 2"
    assert outputs[0] == outputs[2], "output differs between PYTHONHASHSEED 1 and 12345"
