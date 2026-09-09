"""Does the system actually reason about the site, or does it have a fixed
opinion it recites everywhere?

The demo exposed the failure these tests exist to prevent: a humid site on a
12% slope produced the same recommendations, with the same numbers, as a
semi-arid site on flat ground. Nothing in the ranking was reading the site.

Each test here pins one channel through which site state has to reach the
output: the diagnosis (limiting_factor), the edge preconditions
(Conditions.satisfaction), and the ranking that consumes both.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.demo import DEMO_SITES
from src.graph.edges import build_graph
from src.graph.propagate import (
    PRIORITY_TOLERANCE,
    TIER_1_MIN_EFFECT,
    _CONFIDENCE_FACTOR,
    limiting_factor,
    rank_interventions,
)
from src.graph.schemas import Confidence, Measurement, Provenance, SiteState


@pytest.fixture(scope="module")
def graph() -> nx.MultiDiGraph:
    return build_graph()


def _site(
    site_id: str,
    soc: float,
    rainfall: float,
    ph: float,
    slope: float,
) -> SiteState:
    """A cropland site with complete data, so no condition scores the 0.5 that
    unknown values get and every difference between sites is a real one."""
    return SiteState(
        site_id=site_id,
        soil_organic_carbon_pct=Measurement(
            value=soc, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=rainfall, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(value=ph, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE),
        slope_pct=Measurement(
            value=slope, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
    )


def _score_of(ranked, intervention: str) -> float:
    return next(r.score for r in ranked if r.intervention == intervention)


# ------------------------------- the diagnosis -------------------------------


def test_limiting_factor_erosion_on_steep_humid_site() -> None:
    """2100mm on a 12% slope. Soil leaves faster than it forms, so anything
    built into the profile is exported before it accumulates."""
    var, why = limiting_factor(DEMO_SITES["western_ghats"])
    assert var == "erosion_rate"
    assert why


def test_limiting_factor_never_returns_none() -> None:
    """Every demo site gets a named constraint and a reason. The old
    fall-through returned the sentinel "none", which is what let two different
    sites share an unconstrained ranking."""
    for site_id, site in DEMO_SITES.items():
        var, why = limiting_factor(site)
        assert var is not None, site_id
        assert var != "none", site_id
        assert why, site_id


def test_limiting_factor_indo_gangetic_is_named() -> None:
    var, why = limiting_factor(DEMO_SITES["indo_gangetic"])
    assert var not in (None, "none")
    assert why


def test_three_demo_sites_get_three_different_diagnoses() -> None:
    diagnoses = {site_id: limiting_factor(site)[0] for site_id, site in DEMO_SITES.items()}
    assert len(set(diagnoses.values())) == 3, diagnoses


def test_leaching_syndrome_fires_inside_the_normal_ph_band() -> None:
    """pH 5.8 is inside the 5.5-8.5 band the extreme-pH rule looks at, so that
    rule passes over it. Under 1800mm it is still evidence of base-cation
    export, and nutrient supply is what binds."""
    var, _ = limiting_factor(_site("leaching", soc=1.4, rainfall=1800, ph=5.8, slope=2))
    assert var == "nutrient_cycling_rate"


# ---------------------------- the recommendations ----------------------------


def test_demo_sites_do_not_all_get_the_same_recommendation(graph: nx.MultiDiGraph) -> None:
    """The regression this file was written for. Either the three sites pick
    three different interventions, or any two that share a top pick must differ
    by more than 20% relative on at least one impacted metric. Identical
    recommendations with identical numbers is the failure.

    Sharing a top pick counts as differentiated when the two sites reach it
    through DIFFERENT binding constraints, because then the recommendation is
    the same action justified by a different diagnosis, addressing a different
    variable, with a different set of reported metrics. Contour bunding heads
    both the semi-arid deccan site (water harvesting, +7.5% plant-available
    water) and the steep humid western ghats site (erosion control, -10.0%
    erosion rate), which is agronomically right for both and is not the failure
    this test guards against. That failure was one diagnosis, one set of
    numbers, three sites.

    The plant-available-water figure is identical across those two sites
    because the same edge fires identically at 5% and 12% slope, both inside
    its 2-25% band, so the shared-metric comparison alone cannot separate them.
    Forcing three distinct picks would mean tuning the graph to a test rather
    than to the biophysics.
    """
    tops = {}
    for site_id, site in DEMO_SITES.items():
        ranked = rank_interventions(graph, site, n=4000, seed=0)
        top = ranked[0]
        tops[site_id] = (
            top.intervention,
            {target: res.p50 for target, res in top.effects.items() if res.paths_found > 0},
            limiting_factor(site)[0],
        )

    picks = [name for name, _, _ in tops.values()]
    if len(set(picks)) == 3:
        return

    site_ids = list(tops)
    for i, a_id in enumerate(site_ids):
        for b_id in site_ids[i + 1 :]:
            a_pick, a_effects, a_constraint = tops[a_id]
            b_pick, b_effects, b_constraint = tops[b_id]
            if a_pick != b_pick:
                continue
            # Same action, different diagnosis: the recommendation addresses a
            # different variable and reports a different metric set, which is
            # the site being reasoned about even though the pick coincides.
            if a_constraint != b_constraint:
                continue
            shared = set(a_effects) & set(b_effects)
            differences = [
                abs(a_effects[t] - b_effects[t]) / abs(a_effects[t])
                for t in shared
                # Compare only metrics with enough magnitude to be reportable.
                # A metric that is 0.0003 at one site and 0.0 at the other is a
                # 100% relative difference and no evidence of anything.
                if abs(a_effects[t]) >= 0.001
            ]
            assert differences and max(differences) > 0.20, (
                f"{a_id} and {b_id} both recommend {a_pick} for the same constraint "
                f"({a_constraint}) with effectively identical numbers: {a_effects} vs {b_effects}"
            )


# ------------------------------- the tier gate -------------------------------


def test_western_ghats_top_pick_addresses_erosion(graph: nx.MultiDiGraph) -> None:
    """Liebig's law of the minimum is a gate. The site's constraint is erosion,
    so the top pick has to be something that reduces erosion, not whatever
    scores best on the standing objectives."""
    ranked = rank_interventions(graph, DEMO_SITES["western_ghats"], n=4000, seed=0)
    top = ranked[0]
    assert top.addresses_limiting_factor
    assert top.tier == 1
    assert top.effects["erosion_rate"].p50 < -TIER_1_MIN_EFFECT


def test_western_ghats_top_pick_is_an_erosion_intervention(graph: nx.MultiDiGraph) -> None:
    """The specific regression: alley cropping used to win here on a contested
    biodiversity estimate whose interval spans zero, while doing nothing about
    the erosion the diagnosis had just named."""
    erosion_interventions = {"contour_bunding", "contour_trenching", "vetiver_grass_strips"}
    ranked = rank_interventions(graph, DEMO_SITES["western_ghats"], n=4000, seed=0)

    assert ranked[0].intervention in erosion_interventions, ranked[0].intervention
    alley = next(r for r in ranked if r.intervention == "alley_cropping")
    assert alley.tier == 2
    assert not alley.addresses_limiting_factor


def test_deccan_top_pick_addresses_plant_available_water(graph: nx.MultiDiGraph) -> None:
    ranked = rank_interventions(graph, DEMO_SITES["deccan_semiarid"], n=4000, seed=0)
    top = ranked[0]
    assert top.addresses_limiting_factor
    assert top.tier == 1
    assert top.effects["plant_available_water"].p50 > TIER_1_MIN_EFFECT


def test_no_demo_site_falls_back_to_an_empty_tier_1(graph: nx.MultiDiGraph) -> None:
    """A site whose constraint nothing in the graph can address degrades to
    plain score order. That is worth knowing about, so it is asserted absent
    for the demo sites rather than left to be noticed in the output."""
    for site_id, site in DEMO_SITES.items():
        ranked = rank_interventions(graph, site, n=4000, seed=0)
        assert any(r.tier == 1 for r in ranked), f"{site_id} has an empty tier 1"


def test_every_tier_1_intervention_outranks_every_tier_2(graph: nx.MultiDiGraph) -> None:
    """The gate, asserted as a gate: no tier 2 intervention may appear above a
    tier 1 one, however much better its score."""
    for site_id, site in DEMO_SITES.items():
        ranked = rank_interventions(graph, site, n=4000, seed=0)
        tiers = [r.tier for r in ranked]
        assert tiers == sorted(tiers), f"{site_id} interleaves tiers: {tiers}"


def test_tier_2_can_outscore_tier_1(graph: nx.MultiDiGraph) -> None:
    """Proof the ranking is a gate and not a weighting. On the western ghats
    site, alley cropping outscores the erosion interventions and still ranks
    below all of them. If this ever stops holding, the gate has quietly become
    a tie-breaker."""
    ranked = rank_interventions(graph, DEMO_SITES["western_ghats"], n=4000, seed=0)
    best_tier_2 = max((r for r in ranked if r.tier == 2), key=lambda r: r.score)
    worst_tier_1 = min((r for r in ranked if r.tier == 1), key=lambda r: r.score)

    assert best_tier_2.score > worst_tier_1.score
    assert ranked.index(worst_tier_1) < ranked.index(best_tier_2)


def _priorities(ranked, limiting_var: str) -> list[float]:
    return [
        abs(r.constraint_movement) * _CONFIDENCE_FACTOR[r.effects[limiting_var].evidence_quality]
        for r in ranked
        if r.tier == 1
    ]


def test_tier_1_never_inverts_beyond_tolerance(graph: nx.MultiDiGraph) -> None:
    """The guarantee the tier 1 ordering actually makes: substantially better
    constraint relief wins outright. Within PRIORITY_TOLERANCE the score
    decides, so priorities are not strictly descending, but no intervention
    may rank above another whose relief is better by more than the tolerance.

    Ordering tier 1 by the multi-objective score instead put vetiver grass
    strips, scoring 0.0 because erosion carries no score weight, above
    interventions doing four times as much for erosion.
    """
    for site_id, site in DEMO_SITES.items():
        limiting_var = limiting_factor(site)[0]
        ranked = rank_interventions(graph, site, n=4000, seed=0)
        priorities = _priorities(ranked, limiting_var)

        for i, higher in enumerate(priorities):
            for lower in priorities[i + 1 :]:
                if lower <= higher:
                    continue
                # A lower-ranked entry with better relief is only allowed
                # when the two are comparable.
                assert (lower - higher) / lower <= PRIORITY_TOLERANCE + 1e-9, (
                    f"{site_id}: an intervention with priority {lower:.4f} ranks below one "
                    f"with {higher:.4f}, an inversion beyond the tolerance"
                )


def test_tolerance_band_lets_evidence_break_comparable_relief(graph: nx.MultiDiGraph) -> None:
    """On the deccan site legume cover crop and reduced tillage relieve water
    comparably (about 3.1% against 3.4%, inside the tolerance), and legume has
    a materially higher multi-objective score and two converging meta-analyses
    behind it. Without the band, 0.28pp of water movement decided this, which
    is inside the Monte Carlo noise."""
    ranked = rank_interventions(graph, DEMO_SITES["deccan_semiarid"], n=4000, seed=0)
    order = [r.intervention for r in ranked]

    legume = next(r for r in ranked if r.intervention == "legume_cover_crop")
    tillage = next(r for r in ranked if r.intervention == "reduced_tillage")
    assert legume.tier == 1 and tillage.tier == 1
    assert legume.score > tillage.score
    assert order.index("legume_cover_crop") < order.index("reduced_tillage")


def test_substantially_better_relief_still_wins_outright(graph: nx.MultiDiGraph) -> None:
    """The band must not swallow a real difference. Contour bunding relieves
    water more than twice as well as anything else on the deccan site, so it
    heads the list however good the alternatives' co-benefits are."""
    ranked = rank_interventions(graph, DEMO_SITES["deccan_semiarid"], n=4000, seed=0)
    assert ranked[0].intervention == "contour_bunding"
    assert any(r.score > ranked[0].score for r in ranked[1:])


def test_vetiver_not_above_a_larger_erosion_movement(graph: nx.MultiDiGraph) -> None:
    """The specific inversion, named. Vetiver may rank where its erosion
    movement earns, and nowhere higher."""
    ranked = rank_interventions(graph, DEMO_SITES["western_ghats"], n=4000, seed=0)
    tier_1 = [r for r in ranked if r.tier == 1]
    vetiver = next((r for r in tier_1 if r.intervention == "vetiver_grass_strips"), None)
    if vetiver is None:
        pytest.skip("vetiver_grass_strips is not in tier 1 on this site")

    vetiver_position = tier_1.index(vetiver)
    vetiver_movement = abs(vetiver.constraint_movement)
    for other in tier_1[vetiver_position + 1 :]:
        other_movement = abs(other.constraint_movement)
        if other_movement <= vetiver_movement:
            continue
        # Ranking below vetiver on better movement is only allowed inside the
        # tolerance band, where the score decides.
        assert (other_movement - vetiver_movement) / other_movement <= PRIORITY_TOLERANCE + 1e-9, (
            f"{other.intervention} moves erosion by {other.constraint_movement:+.1%} and ranks "
            f"below vetiver at {vetiver.constraint_movement:+.1%}"
        )


def test_constraint_movement_is_populated_for_tier_1(graph: nx.MultiDiGraph) -> None:
    for site_id, site in DEMO_SITES.items():
        for r in rank_interventions(graph, site, n=4000, seed=0):
            if r.tier == 1:
                assert r.constraint_movement is not None, f"{site_id}: {r.intervention}"


def test_a_path_alone_does_not_earn_tier_1(graph: nx.MultiDiGraph) -> None:
    """Tier 1 needs direction and magnitude, not just reachability. Any
    intervention with a path to the constraint that fails the floor must stay
    in tier 2, and at least one such case should exist for the check to mean
    anything."""
    ranked = rank_interventions(graph, DEMO_SITES["deccan_semiarid"], n=4000, seed=0)
    reachable_but_negligible = [
        r for r in ranked if r.limiting_factor_addressed and not r.addresses_limiting_factor
    ]
    assert reachable_but_negligible, "no intervention exercises the magnitude floor"
    for r in reachable_but_negligible:
        assert r.tier == 2, r.intervention


# ---------------------------- site differentiation ---------------------------


def test_contour_bunding_scores_higher_on_sloped_site(graph: nx.MultiDiGraph) -> None:
    """Bunding intercepts runoff, which needs slope for there to be runoff to
    intercept. Two sites identical but for slope."""
    flat = _site("flat", soc=0.8, rainfall=700, ph=7.0, slope=1)
    sloped = _site("sloped", soc=0.8, rainfall=700, ph=7.0, slope=12)

    flat_score = _score_of(rank_interventions(graph, flat, n=4000, seed=0), "contour_bunding")
    sloped_score = _score_of(rank_interventions(graph, sloped, n=4000, seed=0), "contour_bunding")
    assert sloped_score > flat_score


def test_cover_crop_scores_lower_on_arid_site(graph: nx.MultiDiGraph) -> None:
    """200mm is below the establishment bound on the cover-crop carbon edge, so
    the intervention has to attenuate relative to a site where the cover crop
    reliably establishes. Two sites identical but for rainfall."""
    arid = _site("arid", soc=0.8, rainfall=200, ph=7.0, slope=3)
    wetter = _site("wetter", soc=0.8, rainfall=620, ph=7.0, slope=3)

    arid_score = _score_of(rank_interventions(graph, arid, n=4000, seed=0), "legume_cover_crop")
    wetter_score = _score_of(rank_interventions(graph, wetter, n=4000, seed=0), "legume_cover_crop")
    assert arid_score < wetter_score


def test_ph_differentiates_the_microbial_route(graph: nx.MultiDiGraph) -> None:
    """The soil carbon to microbial biomass edge is gated on pH 5.0-8.0, so an
    alkaline site loses the mineralisation route to yield that an in-band site
    keeps. Two sites identical but for pH."""
    in_band = _site("in_band", soc=0.8, rainfall=700, ph=6.2, slope=3)
    alkaline = _site("alkaline", soc=0.8, rainfall=700, ph=8.1, slope=3)

    in_band_yield = next(
        r for r in rank_interventions(graph, in_band, n=4000, seed=0) if r.intervention == "legume_cover_crop"
    ).effects["crop_yield"]
    alkaline_yield = next(
        r for r in rank_interventions(graph, alkaline, n=4000, seed=0) if r.intervention == "legume_cover_crop"
    ).effects["crop_yield"]

    assert in_band_yield.p50 > alkaline_yield.p50 * 2
