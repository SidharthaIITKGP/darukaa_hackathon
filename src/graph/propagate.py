"""Monte Carlo propagation of intervention effects through the causal graph.

Magnitude comes from node-wise forward propagation (forward_propagate):
each node's proportional change is computed exactly once, as a noisy-OR over
its incoming edges. This is the only place noisy-OR is applied to combine
distinct mechanisms, because that is the only place the independence
assumption behind noisy-OR is defensible.

Path enumeration (_enumerate_edge_paths) still exists, but only to explain
results: which mechanisms exist between an intervention and a target, how
strong the evidence is, what preconditions gate them. Path magnitudes are
never combined across paths to produce a headline number, because two paths
sharing a trunk would reinject that trunk's effect once per path -- exactly
the double-counting noisy-OR is supposed to prevent, and does not, when
applied across paths that are not independent.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import yaml
from pydantic import BaseModel

from src.graph.nodes import INTERVENTIONS, STATE_VARIABLES
from src.graph.schemas import CausalEdge, Confidence, EvidenceStrength, SiteState

_SOURCES_YAML_PATH = Path(__file__).resolve().parents[2] / "sources.yaml"

_STRENGTH_RANK = {
    EvidenceStrength.META_ANALYSIS: 0,
    EvidenceStrength.MULTI_SITE: 1,
    EvidenceStrength.SINGLE_SITE: 2,
    EvidenceStrength.MECHANISTIC: 3,
    EvidenceStrength.EXPERT: 4,
}

# Weights for mixing parallel (source, target) edges: alternative estimates
# of the SAME relationship, not distinct mechanisms. Higher-strength
# estimates are more likely to be drawn per sample, but every estimate stays
# reachable, so the mixture widens the distribution to span disagreement
# instead of collapsing it to one number.
_STRENGTH_WEIGHT = {
    EvidenceStrength.META_ANALYSIS: 3.0,
    EvidenceStrength.MULTI_SITE: 2.0,
    EvidenceStrength.SINGLE_SITE: 1.5,
    EvidenceStrength.MECHANISTIC: 1.0,
    EvidenceStrength.EXPERT: 0.5,
}

_CONFIDENCE_FACTOR = {
    Confidence.HIGH: 1.0,
    Confidence.MODERATE: 0.7,
    Confidence.LOW: 0.4,
}

# Modelling assumption, not an empirical value: signal dissipates crossing
# each edge rather than transmitting perfectly. Applied once per edge inside
# forward_propagate, so depth attenuation emerges naturally from the chain
# (a 3-edge chain carries three factors of TRANSMISSION, a 1-edge link one)
# rather than from an explicit hop-index exponent. NOT applied to an edge
# whose source is the intervention itself: a published edge out of the
# intervention already IS the measured end-to-end effect of adopting it, so
# attenuating it would silently contradict its citation. Attenuation models
# loss along a chain after that point, not loss at the point of application.
# Reported in the system's methodology notes alongside every propagation
# result via PropagationResult.transmission_factor.
TRANSMISSION = 0.75

# Reference upstream proportional change against which a published or
# estimated edge effect is assumed to have been measured. A modelling
# assumption, not an empirical value: an edge fires at (up to) full strength
# only when its upstream node has moved by at least this much, and scales
# down for smaller upstream moves. Clipped at 1.5x to prevent runaway
# transmission on large upstream changes. Not applied to an edge whose
# source is the intervention itself -- see TRANSMISSION's comment above for
# why that special case exists. Reported alongside every propagation result
# via PropagationResult.delta_ref.
DELTA_REF = 0.10

# Reward for evidential convergence: when 2+ independent sources AGREE on an
# edge's direction, that agreement is a genuine quality signal the score
# would otherwise ignore. A modelling assumption, kept deliberately small so
# it nudges rather than dominates ranking -- convergence tops out at 1.2x
# once a dominant path cites 3+ agreeing source_ids. Counted from primary and
# corroborating refs only: a contradicting result is not corroboration, and a
# methodological critique of another study is not an independent measurement
# of the relationship, so neither can raise this factor. Reported alongside
# every propagation result via PropagationResult.convergence_factor.
CONVERGENCE_STEP = 0.1
CONVERGENCE_SOURCE_CAP = 2

# How hard a tradeoff counts against an intervention's score. Modelling
# assumptions. A tradeoff landing on the site's binding constraint (or on a
# variable upstream of it) is weighted 4x the ordinary one, because it does
# not merely offset the gains: it holds shut the gate everything downstream
# has to pass through. The reward for relieving that constraint is not the
# mirror of this penalty and is deliberately not on this axis at all: it is
# the tier partition in rank_interventions, because Liebig's law of the
# minimum is a gate and not a weighting.
TRADEOFF_PENALTY_WEIGHT = 0.5
LIMITING_FACTOR_PENALTY_WEIGHT = 2.0

_AGREEING_ROLES = ("primary", "corroborating")

_QUALITY_RANK = {Confidence.LOW: 0, Confidence.MODERATE: 1, Confidence.HIGH: 2}


def _improvement_direction(variable: str, site: SiteState) -> float:
    """Which way `variable` has to move for this site to be better off.

    Resolved per site rather than per variable because soil_ph has no fixed
    answer: an acid soil needs pH raised and an alkaline one needs it
    lowered, so a single sign in CONSTRAINT_DIRECTION would reward liming a
    pH 8.6 soil. Where the site's own pH is unknown the direction is
    genuinely undetermined, and 0.0 is returned so that nothing can qualify
    as addressing it on a guess.
    """
    if variable in CONSTRAINT_DIRECTION:
        return CONSTRAINT_DIRECTION[variable]
    if variable == "soil_ph":
        if site.ph is None or site.ph.value is None:
            return 0.0
        return 1.0 if site.ph.value < 7.0 else -1.0
    return 1.0


def _convergence_factor(n_agreeing: int) -> float:
    return 1.0 + CONVERGENCE_STEP * min(max(n_agreeing - 1, 0), CONVERGENCE_SOURCE_CAP)


def _cap_quality(quality: Confidence, ceiling: Confidence) -> Confidence:
    return quality if _QUALITY_RANK[quality] <= _QUALITY_RANK[ceiling] else ceiling

DEFAULT_OBJECTIVES: dict[str, float] = {
    "species_richness": 0.3,
    "soil_organic_carbon": 0.25,
    "plant_available_water": 0.2,
    "crop_yield": 0.15,
    "pollinator_abundance": 0.1,
}

# Which way a variable has to move for the site to be better off. Needed
# because "addresses the binding constraint" is a claim about direction, not
# just about a path existing: a path from an intervention to erosion_rate
# that RAISES erosion is not addressing an erosion constraint, it is making
# it worse.
#
# Every variable in DEFAULT_OBJECTIVES is higher-is-better and so has
# direction +1 implicitly. soil_ph is absent because its direction is a
# property of the site rather than of the variable: raising pH helps an acid
# soil and harms an alkaline one. _improvement_direction resolves that from
# the site's own pH instead of guessing a fixed sign here.
CONSTRAINT_DIRECTION: dict[str, float] = {
    "erosion_rate": -1.0,
    "nutrient_cycling_rate": 1.0,
    "habitat_connectivity": 1.0,
}

# Smallest movement on the binding constraint that counts as addressing it.
# A modelling assumption. Below this an intervention has a path to the
# constraint on paper but does nothing about it in practice, and letting it
# into tier 1 would promote it above interventions that genuinely help.
TIER_1_MIN_EFFECT = 0.01

# How close two tier 1 constraint-relief priorities have to be before they
# count as comparable, as a fraction of the higher one. A modelling
# assumption, and specifically an admission about precision: these p50s come
# from a 4000-sample Monte Carlo, so a gap of a fraction of a percentage
# point is sampling noise rather than a real difference in how much an
# intervention relieves the constraint. Treating such a gap as decisive was
# false precision that let a mechanistic-only intervention outrank one with
# two converging meta-analyses behind it on 0.28pp of water movement.
#
# Within a tolerance group the multi-objective score decides, so evidence
# quality and co-benefits break comparable relief. Outside it, substantially
# better relief still wins outright.
PRIORITY_TOLERANCE = 0.10


def _source_scopes() -> dict[str, str]:
    data = yaml.safe_load(_SOURCES_YAML_PATH.read_text())
    return {source_id: (info.get("scope") or "") for source_id, info in data.get("sources", {}).items()}


_SOURCE_SCOPES = _source_scopes()


class PathContribution(BaseModel):
    """One mechanism's isolated contribution, computed independently of every
    other path. Useful for saying which mechanism dominates the story behind
    a result. These p50s do NOT sum, average, or noisy-OR to the headline
    PropagationResult.p50 by design: that number comes from node-wise
    forward_propagate, which shares a common trunk between paths instead of
    re-injecting it once per path.
    """

    path: list[str]
    p50: float
    ci90: tuple[float, float]
    min_satisfaction: float
    weakest_strength: str
    lag_years: tuple[float, float]
    contested: bool


class PropagationResult(BaseModel):
    intervention: str
    target: str
    paths_found: int
    p50: float
    ci90: tuple[float, float]
    mean: float
    p_positive: float
    p_exceeds: dict[str, float]
    lag_years: tuple[float, float]
    # Explanatory only: which mechanisms connect intervention to target, and
    # how strong/contested/precondition-gated each one is. Do not combine
    # these across paths to get a magnitude; p50/ci90/mean above already come
    # from node-wise forward propagation, not from these paths.
    paths: list[PathContribution]
    # Two independent axes, deliberately not collapsed into one field:
    #   evidence_quality     -- how well-supported the relationship is
    #                           (strength of evidence, breadth of sources).
    #                           Disagreement on magnitude between sources
    #                           that agree on direction does NOT lower this;
    #                           two studies converging on "yes, positive" is
    #                           better evidence than one study asserting a
    #                           number, even if they disagree on the number.
    #   magnitude_confidence -- how precisely the SIZE of the effect is
    #                           known: contested disagreement on magnitude,
    #                           unmet preconditions, and mechanistic-only
    #                           evidence all lower this. Already reflected in
    #                           the width of ci90, so this is the field a
    #                           score should discount by at most once.
    # Scoring on evidence_quality avoids double-penalising a contested-but-
    # well-evidenced edge: its wider CI already encodes the magnitude
    # disagreement, so multiplying the score by a magnitude-confidence
    # factor as well would count that same disagreement twice.
    evidence_quality: Confidence
    magnitude_confidence: Confidence
    caveats: list[str]
    transmission_factor: float
    delta_ref: float
    n_sources: int
    n_agreeing_sources: int
    convergence_factor: float


class Tradeoff(BaseModel):
    intervention: str
    positive_target: str
    negative_target: str
    positive_effect: float
    negative_effect: float
    conditions_note: str
    evidence: list[str]


class RankedIntervention(BaseModel):
    intervention: str
    score: float
    effects: dict[str, PropagationResult]
    tradeoffs: list[Tradeoff]
    n_variables_touched: int
    # Two related fields, deliberately not collapsed, because the gap between
    # them is what makes a tier assignment auditable:
    #   limiting_factor_addressed  -- a causal path from this intervention
    #                                 reaches the binding constraint. Says
    #                                 nothing about direction or size, so a
    #                                 path that moves the constraint the
    #                                 wrong way, or by 0.1%, still sets it.
    #   addresses_limiting_factor  -- that path also moves the constraint in
    #                                 the direction the site needs, by at
    #                                 least TIER_1_MIN_EFFECT. This is the
    #                                 one that gates tier 1.
    # An intervention with the first True and the second False is the
    # interesting case: the mechanism exists on paper and does nothing here.
    limiting_factor_addressed: bool
    addresses_limiting_factor: bool
    # Liebig's law of the minimum as a gate, not a weighting. Tier 1
    # addresses the site's binding constraint; tier 2 does not. Every tier 1
    # intervention ranks above every tier 2 intervention regardless of score,
    # because a gain downstream of a constraint that is still shut cannot be
    # realised.
    tier: Literal[1, 2]
    # Propagated p50 on the site's binding constraint, signed as the variable
    # moves (negative for a reduction in erosion). None when no path reaches
    # the constraint. This is what orders tier 1, so it is reported rather
    # than left implicit in the ordering.
    constraint_movement: float | None
    # True when a tradeoff whose preconditions this site meets lands on the
    # site's binding constraint, or on a variable upstream of it. The
    # intervention may still be sound elsewhere and later here, which is why
    # this is reported as sequencing guidance rather than folded silently
    # into the score.
    conflicts_with_limiting_factor: bool
    sequencing_note: str | None = None


def _enumerate_edge_paths(
    graph: nx.MultiDiGraph, intervention: str, target: str, max_hops: int
) -> list[list[tuple[str, str, int]]]:
    """Expand each simple node-path into all its parallel-edge combinations."""
    edge_paths: list[list[tuple[str, str, int]]] = []
    if intervention not in graph or target not in graph:
        return edge_paths
    # nx.all_simple_paths on a MultiDiGraph yields the same node sequence once
    # per parallel edge, so dedupe node-paths before expanding edge-key combos
    # ourselves, or parallel edges get double-counted.
    seen_node_paths = sorted({tuple(p) for p in nx.all_simple_paths(graph, intervention, target, cutoff=max_hops)})
    for node_path in seen_node_paths:
        hop_keys = [list(graph[u][v].keys()) for u, v in zip(node_path, node_path[1:])]
        for combo in itertools.product(*hop_keys):
            edge_paths.append(list(zip(node_path[:-1], node_path[1:], combo)))
    return edge_paths


def _weakest_strength(edges: list[CausalEdge]) -> EvidenceStrength:
    return max(edges, key=lambda e: _STRENGTH_RANK[e.strength]).strength


def _path_caveats(
    node_path: list[str], edges: list[CausalEdge], contribution: PathContribution, site: SiteState
) -> list[str]:
    caveats: list[str] = []
    label = " -> ".join(node_path)

    if contribution.contested:
        for edge in edges:
            if edge.contested and edge.contested_note:
                caveats.append(edge.contested_note)

    if contribution.weakest_strength in (EvidenceStrength.MECHANISTIC.value, EvidenceStrength.EXPERT.value):
        caveats.append(f"Path {label} rests on {contribution.weakest_strength} evidence only.")

    if contribution.min_satisfaction < 1.0:
        caveats.append(
            f"Path {label} preconditions are only {contribution.min_satisfaction:.0%} met at this site."
        )

    zone = site.climate_zone()
    if zone in ("semi_arid", "arid"):
        for edge in edges:
            for ref in edge.evidence:
                scope = _SOURCE_SCOPES.get(ref.source_id, "")
                if "temperate" in scope.lower():
                    caveats.append(
                        f"{ref.source_id} evidence scope is '{scope}'; site is {zone}, "
                        "outside that scope."
                    )

    return caveats


def _break_cycles(sub: nx.MultiDiGraph) -> list[str]:
    """Break cycles in place by dropping the weakest edge in each one.

    Weakest = lowest evidence strength first, widest interval as a
    tiebreaker among edges of equal strength. Returns a log message per
    dropped edge; callers must surface these rather than swallow them.
    """
    dropped: list[str] = []
    while True:
        try:
            cycle_edges = nx.find_cycle(sub, orientation="original")
        except nx.NetworkXNoCycle:
            break
        candidates = [(u, v, key, sub[u][v][key]["edge"]) for u, v, key, _direction in cycle_edges]
        u, v, key, edge = max(
            candidates,
            key=lambda c: (_STRENGTH_RANK[c[3].strength], c[3].effect.ci_high - c[3].effect.ci_low),
        )
        sub.remove_edge(u, v, key)
        cycle_nodes = " -> ".join(c[0] for c in candidates) + f" -> {candidates[0][0]}"
        dropped.append(
            f"Cycle detected ({cycle_nodes}); dropped {u} -> {v} (key={key}, "
            f"strength={edge.strength.value}) as the weakest edge to break it."
        )
    return dropped


def forward_propagate(
    graph: nx.MultiDiGraph,
    intervention: str,
    site: SiteState,
    n: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Node-wise forward propagation of an intervention's effect.

    Each node's proportional change is computed exactly once. Aggregation at
    a node is two-level:

      1. Parallel edges between the SAME (source, target) pair are
         ALTERNATIVE ESTIMATES of one relationship (e.g. Joshi's pooled
         cover-crop figure vs. its no-till subgroup), not distinct
         mechanisms. They are combined as a weighted MIXTURE: per Monte
         Carlo sample, one estimate is drawn (weighted by evidence
         strength) and used whole. This widens the distribution to span
         the disagreement without inflating the central estimate -- the
         same principle as encoding disagreement rather than reconciling
         it. A mixture's median sits between its inputs' medians; a
         noisy-OR or a sum does not.
      2. Distinct (source, target) pairs -- genuinely different mechanisms
         feeding the same node -- are combined with noisy-OR. This is
         where the independence assumption behind noisy-OR is defensible.

    A shared trunk between two downstream variables is traversed once, not
    once per path -- this is the fix for path-based noisy-OR
    double-counting shared mechanisms.
    """
    if rng is None:
        rng = np.random.default_rng()

    if intervention not in graph:
        return {}

    reachable = {intervention} | nx.descendants(graph, intervention)
    sub = nx.MultiDiGraph(graph.subgraph(reachable))

    if not nx.is_directed_acyclic_graph(sub):
        for message in _break_cycles(sub):
            warnings.warn(message, stacklevel=2)

    order = list(nx.topological_sort(sub))

    # An intervention is a binary presence, not a "+100% change in a state
    # variable": the (1 + delta[u]) modulation below has no meaning for it,
    # so its multiplier must be 1.0. Activation is carried by the edge
    # existing at all, not by a fictitious upstream proportional change.
    delta: dict[str, np.ndarray] = {intervention: np.zeros(n)}
    for v in order:
        if v == intervention:
            continue
        mechanism_contributions: list[np.ndarray] = []
        for u in sub.predecessors(v):
            if u not in delta:
                continue
            keys = list(sub[u][v].keys())
            edges = [sub[u][v][key]["edge"] for key in keys]

            estimates: list[np.ndarray] = []
            weights: list[float] = []
            for edge in edges:
                raw = edge.effect.sample(n, rng)
                sat = edge.conditions.satisfaction(site)
                local = raw * sat
                # An edge fires at a fraction of its published/estimated
                # strength proportional to how much its upstream node has
                # actually moved, relative to DELTA_REF -- the upstream
                # change the edge's effect is assumed to have been measured
                # against. A 2% upstream move should not transmit a
                # mechanism at near-full published strength; a 10%+ move
                # does (clipped at 1.5x to bound runaway on large upstream
                # changes). TRANSMISSION models loss along a chain, so it is
                # skipped at the point of application -- an edge whose
                # source is the intervention itself already IS the measured
                # end-to-end effect, and delta[intervention] = 0 so the
                # scale factor has no meaning there either.
                if u == intervention:
                    transmitted = local
                else:
                    scale = np.clip(delta[u] / DELTA_REF, 0.0, 1.5)
                    transmitted = local * scale * TRANSMISSION
                estimates.append(-transmitted if edge.sign == "-" else transmitted)
                weights.append(_STRENGTH_WEIGHT[edge.strength])

            if len(estimates) == 1:
                mixture = estimates[0]
            else:
                # Alternative estimates of the same (u, v) relationship:
                # mix, don't noisy-OR. Per sample, draw one estimate
                # weighted by evidence strength rather than combining all of
                # them, which would treat "two measurements of one effect"
                # as "two effects happening at once".
                probs = np.asarray(weights) / sum(weights)
                choice = rng.choice(len(estimates), size=n, p=probs)
                stacked = np.stack(estimates, axis=0)
                mixture = stacked[choice, np.arange(n)]

            mechanism_contributions.append(mixture)

        if not mechanism_contributions:
            delta[v] = np.zeros(n)
            continue

        # Noisy-OR across distinct (source, target) mechanisms. This is
        # where the independence assumption is defensible: distinct
        # mechanisms feeding one variable are partly independent. Applying
        # it across whole paths, or across alternative estimates of the
        # same edge, was the bug -- neither of those is independent.
        gain = np.ones(n)
        for c in mechanism_contributions:
            gain *= 1.0 - c
        delta[v] = 1.0 - gain

    return delta


def propagate(
    graph: nx.MultiDiGraph,
    intervention: str,
    target: str,
    site: SiteState,
    n: int = 10_000,
    max_hops: int = 9,
    rng: np.random.Generator | None = None,
) -> PropagationResult:
    if rng is None:
        rng = np.random.default_rng()

    edge_paths = _enumerate_edge_paths(graph, intervention, target, max_hops)
    deltas = forward_propagate(graph, intervention, site, n=n, rng=rng)

    if not edge_paths or target not in deltas:
        return PropagationResult(
            intervention=intervention,
            target=target,
            paths_found=0,
            p50=0.0,
            ci90=(0.0, 0.0),
            mean=0.0,
            p_positive=0.0,
            p_exceeds={"0.05": 0.0, "0.10": 0.0, "0.20": 0.0},
            lag_years=(0.0, 0.0),
            paths=[],
            evidence_quality=Confidence.LOW,
            magnitude_confidence=Confidence.LOW,
            caveats=["No causal path found from intervention to target within max_hops."],
            transmission_factor=TRANSMISSION,
            delta_ref=DELTA_REF,
            n_sources=0,
            n_agreeing_sources=0,
            convergence_factor=1.0,
        )

    path_contributions: list[PathContribution] = []
    path_edges: list[list[CausalEdge]] = []
    caveats: list[str] = []

    for edge_path in edge_paths:
        node_path = [edge_path[0][0]] + [v for _, v, _ in edge_path]
        edges = [graph[u][v][key]["edge"] for u, v, key in edge_path]

        # Isolated, single-path composition: explanatory only (see
        # PathContribution docstring). TRANSMISSION applies once per edge
        # after the first, so a longer path naturally carries more
        # attenuation without an explicit hop-index term. The first edge is
        # never attenuated when it starts at the intervention: a published
        # edge out of the intervention already IS the measured end-to-end
        # effect, not a signal that has travelled and dissipated.
        gain = np.ones(n)
        lag_low = 0.0
        lag_high = 0.0
        min_satisfaction = 1.0
        for edge in edges:
            raw = edge.effect.sample(n, rng)
            sat = edge.conditions.satisfaction(site)
            min_satisfaction = min(min_satisfaction, sat)
            factor = 1.0 if edge.source == intervention else TRANSMISSION
            eff = raw * sat * factor
            gain *= (1.0 - eff) if edge.sign == "-" else (1.0 + eff)
            lag_low += edge.lag_years[0]
            lag_high += edge.lag_years[1]

        contribution = gain - 1.0

        weakest = _weakest_strength(edges)
        contested = any(edge.contested for edge in edges)
        pc = PathContribution(
            path=node_path,
            p50=float(np.median(contribution)),
            ci90=(float(np.percentile(contribution, 5)), float(np.percentile(contribution, 95))),
            min_satisfaction=min_satisfaction,
            weakest_strength=weakest.value,
            lag_years=(lag_low, lag_high),
            contested=contested,
        )
        path_contributions.append(pc)
        path_edges.append(edges)
        caveats.extend(_path_caveats(node_path, edges, pc, site))

    combined = deltas[target]
    p50 = float(np.median(combined))
    mean = float(np.mean(combined))
    ci90 = (float(np.percentile(combined, 5)), float(np.percentile(combined, 95)))
    p_positive = float(np.mean(combined > 0))

    direction = 1.0 if p50 >= 0 else -1.0
    p_exceeds = {}
    for label in ("0.05", "0.10", "0.20"):
        threshold = float(label)
        if direction >= 0:
            p_exceeds[label] = float(np.mean(combined > threshold))
        else:
            p_exceeds[label] = float(np.mean(combined < -threshold))

    lag_years = (
        min(pc.lag_years[0] for pc in path_contributions),
        max(pc.lag_years[1] for pc in path_contributions),
    )

    min_satisfaction_overall = min(pc.min_satisfaction for pc in path_contributions)
    any_contested = any(pc.contested for pc in path_contributions)
    all_meta_analysis = all(
        pc.weakest_strength == EvidenceStrength.META_ANALYSIS.value for pc in path_contributions
    )
    any_weak = any(
        pc.weakest_strength in (EvidenceStrength.MECHANISTIC.value, EvidenceStrength.EXPERT.value)
        for pc in path_contributions
    )

    # magnitude_confidence: how precisely the SIZE of the effect is known.
    # Contested disagreement on magnitude blocks HIGH (we cannot be fully
    # confident in a number that sources disagree on), but demotes only to
    # MODERATE, not LOW: LOW is reserved for actually weak evidence
    # (mechanistic/expert-only) or badly unmet preconditions. Forcing
    # contested straight to LOW would apply the same disagreement penalty
    # twice -- once via the wider CI (convention: encode disagreement,
    # don't reconcile it) and again via this factor.
    if all_meta_analysis and min_satisfaction_overall >= 0.75 and not any_contested:
        magnitude_confidence = Confidence.HIGH
    elif any_weak or min_satisfaction_overall < 0.5:
        magnitude_confidence = Confidence.LOW
    else:
        magnitude_confidence = Confidence.MODERATE

    combined_sorted = sorted(zip(path_contributions, path_edges), key=lambda pe: abs(pe[0].p50), reverse=True)
    path_contributions = [pc for pc, _ in combined_sorted]
    dominant_edges = combined_sorted[0][1] if combined_sorted else []
    dominant_refs = [ref for edge in dominant_edges for ref in edge.evidence]
    n_sources = len({ref.source_id for ref in dominant_refs})
    n_agreeing_sources = len({ref.source_id for ref in dominant_refs if ref.role in _AGREEING_ROLES})
    any_contradicting = any(ref.role == "contradicting" for ref in dominant_refs)
    any_direction_unresolved = any(edge.effect.spans_zero() for edge in dominant_edges)

    # evidence_quality: how well-supported the relationship is, from evidence
    # strength and AGREEING source count on the dominant path ONLY. Contested
    # (disagreement about magnitude) does NOT lower this: two independent
    # meta-analyses agreeing on direction while disagreeing on size is BETTER
    # evidence than one source asserting a single number, not worse. That
    # disagreement already lives in ci90 and in magnitude_confidence above;
    # penalising it here as well would double-count the same fact.
    dominant_weakest = _weakest_strength(dominant_edges) if dominant_edges else EvidenceStrength.EXPERT
    if dominant_weakest == EvidenceStrength.META_ANALYSIS and n_agreeing_sources >= 2:
        evidence_quality = Confidence.HIGH
    elif dominant_weakest in (EvidenceStrength.MECHANISTIC, EvidenceStrength.EXPERT):
        evidence_quality = Confidence.LOW
    else:
        evidence_quality = Confidence.MODERATE

    # Disagreement about magnitude is not the same as disagreement about
    # whether the effect exists, and the two cap evidence_quality
    # differently. A source finding no effect keeps the relationship out of
    # HIGH however many other sources back it. An interval spanning zero is
    # weaker still: the direction itself is unresolved, which is worse
    # evidence than a narrow mechanistic estimate that at least commits to a
    # sign, so it floors the axis at LOW.
    if any_contradicting:
        evidence_quality = _cap_quality(evidence_quality, Confidence.MODERATE)
    if any_direction_unresolved:
        evidence_quality = Confidence.LOW

    caveats = list(dict.fromkeys(caveats))

    return PropagationResult(
        intervention=intervention,
        target=target,
        paths_found=len(path_contributions),
        p50=p50,
        ci90=ci90,
        mean=mean,
        p_positive=p_positive,
        p_exceeds=p_exceeds,
        lag_years=lag_years,
        paths=path_contributions,
        evidence_quality=evidence_quality,
        magnitude_confidence=magnitude_confidence,
        caveats=caveats,
        n_sources=n_sources,
        n_agreeing_sources=n_agreeing_sources,
        convergence_factor=_convergence_factor(n_agreeing_sources),
        transmission_factor=TRANSMISSION,
        delta_ref=DELTA_REF,
    )


def _conditions_note(paths: list[PathContribution], graph: nx.MultiDiGraph) -> str:
    notes: set[str] = set()
    for pc in paths:
        for u, v in zip(pc.path, pc.path[1:]):
            for _, data in graph[u][v].items():
                conditions = data["edge"].conditions
                if conditions.rainfall_mm is not None:
                    notes.add(f"rainfall {conditions.rainfall_mm[0]:g}-{conditions.rainfall_mm[1]:g}mm")
                if conditions.slope_pct is not None:
                    notes.add(f"slope {conditions.slope_pct[0]:g}-{conditions.slope_pct[1]:g}%")
                if conditions.climate_zone is not None:
                    notes.add(f"climate zone in {conditions.climate_zone}")
                if conditions.land_use is not None:
                    notes.add(f"land use in {conditions.land_use}")
    if notes:
        return "Bites under: " + "; ".join(sorted(notes)) + "."
    return "No specific site conditions gate this tradeoff; it applies generally."


def _evidence_ids(paths: list[PathContribution], graph: nx.MultiDiGraph) -> list[str]:
    ids: set[str] = set()
    for pc in paths:
        for u, v in zip(pc.path, pc.path[1:]):
            for _, data in graph[u][v].items():
                for ref in data["edge"].evidence:
                    ids.add(ref.source_id)
    return sorted(ids)


def find_tradeoffs(
    graph: nx.MultiDiGraph,
    intervention: str,
    site: SiteState,
    targets: list[str] | None = None,
    n: int = 4000,
    seed: int = 0,
) -> list[Tradeoff]:
    if targets is None:
        targets = [name for name in STATE_VARIABLES if name != intervention]

    results: dict[str, PropagationResult] = {}
    for target in targets:
        if target == intervention:
            continue
        result = propagate(graph, intervention, target, site, n=n, rng=np.random.default_rng(seed))
        if result.paths_found > 0:
            results[target] = result

    tradeoffs: list[Tradeoff] = []

    for target, result in results.items():
        positive = [p for p in result.paths if p.p50 > 0]
        negative = [p for p in result.paths if p.p50 < 0]
        if positive and negative:
            tradeoffs.append(
                Tradeoff(
                    intervention=intervention,
                    positive_target=target,
                    negative_target=target,
                    positive_effect=max(p.p50 for p in positive),
                    negative_effect=min(p.p50 for p in negative),
                    conditions_note=_conditions_note(negative, graph),
                    evidence=_evidence_ids(positive + negative, graph),
                )
            )

    items = list(results.items())
    for i, (t1, r1) in enumerate(items):
        for t2, r2 in items[i + 1 :]:
            if r1.p50 > 0 and r2.p50 < 0:
                pos_target, pos_res, neg_target, neg_res = t1, r1, t2, r2
            elif r2.p50 > 0 and r1.p50 < 0:
                pos_target, pos_res, neg_target, neg_res = t2, r2, t1, r1
            else:
                continue
            tradeoffs.append(
                Tradeoff(
                    intervention=intervention,
                    positive_target=pos_target,
                    negative_target=neg_target,
                    positive_effect=pos_res.p50,
                    negative_effect=neg_res.p50,
                    conditions_note=_conditions_note(neg_res.paths, graph),
                    evidence=_evidence_ids(pos_res.paths + neg_res.paths, graph),
                )
            )

    return tradeoffs


def _missing_data_note(site: SiteState) -> str:
    """Names the measurements a diagnosis had to do without, or an empty string.

    A fall-through diagnosis rests on the absence of a trigger, which is only
    as strong as the data that could have triggered one. Saying which
    measurements were missing keeps that visible instead of presenting a
    default as a finding.
    """
    wanted = ("soil_organic_carbon_pct", "annual_rainfall_mm", "ph", "slope_pct", "edge_density")
    absent = [name for name in wanted if getattr(site, name) is None]
    if not absent:
        return ""
    return " No measurement available for " + ", ".join(absent) + ", so a trigger may have been missed."


def limiting_factor(site: SiteState) -> tuple[str, str]:
    """Liebig's law of the minimum: which single variable most constrains this site.

    Syndromes are evaluated in a fixed order and the first match wins. The
    order encodes priority, not likelihood: a constraint that exports the
    others has to be relieved before the things it exports are worth
    building, so erosion is tested ahead of fertility, and fertility ahead
    of biodiversity.

    Always returns a variable and a reason, never None and never an empty
    diagnosis. A site always has something that binds hardest; where no
    threshold is crossed the fall-through says which variable it settles on
    and admits how little that rests on.
    """
    rainfall = site.annual_rainfall_mm.value if site.annual_rainfall_mm else None
    soc = site.soil_organic_carbon_pct.value if site.soil_organic_carbon_pct else None
    ph = site.ph.value if site.ph else None
    slope = site.slope_pct.value if site.slope_pct else None
    edge_density = site.edge_density

    # 1. Erosion first. Everything below is a stock being built up in the
    # profile, and on steep wet ground that stock leaves the field faster
    # than it accumulates, so relieving any other constraint first is
    # building on ground that is washing away.
    if slope is not None and rainfall is not None and slope > 8 and rainfall > 1000:
        return (
            "erosion_rate",
            f"Slope is {slope:g}% under {rainfall:g}mm/yr of rainfall. On steep ground "
            "under high rainfall, soil loss outpaces soil formation, so carbon and "
            "nutrients added to the profile are exported downslope before they can "
            "accumulate. Erosion control must precede fertility building.",
        )
    if rainfall is not None and soc is not None and rainfall < 500 and soc < 1.0:
        return (
            "plant_available_water",
            "Rainfall is below 500mm/yr and soil organic carbon is below 1%: "
            "biomass-based carbon interventions cannot establish without moisture, "
            "so water harvesting must precede them.",
        )
    if soc is not None and soc < 0.5:
        return (
            "soil_organic_carbon",
            f"Soil organic carbon is {soc}%, below the 0.5% threshold at which soil "
            "structure and nutrient cycling are severely constrained.",
        )
    if ph is not None and (ph < 5.5 or ph > 8.5):
        return (
            "soil_ph",
            f"Soil pH of {ph} falls outside the 5.5-8.5 range in which nutrient "
            "availability and microbial activity are not impaired.",
        )
    # 5. Leaching. This fires at a pH the extreme-pH rule above deliberately
    # passes over: 6.0 is inside the 5.5-8.5 band and is not itself a
    # problem, but under heavy rainfall it is evidence of ongoing base-cation
    # export, and it is the nutrient supply rather than the pH reading that
    # binds.
    if rainfall is not None and ph is not None and rainfall > 1500 and ph < 6.0:
        return (
            "nutrient_cycling_rate",
            f"Rainfall of {rainfall:g}mm/yr with soil pH {ph:g}: high rainfall leaches "
            "base cations and acidifies the profile, so nitrogen and phosphorus "
            "availability limits productivity even where carbon stocks are adequate. "
            "The pH itself is inside the 5.5-8.5 band and is not the constraint; it is "
            "the marker of the leaching that is.",
        )
    if edge_density is not None and (
        edge_density.band == "high" or (edge_density.value is not None and edge_density.value > 0.7)
    ):
        return (
            "habitat_connectivity",
            "Edge density is high, indicating fragmented habitat: patches are too "
            "disconnected for species movement regardless of on-field soil condition.",
        )
    if soc is not None and rainfall is not None and soc >= 1.0 and 500 <= rainfall <= 1500:
        return (
            "species_richness",
            f"Soil organic carbon is {soc:g}% and rainfall {rainfall:g}mm/yr, so neither "
            "soil carbon nor water is limiting. Where soil and water are adequate, "
            "biodiversity is usually constrained by habitat structure and connectivity "
            "rather than by soil fertility.",
        )

    # Fall-through. Nothing crossed a threshold, which is a weaker finding
    # than any branch above and is reported as such.
    if soc is not None and soc < 1.0:
        return (
            "soil_organic_carbon",
            f"Soil organic carbon is {soc:g}%. That is above the 0.5% level at which soil "
            "function is severely constrained, so no threshold is breached, but it is "
            "still below the 1% level at which structure and nutrient supply are secure, "
            "and no other constraint triggered. Carbon is the binding constraint by "
            "elimination rather than by a crossed threshold." + _missing_data_note(site),
        )
    return (
        "species_richness",
        "No soil, water, pH or erosion threshold was crossed by the available "
        "measurements. On that basis the binding constraint is taken to be habitat "
        "structure and connectivity rather than soil fertility, which is a default "
        "rather than a finding." + _missing_data_note(site),
    )


def _targets_limiting_factor(graph: nx.MultiDiGraph, negative_target: str, limiting_var: str) -> bool:
    """True when a negative effect on negative_target lands on the site's
    binding constraint, or on a variable upstream of it (so the harm reaches
    the constraint by propagation).
    """
    if limiting_var == "none" or negative_target not in graph or limiting_var not in graph:
        return False
    if negative_target == limiting_var:
        return True
    return limiting_var in nx.descendants(graph, negative_target)


def _constraint_priority(ranked: RankedIntervention, limiting_var: str) -> float:
    """How strongly a tier 1 intervention relieves the binding constraint.

    Magnitude of the movement discounted by evidence_quality, the same axis
    the multi-objective score discounts by, so a large mechanistic-only claim
    does not displace a smaller well-evidenced one. Returns 0.0 for tier 2 so
    that tier's ordering is left to the score alone.
    """
    if ranked.tier != 1 or ranked.constraint_movement is None:
        return 0.0
    result = ranked.effects.get(limiting_var)
    if result is None:
        return 0.0
    return abs(ranked.constraint_movement) * _CONFIDENCE_FACTOR[result.evidence_quality]


def _order_tier_1(
    entries: list[RankedIntervention], limiting_var: str
) -> list[RankedIntervention]:
    """Order tier 1 by constraint relief, with comparable relief decided on
    the multi-objective score.

    Entries are grouped by priority: each group starts with the highest
    remaining entry as its leader and absorbs every following entry within
    PRIORITY_TOLERANCE of that leader. Groups stay in priority order, and
    within a group the score decides, so substantially better relief wins
    outright while comparable relief is settled by evidence quality and
    co-benefits.

    Grouping is greedy from the leader rather than transitive between
    neighbours, so a long chain of small steps cannot merge into one group
    whose ends differ by far more than the tolerance.
    """
    remaining = sorted(entries, key=lambda r: -_constraint_priority(r, limiting_var))

    ordered: list[RankedIntervention] = []
    group: list[RankedIntervention] = []
    leader_priority = 0.0

    for entry in remaining:
        priority = _constraint_priority(entry, limiting_var)
        if group and priority >= leader_priority * (1.0 - PRIORITY_TOLERANCE):
            group.append(entry)
            continue
        ordered.extend(sorted(group, key=lambda r: -r.score))
        group = [entry]
        leader_priority = priority

    ordered.extend(sorted(group, key=lambda r: -r.score))
    return ordered


def _sequencing_note(intervention: str, negative_targets: set[str], limiting_var: str, why: str) -> str:
    targets = ", ".join(sorted(negative_targets))
    return (
        f"{why} {intervention} works against that constraint through its negative effect "
        f"on {targets}, so it should follow interventions that address {limiting_var} "
        f"rather than precede them."
    )


def rank_interventions(
    graph: nx.MultiDiGraph,
    site: SiteState,
    objectives: dict[str, float] | None = None,
    n: int = 4000,
    seed: int = 0,
) -> list[RankedIntervention]:
    if objectives is None:
        objectives = DEFAULT_OBJECTIVES

    limiting_var, limiting_why = limiting_factor(site)
    constraint_direction = _improvement_direction(limiting_var, site)

    # The binding constraint is propagated for every intervention so the tier
    # partition below can be decided on a real effect size, but it is scored
    # at no weight and stays out of `weights`. Priority for relieving it is
    # expressed by the tier and nowhere else: weighting it as well would
    # reintroduce, in a smaller way, the same category error as boosting it
    # -- treating a gate as a preference. It is still reported, because a
    # reader should see what an intervention does to the constraint.
    scored_targets = dict(objectives)
    propagated_targets = list(scored_targets)
    if limiting_var in STATE_VARIABLES and limiting_var not in propagated_targets:
        propagated_targets.append(limiting_var)

    ranked: list[RankedIntervention] = []
    for intervention in INTERVENTIONS:
        effects: dict[str, PropagationResult] = {}
        score = 0.0
        vars_touched: set[str] = set()

        for target in propagated_targets:
            result = propagate(graph, intervention, target, site, n=n, rng=np.random.default_rng(seed))
            effects[target] = result
            if result.paths_found == 0:
                continue
            for pc in result.paths:
                vars_touched.update(pc.path)

            weight = scored_targets.get(target, 0.0)
            if weight:
                # Score on evidence_quality, not magnitude_confidence: p50
                # already comes from a distribution whose width reflects
                # magnitude uncertainty (including contested disagreement),
                # so discounting the score by magnitude_confidence as well
                # would count that same uncertainty twice. evidence_quality
                # -- how well-supported the relationship is -- is the axis
                # a ranking score should discount by.
                cf = _CONFIDENCE_FACTOR[result.evidence_quality]
                # Improvement is a rise for every standing objective and a
                # fall for a variable like erosion, so the propagated change
                # enters the score with the target's improvement direction
                # applied. A no-op for the default objectives, which are all
                # higher-is-better, and correct if a caller passes others.
                direction = CONSTRAINT_DIRECTION.get(target, 1.0)
                score += weight * result.p50 * direction * cf * result.convergence_factor

        # Deliberately objectives, not weights: an injected constraint from
        # CONSTRAINT_DIRECTION can be lower-is-better, and the pairing below
        # reads a negative p50 as harm. Feeding erosion_rate in would flag
        # contour bunding as conflicting with the erosion constraint it
        # relieves. Tradeoff detection stays on the higher-is-better set
        # until it understands direction.
        tradeoffs = find_tradeoffs(graph, intervention, site, targets=list(objectives.keys()), n=n, seed=seed)
        conflicting_targets: set[str] = set()
        for tradeoff in tradeoffs:
            neg_result = effects.get(tradeoff.negative_target)
            if neg_result is not None:
                neg_paths = [p for p in neg_result.paths if p.p50 < 0]
                site_meets = any(p.min_satisfaction >= 0.5 for p in neg_paths) if neg_paths else True
            else:
                site_meets = True
            if not site_meets:
                continue
            # A tradeoff that lands on the binding constraint is not one cost
            # among several: until that constraint is relieved, nothing
            # downstream of it can be realised, so the harm outweighs gains
            # elsewhere instead of merely offsetting them.
            if _targets_limiting_factor(graph, tradeoff.negative_target, limiting_var):
                penalty_weight = LIMITING_FACTOR_PENALTY_WEIGHT
                conflicting_targets.add(tradeoff.negative_target)
            else:
                penalty_weight = TRADEOFF_PENALTY_WEIGHT
            score -= penalty_weight * abs(tradeoff.negative_effect)

        n_variables_touched = len({v for v in vars_touched if v in STATE_VARIABLES})

        # Tier 1 requires all three of: a path to the constraint, movement in
        # the direction the site needs, and enough of it to matter. A path
        # alone is not enough -- an intervention that raises erosion has a
        # path to erosion_rate too.
        constraint_result = effects.get(limiting_var)
        addresses = False
        constraint_movement: float | None = None
        if constraint_result is not None and constraint_result.paths_found > 0:
            constraint_movement = constraint_result.p50
            if constraint_direction:
                improvement = constraint_result.p50 * constraint_direction
                addresses = improvement > TIER_1_MIN_EFFECT

        ranked.append(
            RankedIntervention(
                intervention=intervention,
                score=score,
                effects=effects,
                tradeoffs=tradeoffs,
                n_variables_touched=n_variables_touched,
                limiting_factor_addressed=limiting_var in vars_touched,
                addresses_limiting_factor=addresses,
                tier=1 if addresses else 2,
                constraint_movement=constraint_movement,
                conflicts_with_limiting_factor=bool(conflicting_targets),
                sequencing_note=(
                    _sequencing_note(intervention, conflicting_targets, limiting_var, limiting_why)
                    if conflicting_targets
                    else None
                ),
            )
        )

    # Liebig's law of the minimum as a gate: every intervention that relieves
    # the binding constraint ranks above every intervention that does not,
    # whatever their scores, because a gain sitting behind a shut gate cannot
    # be realised.
    #
    # Within tier 1 the question is no longer "what is best overall" but
    # "what most relieves the constraint", since by Liebig's law nothing else
    # can be realised until it lifts. So tier 1 is ordered by how far it
    # moves the constraint, discounted by evidence_quality on the same axis
    # used everywhere else, with the multi-objective score deciding between
    # interventions whose relief is comparable (see _order_tier_1 and
    # PRIORITY_TOLERANCE). Co-benefits never outweigh substantially better
    # relief. Ordering tier 1 by the multi-objective score instead put an
    # intervention scoring 0.0 -- because the constraint carries no score
    # weight -- above better ones, which defeated the gate from inside.
    #
    # Tier 2 ordering is by score alone. Its members do not move the
    # constraint by definition, so no grouping applies.
    #
    # When no intervention in the graph can address the constraint, tier 1 is
    # empty and this degrades to the plain score ranking. That case is left
    # visible rather than patched: every returned item carries tier 2, which
    # is what callers check to report the fallback instead of presenting a
    # list that silently ignores the diagnosis.
    tier_1 = _order_tier_1([r for r in ranked if r.tier == 1], limiting_var)
    tier_2 = sorted((r for r in ranked if r.tier == 2), key=lambda r: -r.score)
    return tier_1 + tier_2
