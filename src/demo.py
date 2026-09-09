"""End-to-end vertical slice: site state in, diagnosed recommendations out.

No retrieval, no agent loop, no interface. This module exists to prove that
the curated causal graph plus the Monte Carlo propagation engine already
produce the output shape the brief asks for: a diagnosis, a ranked set of
interventions, a quantified effect per environmental variable with an
interval and a time horizon, a confidence statement on two axes, real
citations, caveats, and sequencing.

Every number printed here comes from src.graph.propagate. Every citation
string is read verbatim from sources.yaml by source_id. Nothing in this file
constructs a citation, and nothing rounds a distribution down to a point
value before the interval has been shown alongside it.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from src.graph.edges import build_graph
from src.graph.propagate import (
    CONVERGENCE_SOURCE_CAP,
    CONVERGENCE_STEP,
    DELTA_REF,
    PRIORITY_TOLERANCE,
    TIER_1_MIN_EFFECT,
    TRANSMISSION,
    PropagationResult,
    RankedIntervention,
    limiting_factor,
    rank_interventions,
)
from src.graph.schemas import Confidence, Measurement, Provenance, SiteState

_SOURCES_YAML_PATH = Path(__file__).resolve().parents[1] / "sources.yaml"

# Effects smaller than this are not worth printing as an "impacted metric".
# A tenth of a percent is below the resolution of any field measurement a
# farmer could verify, so reporting it would imply precision the model does
# not have.
_REPORTING_FLOOR = 0.001

# Display labels only. The graph node name stays the identifier everywhere
# else; this dict never feeds back into computation.
_VARIABLE_LABELS: dict[str, str] = {
    "soil_ph": "soil pH",
    "plant_available_water": "plant available water",
    "soil_organic_carbon": "soil organic carbon",
    "microbial_biomass_carbon": "microbial biomass carbon",
    "mycorrhizal_colonisation": "mycorrhizal colonisation",
}

# SiteState field names to display labels. Separate from _VARIABLE_LABELS
# because site fields carry unit suffixes the graph nodes do not.
_SITE_FIELD_LABELS: dict[str, str] = {
    "soil_organic_carbon_pct": "soil organic carbon",
    "ph": "pH",
    "clay_pct": "clay",
    "sand_pct": "sand",
    "annual_rainfall_mm": "annual rainfall",
    "mean_temperature_c": "mean temperature",
    "slope_pct": "slope",
    "land_use": "land use",
    "observed_species_richness": "observed species richness",
    "edge_density": "edge density",
    "largest_patch_index": "largest patch index",
}

# Human-facing title and the concrete action for each intervention node.
# Authored prose describing the practice, not an empirical claim, so it
# carries no source_id. Every quantitative claim about these practices comes
# from the graph instead.
_ACTIONS: dict[str, tuple[str, str]] = {
    "legume_cover_crop": (
        "Legume-based cover crop",
        "Sow a legume cover crop (cowpea, horsegram or sunn hemp) into the fallow "
        "window straight after the main harvest, at 20 to 25 kg/ha. Terminate it by "
        "rolling or shallow incorporation two to three weeks before the next sowing "
        "so the residue stays on the surface.",
    ),
    "non_legume_cover_crop": (
        "Non-legume cover crop",
        "Sow a grass or brassica cover (oats, mustard, fodder sorghum) in the fallow "
        "window for root mass and surface cover. Terminate before it sets seed and "
        "leave the residue in place.",
    ),
    "crop_rotation_diversification": (
        "Crop rotation diversification",
        "Break the current monoculture sequence by adding at least one unrelated "
        "crop family to the rotation, ideally a pulse. Plan the sequence three "
        "seasons ahead rather than deciding crop by crop.",
    ),
    "intercropping": (
        "Intercropping",
        "Plant a companion species between the main crop rows at a fixed row ratio "
        "such as 4:2. Pick a companion with a different rooting depth so the two "
        "crops draw water and nutrients from different parts of the profile.",
    ),
    "alley_cropping": (
        "Alley cropping",
        "Establish tree or shrub rows across the field at 10 to 20 m spacing and "
        "keep cropping the alleys between them. Prune the rows annually and return "
        "the prunings to the alley soil.",
    ),
    "boundary_tree_planting": (
        "Boundary tree planting",
        "Plant a single line of trees along field boundaries at 3 to 5 m spacing, "
        "leaving the cropped area itself untouched. Prefer species already "
        "established locally so survival does not depend on irrigation.",
    ),
    "hedgerow_planting": (
        "Hedgerow planting",
        "Plant a mixed woody shrub strip 1 to 2 m wide along field margins, using "
        "three or more species so it flowers across a long window. Cut it back on a "
        "rotation so only part of the hedge is trimmed in any one year.",
    ),
    "reduced_tillage": (
        "Reduced tillage",
        "Replace full inversion ploughing with one shallow pass, no deeper than "
        "100 mm, or with strip tillage in the seeding row only. Leave the "
        "inter-row undisturbed.",
    ),
    "no_tillage": (
        "No tillage",
        "Direct-seed into the previous crop's residue with a zero-till drill and "
        "stop all mechanical disturbance between crops. Manage weeds by residue "
        "cover and rotation rather than by cultivation.",
    ),
    "residue_retention": (
        "Residue retention",
        "Stop removing or burning crop residue. Leave at least 30 percent of the "
        "surface covered by anchored stubble and spread the remainder evenly rather "
        "than windrowing it.",
    ),
    "mulching": (
        "Mulching",
        "Apply 5 to 8 t/ha of organic mulch (crop residue, pruning material or dry "
        "grass) over the soil surface, concentrating it around the crop root zone "
        "before the hottest part of the season.",
    ),
    "farmyard_manure": (
        "Farmyard manure",
        "Apply 5 to 10 t/ha of well-composted farmyard manure before sowing and "
        "incorporate it shallowly. Compost it for at least eight weeks first so "
        "nitrogen is not lost as ammonia on application.",
    ),
    "compost_application": (
        "Compost application",
        "Apply 4 to 8 t/ha of mature compost before sowing. Build the compost from "
        "on-farm residue and animal waste rather than buying it in, so the practice "
        "is repeatable every season.",
    ),
    "biochar_application": (
        "Biochar application",
        "Apply 2 to 5 t/ha of biochar, charged with manure or compost before "
        "application so it does not immobilise nutrients in the first season. "
        "Incorporate into the top 100 mm.",
    ),
    "contour_bunding": (
        "Contour bunding",
        "Survey the field on the contour and build earthen bunds 0.5 to 0.75 m high "
        "along those lines, spaced by slope so each bund intercepts runoff before "
        "it concentrates. Stabilise the bund face with grass.",
    ),
    "contour_trenching": (
        "Contour trenching",
        "Dig staggered trenches roughly 0.45 m wide and deep along the contour, "
        "with earth thrown downslope. Space them so each trench holds the runoff "
        "from the strip above it.",
    ),
    "farm_pond": (
        "Farm pond",
        "Excavate a pond at the lowest point of the catchment, sized to the runoff "
        "the field actually generates, and lead runoff into it through a silt trap. "
        "Fence it and desilt annually.",
    ),
    "check_dam": (
        "Check dam",
        "Build a low masonry or loose-boulder barrier across the drainage line "
        "where flow already concentrates, keyed into both banks. Add a stone apron "
        "downstream so the structure does not undercut itself.",
    ),
    "vetiver_grass_strips": (
        "Vetiver grass strips",
        "Plant vetiver slips 100 to 150 mm apart in a single dense line along the "
        "contour, in strips spaced by slope. Trim to 300 to 500 mm so the hedge "
        "stays thick at the base.",
    ),
    "rotational_grazing": (
        "Rotational grazing",
        "Split the grazing area into paddocks and move stock on a planned rotation, "
        "giving each paddock a rest long enough for regrowth before it is grazed "
        "again. Match stocking to the season's actual forage.",
    ),
    "grazing_exclosure": (
        "Grazing exclosure",
        "Fence livestock out of the most degraded block and leave it undisturbed "
        "for at least two growing seasons. Agree the closure with other users of "
        "the land before fencing.",
    ),
    "integrated_nutrient_management": (
        "Integrated nutrient management",
        "Soil-test the field, then meet crop demand from a planned combination of "
        "organic amendment and mineral fertiliser rather than from either alone. "
        "Split the mineral portion across the season.",
    ),
    "agroforestry_silvopasture": (
        "Silvopasture",
        "Establish scattered or clustered trees across the grazed area at low "
        "density, protecting each seedling from browsing until it is above "
        "browse height. Graze the understorey on a rotation.",
    ),
}


def _citations() -> dict[str, str]:
    """source_id to citation string, read verbatim from sources.yaml.

    Citation strings are never assembled in code. A source_id with no
    citation field is a registry error and must fail loudly rather than
    print a placeholder.
    """
    data = yaml.safe_load(_SOURCES_YAML_PATH.read_text())
    citations: dict[str, str] = {}
    for source_id, info in data.get("sources", {}).items():
        if not isinstance(info, dict):
            continue
        citation = info.get("citation")
        if citation:
            citations[source_id] = citation
    return citations


_CITATIONS = _citations()


def _label(node: str) -> str:
    return _VARIABLE_LABELS.get(node, node.replace("_", " "))


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _years(lag: tuple[float, float]) -> str:
    if lag[0] == lag[1]:
        return f"{lag[0]:g} years"
    return f"{lag[0]:g}-{lag[1]:g} years"


def _horizon_band(lag_low: float) -> str:
    """Band the horizon by when the effect starts to appear, not when it
    finishes. A practice whose first benefit lands in year two is a
    medium-term action even if the chain keeps accruing for a decade.
    """
    if lag_low < 2:
        return "short term"
    if lag_low < 6:
        return "medium term"
    return "long term"


def _reportable(ranked: RankedIntervention) -> list[tuple[str, PropagationResult]]:
    """Impacted metrics worth printing, strongest effect first."""
    items = [
        (target, result)
        for target, result in ranked.effects.items()
        if result.paths_found > 0 and abs(result.p50) >= _REPORTING_FLOOR
    ]
    items.sort(key=lambda item: abs(item[1].p50), reverse=True)
    return items


# Longest chain worth narrating in prose, counted in edges. Beyond this the
# mechanism paragraph turns into a wall of text that obscures the reasoning
# it is meant to expose. The deeper paths still contribute to the numbers and
# still appear in the caveats.
_MAX_EXPLANATORY_EDGES = 4


def _explanatory_path(reportable: list[tuple[str, PropagationResult]]) -> list[str]:
    """The chain to narrate under "Why it works".

    Each result's paths are already sorted with the dominant mechanism
    first, so this considers the dominant path of every reported metric.
    Narrating the dominant path to the largest effect alone would often be a
    single edge (intervention straight to soil organic carbon), which hides
    the multi-variable reasoning that is the point of the graph. So this
    takes the deepest dominant path that still fits in a readable paragraph,
    falling back to the shortest available when every candidate is deeper
    than that. Either way it is a mechanism the engine actually propagated,
    not a chain assembled for the retelling.
    """
    candidates = [result.paths[0].path for _, result in reportable if result.paths]
    if not candidates:
        return []
    readable = [path for path in candidates if len(path) - 1 <= _MAX_EXPLANATORY_EDGES]
    if readable:
        return max(readable, key=len)
    return min(candidates, key=len)


def _why_it_works(graph: nx.MultiDiGraph, node_path: list[str]) -> str:
    """Mechanism prose for one path, read off the graph.

    The sentences are the curated mechanism fields of the edges along the
    path, so the explanation cannot drift away from what the model actually
    computed.
    """
    if not node_path:
        return "No mechanism recorded."

    sentences: list[str] = []
    for u, v in zip(node_path, node_path[1:]):
        for _, data in graph[u][v].items():
            mechanism = data["edge"].mechanism.strip()
            if mechanism and mechanism not in sentences:
                sentences.append(mechanism)
            break

    chain = " -> ".join(_label(node) for node in node_path)
    body = " ".join(sentences)
    # Named by its endpoint. The deepest dominant path does not always lead to
    # the largest-effect metric, so an unlabelled chain can read as an
    # explanation of the headline number when it explains a different one.
    return f"{body}\n    Causal chain to {_label(node_path[-1])}: {chain}."


def _evidence_lines(graph: nx.MultiDiGraph, node_path: list[str]) -> list[str]:
    """Citation strings with roles, for the edges along one path.

    Takes the same path that "Why it works" narrates, so the citations back
    the mechanism actually described. Only source_ids registered in
    sources.yaml are emitted. An unregistered id raises rather than printing
    an unciteable claim.
    """
    if not node_path:
        return []

    seen: list[tuple[str, str]] = []
    for u, v in zip(node_path, node_path[1:]):
        for _, data in graph[u][v].items():
            for ref in data["edge"].evidence:
                key = (ref.source_id, ref.role)
                if key not in seen:
                    seen.append(key)
            break

    lines: list[str] = []
    for source_id, role in seen:
        if source_id not in _CITATIONS:
            raise ValueError(
                f"source_id {source_id!r} is not registered in sources.yaml with a citation; "
                "it must not appear in output"
            )
        lines.append(f"[{role}] {_CITATIONS[source_id]}")
    return lines


def _caveat_lines(ranked: RankedIntervention, reportable: list[tuple[str, PropagationResult]]) -> list[str]:
    caveats: list[str] = []
    mechanistic_paths: list[str] = []

    for _, result in reportable:
        for caveat in result.caveats:
            # One "rests on mechanistic evidence only" line per path buries
            # everything else under near-identical text. Count them and say
            # so once, keeping the fact without the wall.
            if caveat.startswith("Path ") and "mechanistic evidence only" in caveat:
                if caveat not in mechanistic_paths:
                    mechanistic_paths.append(caveat)
                continue
            if caveat not in caveats:
                caveats.append(caveat)

    if mechanistic_paths:
        count = len(mechanistic_paths)
        subject = "path" if count == 1 else "paths"
        verb = "rests" if count == 1 else "rest"
        caveats.append(
            f"{count} causal {subject} behind these figures {verb} on mechanistic evidence "
            "only: assessment reports give the direction, no meta-analysis gives the size. "
            "That contribution to the intervals above is an estimate, not a published number."
        )

    low_magnitude = [
        _label(result.target)
        for _, result in reportable
        if result.magnitude_confidence == Confidence.LOW
    ]
    if low_magnitude:
        caveats.append(
            f"Magnitude is low confidence for {', '.join(low_magnitude)}: treat the size of "
            "those effects as indicative and the direction as the load-bearing claim."
        )

    for tradeoff in ranked.tradeoffs:
        note = (
            f"Tradeoff: {_label(tradeoff.positive_target)} gains {_pct(tradeoff.positive_effect)} "
            f"while {_label(tradeoff.negative_target)} loses {_pct(abs(tradeoff.negative_effect))}. "
            f"{tradeoff.conditions_note}"
        )
        if note not in caveats:
            caveats.append(note)
    return caveats


def _render_recommendation(
    graph: nx.MultiDiGraph, index: int, ranked: RankedIntervention, limiting_var: str
) -> list[str]:
    title, action = _ACTIONS[ranked.intervention]
    reportable = _reportable(ranked)

    lines: list[str] = []
    lines.append(f"RECOMMENDATION {index}: {title}")
    lines.append(f"What to do: {action}")

    if not reportable:
        lines.append(
            "Why it works: no causal path from this intervention reaches the scored "
            "objectives at this site, so it is not justified here."
        )
        return lines

    primary = reportable[0][1]
    explanatory = _explanatory_path(reportable)
    lines.append(f"Why it works: {_why_it_works(graph, explanatory)}")

    lines.append("Impacted metrics:")
    width = max(len(_label(target)) for target, _ in reportable)
    for target, result in reportable:
        ci = f"(90% CI {_pct(result.ci90[0])} to {_pct(result.ci90[1])})"
        row = f"  {_label(target):<{width}}  {_pct(result.p50):>7}  {ci:<34}  {_years(result.lag_years)}"
        # Mark the binding constraint's own row. Tier 1 is ordered by this
        # number, so showing which figure did the ordering makes the position
        # of every tier 1 recommendation checkable from the output alone.
        if target == limiting_var:
            row += "   <- binding constraint, orders tier 1"
        lines.append(row)

    if ranked.constraint_movement is None:
        lines.append(
            f"  (no path to {_label(limiting_var)}, this site's binding constraint)"
        )
    elif limiting_var not in {target for target, _ in reportable}:
        # Reported even when it is too small for the table, because "this
        # moves the constraint by almost nothing" is the reason an
        # intervention sits in tier 2 and is worth stating.
        lines.append(
            f"  {_label(limiting_var):<{width}}  {_pct(ranked.constraint_movement):>7}  "
            f"   <- binding constraint, below the reporting floor"
        )

    # Band the horizon on the primary metric, which is the effect the
    # recommendation actually rests on. Banding on the longest chain in the
    # table would call a two-year soil carbon gain "long term" purely
    # because a nine-hop yield path accumulates lag along the way.
    lag_high = max(result.lag_years[1] for _, result in reportable)
    band = _horizon_band(primary.lag_years[0])
    horizon = f"Time horizon: {band} ({_years(primary.lag_years)} on {_label(primary.target)}"
    if lag_high > primary.lag_years[1]:
        horizon += f"; the full chain keeps accruing out to {lag_high:g} years"
    lines.append(horizon + ")")

    noun = "source" if primary.n_sources == 1 else "sources"
    lines.append(
        f"Confidence: evidence {primary.evidence_quality.value}, "
        f"magnitude {primary.magnitude_confidence.value} "
        f"(assessed on the dominant path to {_label(primary.target)}; "
        f"{primary.n_agreeing_sources} agreeing of {primary.n_sources} {noun})"
    )
    lines.append(
        f"Variables in the justification: {ranked.n_variables_touched} "
        f"(the brief's floor is three)"
    )
    # Why this intervention sits where it does in the list. The tier is the
    # whole ranking decision, so stating it beats leaving the reader to infer
    # it from the order.
    if ranked.tier == 1:
        constraint = ranked.effects.get(limiting_var)
        movement = f" ({_pct(constraint.p50)} on it)" if constraint is not None else ""
        lines.append(
            f"Tier 1: addresses {_label(limiting_var)}, this site's binding "
            f"constraint{movement}. Ranked above everything that does not, whatever the "
            f"scores."
        )
    elif ranked.limiting_factor_addressed:
        lines.append(
            f"Tier 2: has a causal path to {_label(limiting_var)}, this site's binding "
            f"constraint, but does not move it far enough in the direction the site needs "
            f"to count as addressing it (floor {TIER_1_MIN_EFFECT:.0%})."
        )
    else:
        lines.append(
            f"Tier 2: no causal path from this intervention reaches "
            f"{_label(limiting_var)}, this site's binding constraint. It earns its place on "
            f"the standing objectives alone."
        )

    lines.append("Evidence:")
    for line in _evidence_lines(graph, explanatory):
        lines.append(f"  - {line}")

    caveats = _caveat_lines(ranked, reportable)
    if caveats:
        lines.append("Caveats:")
        for caveat in caveats:
            lines.append(f"  - {caveat}")
    else:
        lines.append("Caveats: none recorded for this site.")

    if ranked.sequencing_note:
        lines.append(f"Sequencing: {ranked.sequencing_note}")

    return lines


def _methodology_lines() -> list[str]:
    convergence_cap = 1.0 + CONVERGENCE_STEP * CONVERGENCE_SOURCE_CAP
    return [
        "METHODOLOGY",
        "The following are modelling assumptions, not empirical constants. They are",
        "choices about how evidence is combined, and no published study supports any",
        "of these values. They are listed so the numbers above can be reproduced and",
        "argued with.",
        "",
        f"  TRANSMISSION                      {TRANSMISSION:g}",
        "    Fraction of a signal that survives each edge crossed after the point of",
        "    application. Not applied to an edge leaving the intervention itself,",
        "    because a published edge out of an intervention already is the measured",
        "    end-to-end effect of adopting it.",
        f"  DELTA_REF                         {DELTA_REF:g}",
        "    Reference upstream proportional change an edge's effect is assumed to",
        "    have been measured against. An edge fires at full strength only when its",
        "    upstream node has moved by at least this much, and scales down below it.",
        f"  convergence cap                   {convergence_cap:g}x "
        f"(step {CONVERGENCE_STEP:g}, at most {CONVERGENCE_SOURCE_CAP} extra sources)",
        "    Reward applied when independent sources agree on an edge's direction.",
        "    Counted from primary and corroborating references only.",
        f"  TIER_1_MIN_EFFECT                 {TIER_1_MIN_EFFECT:.0%}",
        "    Smallest movement on the binding constraint, in the direction the site",
        "    needs, that counts as addressing it. Interventions that clear it are",
        "    tier 1 and rank above every intervention that does not, whatever the",
        "    scores. Liebig's law of the minimum is a gate, not a weighting: a gain",
        "    downstream of a constraint that is still shut cannot be realised, so the",
        "    binding constraint is not given a heavier objective weight. It carries no",
        "    weight in the score at all, and is reported rather than scored.",
        f"  PRIORITY_TOLERANCE                {PRIORITY_TOLERANCE:.0%} relative",
        "    How close two tier 1 interventions' constraint relief has to be before",
        "    they count as comparable, at which point the multi-objective score",
        "    decides between them. An admission about precision rather than a claim:",
        "    these figures come from a Monte Carlo sample, so a gap of a fraction of",
        "    a percentage point is sampling noise, and treating it as decisive would",
        "    be false precision. Substantially better relief still wins outright.",
        "",
        "Effect sizes are sampled as distributions, composed multiplicatively along a",
        "causal chain, and combined across distinct mechanisms feeding one variable",
        "with noisy-OR. Intervals are 90% Monte Carlo intervals on the propagated",
        "distribution, not published confidence intervals.",
    ]


def run(site: SiteState, top_n: int = 4) -> str:
    """Diagnose a site and render the top_n interventions as a text report."""
    graph = build_graph()

    limiting_var, limiting_why = limiting_factor(site)
    ranked = rank_interventions(graph, site)

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"SITE: {site.site_id}")
    if site.lat is not None and site.lon is not None:
        lines.append(f"Location: {site.lat}, {site.lon}")
    lines.append("=" * 78)
    lines.append("")

    lines.append("DIAGNOSIS")
    zone = site.climate_zone()
    observed: list[str] = []
    for field_name in site.known():
        measurement: Measurement = getattr(site, field_name)
        label = _SITE_FIELD_LABELS.get(field_name, field_name.replace("_", " "))
        if measurement.value is None:
            observed.append(f"{label} {measurement.band}")
            continue
        unit = measurement.unit or ""
        # Percent signs sit flush against the number; every other unit takes
        # a space, and pH is a scale rather than a unit to print.
        if unit == "%":
            observed.append(f"{label} {measurement.value:g}%")
        elif unit in ("", "pH"):
            observed.append(f"{label} {measurement.value:g}")
        else:
            observed.append(f"{label} {measurement.value:g} {unit}")
    lines.append(
        f"Observed state: {', '.join(observed)}."
        + (f" Climate zone inferred as {zone.replace('_', '-')}." if zone else "")
    )
    # limiting_factor always names a variable now, falling through to a default
    # with its reasoning rather than to a "none" sentinel, so there is no
    # unconstrained branch to render here.
    lines.append(f"Binding constraint: {_label(limiting_var)}.")
    lines.append(f"Reasoning: {limiting_why}")
    tier_1 = [item for item in ranked if item.tier == 1]
    if tier_1:
        lines.append(
            f"Under Liebig's law of the minimum, gains elsewhere cannot be realised until "
            f"{_label(limiting_var)} is relieved, so the ranking is a gate and not a "
            f"weighting: the {len(tier_1)} interventions that move "
            f"{_label(limiting_var)} by at least {TIER_1_MIN_EFFECT:.0%} in the direction "
            f"this site needs are ranked above every intervention that does not, whatever "
            f"their scores. Interventions that work against the constraint are pushed later "
            f"in the sequence rather than dropped."
        )
    else:
        lines.append(
            f"No intervention in the graph addresses {_label(limiting_var)} at this site by "
            f"at least {TIER_1_MIN_EFFECT:.0%} in the direction it needs, so the gate has "
            f"nothing to admit and the ranking below falls back to score order on the "
            f"standing objectives. Read it knowing that the site's binding constraint is "
            f"going unaddressed, which is a gap in the graph or in the site data rather "
            f"than a finding about these interventions."
        )
    lines.append("")

    selected = ranked[:top_n]
    for index, item in enumerate(selected, start=1):
        lines.extend(_render_recommendation(graph, index, item, limiting_var))
        lines.append("")

    conflicting = [item for item in selected if item.conflicts_with_limiting_factor]
    # An intervention whose tradeoff hits the binding constraint is penalised,
    # so it often falls below the top_n cut. Dropping it silently would lose
    # the sequencing story entirely, which is the one case this section
    # exists for: the answer is "later", not "never".
    deferred = [
        item
        for item in ranked[top_n:]
        if item.conflicts_with_limiting_factor and item.intervention in _ACTIONS
    ]
    if conflicting or deferred:
        lines.append("SEQUENCING")
        lines.append(
            f"The interventions below carry a tradeoff that lands on "
            f"{_label(limiting_var)}, this site's binding constraint, or on a variable "
            f"upstream of it. They are not wrong here, but they must not go first: until "
            f"{_label(limiting_var)} is relieved, their own downstream gains cannot be "
            f"realised either."
        )
        for item in conflicting + deferred:
            title, _ = _ACTIONS[item.intervention]
            rank_position = ranked.index(item) + 1
            lines.append(f"  - {title} (ranked {rank_position}): {item.sequencing_note}")
        first = [item for item in selected if not item.conflicts_with_limiting_factor]
        if first:
            names = ", ".join(_ACTIONS[item.intervention][0] for item in first)
            lines.append(
                f"Sequence {names} first, then revisit the interventions above once "
                f"{_label(limiting_var)} has improved."
            )
        lines.append("")

    lines.extend(_methodology_lines())
    return "\n".join(lines)


def _deccan_semiarid() -> SiteState:
    return SiteState(
        site_id="deccan_semiarid",
        lat=17.85,
        lon=75.42,
        soil_organic_carbon_pct=Measurement(
            value=0.35, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=340, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=8.1, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        slope_pct=Measurement(
            value=5, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )


def _indo_gangetic() -> SiteState:
    return SiteState(
        site_id="indo_gangetic",
        lat=29.15,
        lon=76.32,
        soil_organic_carbon_pct=Measurement(
            value=0.55, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=620, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=7.6, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        slope_pct=Measurement(
            value=1, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )


def _western_ghats() -> SiteState:
    return SiteState(
        site_id="western_ghats",
        lat=15.60,
        lon=74.05,
        soil_organic_carbon_pct=Measurement(
            value=1.20, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=2100, unit="mm", provenance=Provenance.API_NASA_POWER, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=6.2, unit="pH", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH),
        slope_pct=Measurement(
            value=12, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )


DEMO_SITES: dict[str, SiteState] = {
    "deccan_semiarid": _deccan_semiarid(),
    "indo_gangetic": _indo_gangetic(),
    "western_ghats": _western_ghats(),
}


if __name__ == "__main__":
    print(run(DEMO_SITES["deccan_semiarid"], top_n=4))
    print()
    for site_id in ("indo_gangetic", "western_ghats"):
        print(run(DEMO_SITES[site_id], top_n=1))
        print()
