"""Tests for the agent state machine: intake, questions, critic, memory.

Every test runs offline. No language model and no live API is required, and
DARUKAA_OFFLINE is set for the module so an accidental network call cannot
turn a unit test into a flaky one. That is a property of the design rather
than a concession to the test harness: the deterministic parser is the
primary intake path, entailment falls back to the retrieval floor, and
acquire is cache-first and degrades to asking.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DARUKAA_OFFLINE", "1")

from src.agents import nodes
from src.agents.graph import Conversation, build_agent_graph
from src.agents.nodes import (
    ASKABLE_FIELDS,
    CHURN_FLOOR,
    MAX_QUESTIONS,
    apply_withdrawals,
    critic_node,
    decompose,
    gap_analysis_node,
    intake_node,
    parse_free_text,
    value_of_information,
)
from src.agents.state import (
    MAX_CRITIC_PASSES,
    Claim,
    initial_state,
    load_site_profile,
    save_site_profile,
    site_profile_path,
)
from src.graph.edges import build_graph
from src.graph.propagate import limiting_factor, rank_interventions
from src.graph.schemas import Confidence, Measurement, Provenance, SiteState
from src.retrieval.index import build_indexes


@pytest.fixture(scope="session", autouse=True)
def indexes() -> None:
    """The critic retrieves, so the indexes have to exist."""
    build_indexes()


@pytest.fixture(scope="session")
def graph():
    return build_graph()


def _deccan(slope: float | None = None) -> SiteState:
    """The Deccan demo site, optionally with slope supplied.

    Slope is left out by default because that is the case the value-of-
    information test is about: contour bunding's preconditions are gated on
    slope 2-25%, so an unknown slope scores 0.5 satisfaction and halves the
    water effect.
    """
    site = SiteState(
        site_id="deccan_test",
        lat=17.85,
        lon=75.42,
        soil_organic_carbon_pct=Measurement(
            value=0.35, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            value=340, unit="mm", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
        ph=Measurement(
            value=8.1, unit="pH", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
        land_use=Measurement(
            band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
    )
    if slope is not None:
        site.slope_pct = Measurement(
            value=slope, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        )
    return site


def _state(site: SiteState, text: str | None = None):
    state = initial_state(site)
    if text is not None:
        state["messages"] = [{"role": "user", "content": text}]
    return state


# ============================== 1. free text ==============================


def test_free_text_intake_populates_soc_rainfall_band_crop_and_land_use():
    text = (
        "biodiversity is declining, soil organic carbon is 0.3%, rainfall is low, "
        "monoculture wheat, semi-arid region"
    )
    result = intake_node(_state(SiteState(site_id="s1"), text))
    site = result["site"]

    assert site.soil_organic_carbon_pct is not None
    assert site.soil_organic_carbon_pct.value == 0.3
    assert site.soil_organic_carbon_pct.provenance == Provenance.USER_STATED

    # A qualitative statement stays qualitative. Coercing "low" into a
    # millimetre figure would put a number the user never gave into the
    # diagnosis, where limiting_factor would treat it as measured.
    assert site.annual_rainfall_mm is not None
    assert site.annual_rainfall_mm.band == "low"
    assert site.annual_rainfall_mm.value is None
    assert site.annual_rainfall_mm.confidence == Confidence.LOW
    assert site.annual_rainfall_mm.provenance == Provenance.USER_STATED

    assert site.crop == "wheat"
    assert site.land_use is not None and site.land_use.band == "cropland"


def test_qualitative_band_does_not_reach_climate_zone():
    """A band must not masquerade as a measurement anywhere downstream."""
    result = intake_node(_state(SiteState(site_id="s1"), "rainfall is low"))
    assert result["site"].climate_zone() is None


# ============================= 2. coordinates =============================


def test_coordinate_intake_sets_lat_lon_and_acquire_runs():
    result = intake_node(_state(SiteState(site_id="s2"), "17.85, 75.42"))
    site = result["site"]
    assert (site.lat, site.lon) == (17.85, 75.42)

    # acquire is reached because lat/lon are present. Offline and with the
    # committed cache in place it fills what it can and records the rest as
    # unavailable; either way it returns rather than blocking.
    acquired = nodes.acquire_node({**result, "site": site})
    assert isinstance(acquired, dict)
    if acquired:
        assert acquired["site"].lat == 17.85


def test_acquire_without_coordinates_is_a_no_op():
    result = nodes.acquire_node(_state(SiteState(site_id="s2b")))
    assert result == {}


# ========================= 3. the brief's example =========================


def test_biodiversity_alone_produces_a_question_not_a_recommendation():
    """The brief's own example. One vague sentence is not a site."""
    intake = intake_node(_state(SiteState(site_id="s3"), "Biodiversity is declining on my land"))
    site = intake["site"]
    assert site.known() == [], "nothing measurable should have been invented from that sentence"

    result = gap_analysis_node({**_state(site), "site": site})
    assert result["pending_question"] is not None
    assert result["asked_about"], "the field asked about must be recorded"


# ====================== 4. value of information: slope ======================


def test_slope_is_high_value_information_on_the_deccan_site(graph):
    """The concrete case from the spec.

    contour_bunding's edges are gated on slope 2-25%. With slope unknown
    Conditions.satisfaction scores 0.5, halving the effect; supplying a
    slope inside the band roughly doubles it. So slope should be near the
    top of the value-of-information ranking on this site.
    """
    site = _deccan()
    unknown = [f for f in site.missing() if f in ASKABLE_FIELDS]
    assert "slope_pct" in unknown

    churn = value_of_information(graph, site, unknown)
    ordered = sorted(churn, key=lambda f: -churn[f])
    assert "slope_pct" in ordered[:2], f"slope not in the top two: {churn}"
    assert churn["slope_pct"] >= CHURN_FLOOR


def test_supplying_slope_changes_the_water_effect(graph):
    """The mechanism behind the churn, checked directly rather than assumed."""
    without = rank_interventions(graph, _deccan(), n=2000)
    with_slope = rank_interventions(graph, _deccan(slope=5.0), n=2000)

    bund_without = next(r for r in without if r.intervention == "contour_bunding")
    bund_with = next(r for r in with_slope if r.intervention == "contour_bunding")

    water_without = bund_without.effects["plant_available_water"].p50
    water_with = bund_with.effects["plant_available_water"].p50
    assert water_with > water_without * 1.5


# ============================ 5. never re-asks ============================


def test_a_field_in_asked_about_is_never_asked_again():
    site = _deccan()
    asked: list[str] = []
    questions: list[str] = []

    for _ in range(3):
        state = {**_state(site), "site": site, "asked_about": list(asked)}
        result = gap_analysis_node(state)
        question = result.get("pending_question")
        if question is None:
            break
        new = result["asked_about"]
        assert len(new) == 1
        assert new[0] not in asked, f"{new[0]} was asked twice"
        asked.extend(new)
        questions.append(question)

    assert len(set(asked)) == len(asked)


# ========================= 6. one question per turn =========================


def test_gap_analysis_asks_at_most_one_question_per_turn():
    site = _deccan()
    result = gap_analysis_node({**_state(site), "site": site})
    assert len(result.get("asked_about", [])) <= 1
    question = result.get("pending_question")
    if question is not None:
        # One question mark in the question body, so two questions have not
        # been packed into one turn. The value-of-information preface is a
        # statement, not a question.
        assert question.count("?") <= 2


# ============================= 7. stops asking =============================


def test_after_max_questions_gap_analysis_routes_to_diagnose():
    site = SiteState(site_id="s7")
    asked = ["soil_organic_carbon_pct", "annual_rainfall_mm", "ph"]
    assert len(asked) == MAX_QUESTIONS
    result = gap_analysis_node({**_state(site), "site": site, "asked_about": asked})
    assert result["pending_question"] is None


def test_stops_asking_when_nothing_left_would_change_the_advice():
    """A fully specified site has nothing worth a turn."""
    site = _deccan(slope=5.0)
    site.clay_pct = Measurement(
        value=45, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
    )
    site.sand_pct = Measurement(
        value=28, unit="%", provenance=Provenance.API_SOILGRIDS, confidence=Confidence.MODERATE
    )
    site.mean_temperature_c = Measurement(
        value=26.6, unit="C", provenance=Provenance.API_NASA_POWER, confidence=Confidence.MODERATE
    )
    result = gap_analysis_node({**_state(site), "site": site})
    assert result["pending_question"] is None


# ================================ 8. critic ================================


def _drafted_state(graph, site: SiteState):
    """A state carrying a real rendered draft, ready for the critic."""
    state = _state(site)
    state["site"] = site
    state["diagnosis"] = limiting_factor(site)
    state["ranked"] = rank_interventions(graph, site)
    return {**state, **nodes.synthesise_node(state)}


def test_critic_marks_a_fabricated_number_unsupported_and_drops_it(graph):
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)

    fabrication = "Soil organic carbon rises by +47.2% within one season."
    doctored = state["draft"].replace(
        "Impacted metrics:", f"{fabrication}\nImpacted metrics:", 1
    )
    assert fabrication in doctored

    result = critic_node({**state, "draft": doctored})
    claims = result["claims"]

    fabricated = [c for c in claims if "+47.2%" in c.text]
    assert fabricated, "the injected figure was not decomposed into a claim"
    assert all(c.kind == "quantitative" for c in fabricated)
    assert all(c.supported is False for c in fabricated)
    assert all("not traceable" in (c.support_note or "") for c in fabricated)

    revised = apply_withdrawals(doctored, claims, state["ranked"])
    assert "+47.2%" not in revised.split("WITHDRAWN CLAIMS")[0], (
        "the fabricated figure survived the revision"
    )
    assert "WITHDRAWN CLAIMS" in revised


def test_a_genuine_propagated_figure_is_traceable(graph):
    """The complement of the test above, so it is not passing by accident."""
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)
    claims = decompose(state["draft"])
    quantitative = [c for c in claims if c.kind == "quantitative"]
    assert quantitative

    traceable = nodes._traceable_figures(state["ranked"])
    for claim in quantitative:
        figures = nodes._figures(claim.text)
        assert figures <= traceable, f"rendered figure not traceable to the graph: {claim.text}"


def test_tier_statement_is_checked_for_traceability_not_for_citation(graph):
    """A tier assignment is a statement about this system's own gate.

    Its figure must be traceable to the propagation result, and it must not
    be asked for a citation: no paper in the corpus discusses this system's
    tiering, so demanding one would withdraw a correct statement for want of
    a citation that cannot exist.
    """
    site = _deccan(slope=5.0)
    ranked = rank_interventions(graph, site)
    figure = nodes.render.pct(next(r for r in ranked if r.tier == 1).constraint_movement)

    honest = Claim(
        text=f"Tier 1: addresses plant available water ({figure} on it).",
        kind="quantitative",
        source_ids=["FAO_2017_VGSSM"],
    )
    assert nodes.verify_claim(honest, ranked, site).supported is True

    invented = Claim(
        text="Tier 1: addresses plant available water (+91.4% on it).",
        kind="quantitative",
        source_ids=["FAO_2017_VGSSM"],
    )
    verified = nodes.verify_claim(invented, ranked, site)
    assert verified.supported is False
    assert "not traceable" in verified.support_note


def test_one_unsupported_mechanism_sentence_does_not_void_the_paragraph():
    """The mark goes on the sentence, not on every claim beside it."""
    line = "Why it works: Sentence one is fine. Sentence two has no support at all."
    claims = [
        Claim(
            text="Sentence two has no support at all.",
            kind="causal",
            source_ids=["FAO_2017_VGSSM"],
            supported=False,
            support_note="unsupported: nothing retrieved.",
        )
    ]
    revised = apply_withdrawals(line, claims, [])
    body = revised.split("WITHDRAWN CLAIMS")[0]
    assert "Sentence one is fine. [unverified" not in body
    assert "Sentence two has no support at all. [unverified" in body


def test_claim_with_no_registered_source_is_refused(graph):
    site = _deccan(slope=5.0)
    ranked = rank_interventions(graph, site)
    claim = Claim(text="Cover crops raise yields by +2.0%.", kind="citation", source_ids=[])
    verified = nodes.verify_claim(claim, ranked, site)
    assert verified.supported is False
    assert "cites no source" in verified.support_note


# =========================== 9. grounding coverage ===========================


def test_grounding_coverage_is_computed_and_in_range(graph):
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)
    result = critic_node(state)
    coverage = result["grounding_coverage"]
    assert coverage is not None
    assert 0.0 <= coverage <= 1.0

    scored = [c for c in result["claims"] if c.kind in nodes.GROUNDED_KINDS]
    grounded = [c for c in scored if c.category in ("traceable", "entailed")]
    assert coverage == pytest.approx(len(grounded) / len(scored))


def test_every_empirical_claim_lands_in_exactly_one_category(graph):
    """The breakdown has to account for the denominator, or it is decoration."""
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)
    result = critic_node(state)

    scored = [c for c in result["claims"] if c.kind in nodes.GROUNDED_KINDS]
    categories = ["traceable", "entailed", "softened", "corpus_gap"]
    counts = {name: sum(1 for c in scored if c.category == name) for name in categories}
    assert sum(counts.values()) == len(scored), f"claims fell outside the breakdown: {counts}"
    assert all(c.category is not None for c in scored)
    # Qualitative prose is deliberately outside the denominator.
    assert all(c.category is None for c in result["claims"] if c.kind == "qualitative")


def test_coverage_report_separates_corpus_gaps_from_failed_checks(graph):
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)
    result = critic_node(state)

    line = nodes.coverage_report(result["claims"], result["grounding_coverage"])
    assert "traceable to propagation:" in line
    assert "entailed by retrieved passage:" in line
    assert "unsupported, softened:" in line
    assert "no corpus evidence available:" in line
    assert "corpus gap, not a failed check" in line


def test_a_traceable_figure_is_not_sent_to_retrieval(graph, monkeypatch):
    """The largest share of the latency saving, asserted rather than assumed.

    A propagated figure cannot appear in any passage, so asking retrieval
    about it spends a cross-encoder pass to learn nothing. If any traceable
    claim reaches search at all, this test fails.
    """
    site = _deccan(slope=5.0)
    ranked = rank_interventions(graph, site)
    figure = nodes.render.pct(ranked[0].effects["plant_available_water"].p50)

    def forbidden(*args, **kwargs):
        raise AssertionError("a traceable claim was sent to retrieval")

    monkeypatch.setattr(nodes, "search", forbidden)

    claim = Claim(
        text=f"Contour bunding changes plant available water by {figure} at this site.",
        kind="quantitative",
        source_ids=["FAO_2017_VGSSM"],
    )
    verified = nodes.verify_claims([claim], ranked, site)[0]
    assert verified.supported is True
    assert verified.category == "traceable"
    assert "traceable to the propagation result" in verified.support_note


# =========================== 10. belief revision ===========================


def test_belief_revision_reports_a_rank_diff(graph):
    """Rainfall moves from a qualitative band to a number.

    That is the interesting case: with rainfall as a band the site has no
    climate zone and the sub-500mm limiting-factor rule cannot fire, so the
    diagnosis changes when the number arrives, and the ranking with it.
    """
    site = SiteState(
        site_id="revision_test",
        soil_organic_carbon_pct=Measurement(
            value=0.35, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            band="low", provenance=Provenance.USER_STATED, confidence=Confidence.LOW
        ),
        land_use=Measurement(
            band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
        slope_pct=Measurement(
            value=5, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
    )

    intake = intake_node(_state(site, "rainfall is actually 340mm"))
    assert "annual_rainfall_mm" in intake["revised_fields"]
    revised_site = intake["site"]
    assert revised_site.annual_rainfall_mm.value == 340

    state = {
        **_state(revised_site),
        **intake,
        "ranked": rank_interventions(graph, revised_site),
    }
    result = nodes.belief_revision_node(state)
    diff = result["belief_diff"]

    assert diff is not None
    assert "BELIEF REVISION" in diff
    assert "340" in diff
    assert "moved" in diff, f"no intervention was reported as moving rank:\n{diff}"
    assert "rank" in diff


def test_a_revision_survives_a_clarifying_question(graph):
    """A correction followed by a question must still produce its diff.

    The question pushes another SiteState onto site_history, so taking the
    diff baseline as the last history entry would compare the new ranking
    with itself and report no change. The baseline is pinned at the
    correction instead, and cleared only once the diff has been reported.
    """
    site = SiteState(
        site_id="revision_across_question",
        soil_organic_carbon_pct=Measurement(
            value=0.35, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.MODERATE
        ),
        annual_rainfall_mm=Measurement(
            band="low", provenance=Provenance.USER_STATED, confidence=Confidence.LOW
        ),
        land_use=Measurement(
            band="cropland", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
    )

    correction = intake_node(_state(site, "rainfall is actually 340mm"))
    assert correction["revised_fields"] == ["annual_rainfall_mm"]
    assert correction["revision_baseline"] is not None

    # The agent then asks about slope, and the answer arrives as a new turn.
    answer = intake_node({**_state(correction["site"]), **correction, "messages": [
        {"role": "user", "content": "slope is about 5%"}
    ]})
    # The rainfall correction is still owed a diff, and the baseline still
    # points at the state before it.
    assert answer["revised_fields"] == ["annual_rainfall_mm"]
    assert answer["revision_baseline"].annual_rainfall_mm.band == "low"

    state = {
        **_state(answer["site"]),
        **answer,
        "ranked": rank_interventions(graph, answer["site"]),
    }
    result = nodes.belief_revision_node(state)
    assert result["belief_diff"] is not None
    assert "340" in result["belief_diff"]
    # Cleared, so the same correction is not re-reported next turn.
    assert result["revised_fields"] == []
    assert result["revision_baseline"] is None


def test_belief_revision_is_silent_when_nothing_was_revised(graph):
    site = _deccan(slope=5.0)
    state = {
        **_state(site),
        "revised_fields": [],
        "site_history": [site],
        "ranked": rank_interventions(graph, site),
    }
    assert nodes.belief_revision_node(state)["belief_diff"] is None


# =========================== 11. site persistence ===========================


def test_site_profile_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agents.state.SITES_DIR", tmp_path)
    site = _deccan(slope=5.0)
    path = save_site_profile(site)
    assert path.exists()

    loaded = load_site_profile(site.site_id)
    assert loaded is not None
    assert loaded.model_dump() == site.model_dump()
    assert loaded.soil_organic_carbon_pct.provenance == Provenance.USER_STATED


def test_missing_site_profile_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agents.state.SITES_DIR", tmp_path)
    assert load_site_profile("never_seen") is None


def test_site_id_that_would_escape_the_profile_directory_is_refused():
    with pytest.raises(ValueError, match="must contain only"):
        site_profile_path("../../etc/passwd")


# ========================== 12. critic does not loop ==========================


def test_critic_passes_never_exceed_the_cap(graph):
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)

    for _ in range(5):
        result = critic_node(state)
        state = {**state, **result, **nodes.synthesise_node({**state, **result})}
        assert state["critic_passes"] <= MAX_CRITIC_PASSES
    assert state["critic_passes"] <= MAX_CRITIC_PASSES


def test_critic_routing_reads_the_critics_own_decision(graph):
    """The router must not re-derive the cap from the pass counter.

    It did, and the two disagreed on the final pass: critic_node evaluated
    the predicate before its own increment and concluded it would revise
    again, the router evaluated it after and ended the run, and the grounding
    block was appended by neither.
    """
    from src.agents.graph import _after_critic

    assert _after_critic({"revision_pending": True}) == "synthesise"
    assert _after_critic({"revision_pending": False}) == "__end__"
    # Falls back to the claim-level check for a hand-assembled state.
    unsupported = [Claim(text="x", kind="citation", source_ids=[], supported=False)]
    assert _after_critic({"claims": unsupported}) == "synthesise"
    assert _after_critic({"claims": []}) == "__end__"


def test_the_final_draft_carries_the_grounding_block(graph):
    """The last critic pass has to report coverage on the draft it checked."""
    site = _deccan(slope=5.0)
    state = _drafted_state(graph, site)

    # Drive the loop the way the graph does, until the critic stops asking.
    for _ in range(MAX_CRITIC_PASSES + 1):
        result = critic_node(state)
        state = {**state, **result}
        if not nodes.needs_revision(state):
            break
        state = {**state, **nodes.synthesise_node(state)}

    assert state["critic_passes"] <= MAX_CRITIC_PASSES
    assert "GROUNDING:" in state["draft"], "the final draft did not report its coverage"
    assert "traceable to propagation:" in state["draft"]


# ======================= 13. API failure degrades well =======================


def test_graph_completes_when_acquire_raises(monkeypatch, graph):
    """An exploding API must cost the user a question, not the answer."""

    def exploding(url: str, timeout: float):
        raise OSError("network is down")

    monkeypatch.setattr(nodes, "_get_json", exploding)
    monkeypatch.setattr(nodes, "_cached", lambda api, lat, lon: None)
    monkeypatch.setenv("DARUKAA_OFFLINE", "")

    site = SiteState(site_id="degraded", lat=17.85, lon=75.42)
    result = nodes.acquire_node(_state(site))
    # Nothing recorded, and the failure is stated rather than swallowed.
    assert all("returned nothing" in note for note in result.get("notes", []))
    assert result.get("site", site).known() == []

    gap = gap_analysis_node({**_state(site), "site": site})
    assert gap["pending_question"] is not None, "gap analysis should ask for what the API could not give"


def test_conversation_reaches_a_recommendation_and_records_a_question():
    """One end-to-end pass through the compiled graph, offline.

    Interrupts are the mechanism, so this checks the interrupt arrives and
    that resuming it carries the answer back into intake rather than
    restarting the conversation.
    """
    conversation = Conversation(SiteState(site_id="e2e_test"), thread_id="test-e2e")

    first = conversation.send("Biodiversity is declining on my land")
    question = conversation.question(first)
    assert question is not None, "a bare complaint should not yield a recommendation"

    second = conversation.send("soil organic carbon 0.35%, rainfall 340mm, wheat, slope 5%")
    state = conversation.state()
    assert state["site"].soil_organic_carbon_pct.value == 0.35
    assert state["turn"] >= 2
    assert len(state["asked_about"]) == len(set(state["asked_about"]))

    if conversation.question(second) is None:
        assert state["draft"]
        assert "RECOMMENDATION 1" in state["draft"]
        assert state["grounding_coverage"] is not None


def test_graph_compiles_with_the_specified_topology():
    app = build_agent_graph()
    nodes_present = set(app.get_graph().nodes)
    for name in (
        "intake",
        "acquire",
        "gap_analysis",
        "ask",
        "diagnose",
        "plan",
        "belief_revision",
        "bind",
        "synthesise",
        "critic",
    ):
        assert name in nodes_present


# ========================= 9. greetings and meta =========================


def test_a_greeting_gets_a_reply_and_no_analysis():
    """A greeting must not cost a propagation run."""
    conversation = Conversation(SiteState(site_id="app_greet1"), thread_id="greet1")
    result = conversation.send("hi")

    assert result.get("ranked") is None, "a greeting must not run the ranking"
    assert conversation.question(result) is None, "a greeting is not a clarifying question"
    draft = result.get("draft") or ""
    assert draft, "a greeting must still get an answer"
    assert "RECOMMENDATION" not in draft and "METHODOLOGY" not in draft
    assert "soil organic carbon" in draft, "the reply should say what the system needs"


def test_a_greeting_on_a_loaded_site_acknowledges_it_rather_than_re_analysing():
    site = SiteState(
        site_id="app_greet2",
        soil_organic_carbon_pct=Measurement(
            value=0.4, unit="%", provenance=Provenance.USER_STATED, confidence=Confidence.HIGH
        ),
    )
    conversation = Conversation(site, thread_id="greet2")
    result = conversation.send("hello")

    assert result.get("ranked") is None
    draft = result.get("draft") or ""
    assert "soil organic carbon" in draft, "the reply should name what is already on file"
    assert "RECOMMENDATION" not in draft


def test_a_greeting_carrying_a_measurement_is_treated_as_data():
    """The gate is what was parsed, not what the sentence opens with."""
    conversation = Conversation(SiteState(site_id="app_greet3"), thread_id="greet3")
    conversation.send("hi, soil organic carbon is 0.4% and rainfall is 340mm")
    known = conversation.state()["site"].known()
    assert "soil_organic_carbon_pct" in known and "annual_rainfall_mm" in known


@pytest.mark.parametrize(
    "text",
    ["hi", "Hello!", "thanks", "what can you do?", "how does this work", "help", "ok"],
)
def test_smalltalk_patterns_match(text):
    assert nodes.is_smalltalk(text)


@pytest.mark.parametrize(
    "text",
    [
        "Biodiversity is declining on my land",
        "soil organic carbon 0.35%, rainfall low, wheat monoculture",
        "17.85, 75.42",
        "rainfall is actually 340mm",
    ],
)
def test_site_descriptions_are_not_smalltalk(text):
    assert not nodes.is_smalltalk(text)
