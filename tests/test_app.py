"""Tests for the Streamlit wrapper's session handling.

Only the parts that hold conversation state are covered here. Rendering is
not: it is Streamlit's job, and asserting on widget calls would test the
framework rather than this app.

Offline, like the rest of the suite. DARUKAA_OFFLINE is set before app is
imported because importing it reads .env and configures provider
credentials, and a unit test must not be able to reach a provider.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DARUKAA_OFFLINE", "1")

from src.agents.nodes import gap_analysis_node, intake_node
from src.agents.state import initial_state
from src.graph.schemas import SiteState

app = pytest.importorskip("app", reason="streamlit is not installed in this environment")


@pytest.fixture
def session(monkeypatch) -> dict:
    """A plain dict standing in for st.session_state.

    Streamlit's own session_state needs a script run context, which does not
    exist under pytest. It behaves as a mapping for everything this module
    does with it, so a dict is a faithful substitute here.
    """
    state: dict = {}
    monkeypatch.setattr(app.st, "session_state", state)
    return state


def test_reset_clears_the_site_not_just_the_transcript(session) -> None:
    """The bug this guards against.

    Loading a preset and then resetting used to leave site_id pointing at
    that preset, so the next conversation started from the preset's fully
    populated SiteState and answered the next message against the old site.
    """
    app.reset_session("western_ghats")
    loaded = app.new_conversation()._site
    assert loaded.known(), "the preset should arrive populated, or this proves nothing"

    app.reset_session()

    assert session["site_id"] != "western_ghats"
    fresh = app.new_conversation()._site
    assert fresh.known() == [], (
        "reset must clear the site, not only the chat history; "
        f"still populated with {fresh.known()}"
    )


def test_reset_starts_a_new_thread_and_empties_the_history(session) -> None:
    """A new thread_id is what abandons site_history and asked_about.

    Those accumulate inside the LangGraph checkpointer rather than in
    session_state, so they cannot be cleared directly: the reset drops the
    key they are stored under instead.
    """
    app.reset_session("deccan_semiarid")
    first_thread = session["thread_id"]
    session["history"] = [{"role": "user", "text": "soil organic carbon 0.35%"}]
    session["conversation"] = object()

    app.reset_session()

    assert session["thread_id"] != first_thread
    assert session["history"] == []
    assert "conversation" not in session


def test_a_vague_message_after_reset_asks_rather_than_diagnoses(session) -> None:
    """The brief's worked example, on the site a reset actually produces.

    test_agents covers this from a hand-built blank SiteState. This runs it
    from the site the app hands a new conversation after a preset has been
    loaded and cleared, which is the path that was broken.
    """
    app.reset_session("western_ghats")
    app.reset_session()
    site = app.new_conversation()._site

    intake = intake_node(
        {**initial_state(site), "messages": [{"role": "user", "content": "Biodiversity is declining on my land"}]}
    )
    parsed: SiteState = intake["site"]
    assert parsed.known() == [], "nothing measurable should have been invented from that sentence"

    result = gap_analysis_node({**initial_state(parsed), "site": parsed})
    assert result["pending_question"] is not None, (
        "a site with nothing measured must produce a clarifying question, not a diagnosis"
    )


def test_a_preset_is_copied_so_one_session_cannot_mutate_the_next(session) -> None:
    """DEMO_SITES is module level and intake writes into the SiteState it is given."""
    app.reset_session("deccan_semiarid")
    first = app.new_conversation()._site
    first.crop = "sorghum"

    app.reset_session("deccan_semiarid")
    second = app.new_conversation()._site
    assert second.crop != "sorghum", "the preset was handed over rather than copied"
