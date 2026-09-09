"""Conversation state and persistent site memory for the agent graph.

Three tiers of memory meet here:

  working     ConversationState, held by the LangGraph checkpointer and keyed
              on thread_id, so an interrupted conversation resumes exactly
              where it stopped.
  episodic    `summary`, a running precis regenerated every SUMMARY_EVERY
              turns so a long conversation does not grow the context without
              bound.
  persistent  a SiteState written to data/derived/sites/<site_id>.json, so a
              site is remembered across processes and sessions.

The state is a TypedDict rather than a Pydantic model because LangGraph
merges partial updates into it per node. Everything inside it that crosses a
module boundary is a Pydantic model.
"""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from src.graph.propagate import RankedIntervention
from src.graph.schemas import SiteState
from src.retrieval.search import RetrievedChunk

_ROOT = Path(__file__).resolve().parents[2]
SITES_DIR = _ROOT / "data" / "derived" / "sites"

# How often the episodic summary is regenerated, in turns. Rebuilding it every
# turn would spend a model call on a summary nobody reads yet; letting it run
# forever would let the transcript grow without bound.
SUMMARY_EVERY = 4

# Hard ceiling on critic revision passes. Two is enough for the loop to fix
# what one pass surfaced, and any more risks a critic and a renderer arguing
# with each other for the rest of the conversation.
MAX_CRITIC_PASSES = 2

_SAFE_SITE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class Claim(BaseModel):
    """One atomic assertion pulled out of a rendered draft.

    kind decides how it is verified, because the four kinds are not
    checkable the same way:
      quantitative  a number. Checked against the propagation result that
                    produced it, and not against retrieval. No passage in
                    the corpus contains a propagated Monte Carlo figure: a
                    published effect size and a model output composed along
                    a path are different objects, and asking retrieval about
                    the second gets silence that then reads as a failure.
                    The RELATIONSHIP behind the number is what the corpus is
                    asked to back, and that is the causal and citation claims
                    of the same recommendation.
      causal        a mechanism sentence. Checked by retrieval plus
                    entailment against the sources the edge cites.
      citation      a claim that a named source backs this recommendation.
                    Checked by retrieval restricted to that source.
      qualitative   framing, sequencing and instruction prose. Carries no
                    empirical assertion, so it is recorded and not scored.
    """

    text: str
    kind: Literal["quantitative", "causal", "citation", "qualitative"]
    source_ids: list[str]
    supported: bool | None = None
    support_note: str | None = None
    # How the verdict was reached, which is what makes a coverage figure
    # readable. The four are not degrees of the same thing:
    #   traceable   the figure is one the propagation engine produced for
    #               this site. Its provenance is the graph, and no passage
    #               could confirm it.
    #   entailed    a retrieved passage from a cited source supports it.
    #   softened    unsupported for a substantive reason: a figure from
    #               nowhere, an unregistered source, or a passage that does
    #               not bear the claim out.
    #   corpus_gap  retrieval found nothing above the support floor. That is
    #               a fact about this corpus, not about the claim, and it is
    #               reported separately so it is not read as a failure of
    #               rigour.
    category: Literal["traceable", "entailed", "softened", "corpus_gap"] | None = None


class Verdict(BaseModel):
    """Structured result of one entailment check."""

    verdict: Literal["supported", "partially", "unsupported"]
    reason: str


class ConversationState(TypedDict, total=False):
    """Working state for one conversation thread.

    messages, asked_about and site_history accumulate: their nodes return
    only what is new and the reducer appends, so a resumed thread does not
    lose what earlier turns established.
    """

    messages: Annotated[list[dict], operator.add]
    site: SiteState
    turn: int
    # Limiting factor and the reasoning behind it, straight from
    # propagate.limiting_factor.
    diagnosis: tuple[str, str] | None
    ranked: list[RankedIntervention] | None
    draft: str | None
    claims: list[Claim] | None
    critic_passes: int
    grounding_coverage: float | None
    pending_question: str | None
    asked_about: Annotated[list[str], operator.add]
    site_history: Annotated[list[SiteState], operator.add]

    # Beyond the spec's list, and each one earns its place:
    # passages bound to the rendered explanation, keyed "source->target", so
    # synthesise can print them and the critic can see what retrieval found.
    evidence: dict[str, list[RetrievedChunk]]
    # fields intake overwrote rather than filled, which is what distinguishes
    # a correction from new information. Accumulates across intake calls and
    # is cleared by belief_revision once it has reported the diff, because a
    # correction can be followed by a clarifying question and the diff is
    # still owed after the answer arrives.
    revised_fields: list[str]
    # the SiteState as it stood before the earliest unreported revision. The
    # diff baseline is held explicitly rather than taken as site_history[-1]
    # because a question between the correction and the answer pushes another
    # entry onto that history, and diffing against it would compare the new
    # ranking with itself.
    revision_baseline: SiteState | None
    # the rank diff produced by belief_revision, rendered into the draft.
    belief_diff: str | None
    # notes from intake and acquire that belong in the reply to the user
    # (what was parsed, what an API estimated, what it could not reach).
    notes: list[str]
    # claim texts the critic withdrew, so synthesise's next pass can remove
    # them and say that it did.
    withdrawn: list[str]
    # whether the critic asked for another synthesise pass. Recorded rather
    # than recomputed by the router, because the two evaluated the same
    # predicate at different points either side of the pass counter's
    # increment and disagreed on the final pass.
    revision_pending: bool
    # value-of-information table from the last gap analysis: field to churn.
    # Reported so the choice of question is inspectable rather than magic.
    voi: dict[str, float]
    # episodic memory: a running precis of the conversation so far, and the
    # turn it was last rebuilt on.
    summary: str | None
    summary_turn: int


def initial_state(site: SiteState) -> ConversationState:
    return ConversationState(
        messages=[],
        site=site,
        turn=0,
        diagnosis=None,
        ranked=None,
        draft=None,
        claims=None,
        critic_passes=0,
        grounding_coverage=None,
        pending_question=None,
        asked_about=[],
        site_history=[],
        evidence={},
        revised_fields=[],
        revision_baseline=None,
        belief_diff=None,
        notes=[],
        withdrawn=[],
        voi={},
        summary=None,
        summary_turn=0,
    )


def site_profile_path(site_id: str) -> Path:
    """Where a site's persistent profile lives.

    site_id becomes a filename, so it is validated rather than trusted: it
    arrives from user input, and a path separator or a parent reference in it
    would write outside the profile directory.
    """
    if not _SAFE_SITE_ID.match(site_id):
        raise ValueError(
            f"site_id {site_id!r} must contain only letters, digits, underscore, dot or "
            "hyphen; it is used as a filename"
        )
    return SITES_DIR / f"{site_id}.json"


def save_site_profile(site: SiteState) -> Path:
    """Write a site to persistent memory and return where it went."""
    path = site_profile_path(site.site_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(site.model_dump_json()), indent=2))
    return path


def load_site_profile(site_id: str) -> SiteState | None:
    """Read a site back, or None when it has never been seen.

    A corrupt or schema-violating profile raises rather than being silently
    discarded: a site remembered wrongly is worse than a site not remembered.
    """
    path = site_profile_path(site_id)
    if not path.exists():
        return None
    return SiteState.model_validate_json(path.read_text())
