"""The LangGraph state machine, its checkpointer, and a replayable demo.

    intake -> smalltalk    when the message is a greeting or meta question
    smalltalk -> END
    intake -> acquire -> gap_analysis
    gap_analysis -> ask        when a question is due; ask interrupts
    ask -> intake              the answer re-enters as a user message
    gap_analysis -> diagnose   when nothing further is worth asking
    diagnose -> plan -> belief_revision -> bind -> synthesise -> critic
    critic -> synthesise       when a claim could not be supported
    critic -> END              otherwise

A clarifying question is a LangGraph interrupt rather than a loop inside a
node, so the conversation stops with its state checkpointed and resumes on
the user's answer. belief_revision sits between plan and bind because the
diff it reports is between the previous ranking and the new one, so it needs
plan to have run, and its text goes into the draft that bind and synthesise
build.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from src.agents.nodes import (
    MAX_CRITIC_PASSES,
    coverage_report,
    acquire_node,
    bind_node,
    belief_revision_node,
    critic_node,
    diagnose_node,
    gap_analysis_node,
    intake_node,
    needs_revision,
    plan_node,
    smalltalk_node,
    synthesise_node,
)
from src.agents.state import Claim, ConversationState, initial_state
from src.graph.propagate import (
    PathContribution,
    PropagationResult,
    RankedIntervention,
    Tradeoff,
)
from src.graph.schemas import Confidence, Measurement, Provenance, SiteState
from src.retrieval.search import RetrievedChunk, warmup

# Room for three question-and-answer cycles through intake and acquire, the
# full diagnose-to-critic chain, and two critic revisions, without the
# default limit cutting a legitimate conversation short.
RECURSION_LIMIT = 80

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env() -> None:
    """Read provider credentials from .env into the environment.

    Called from main rather than at import, so importing the graph does not
    change the process environment underneath a caller that configured its
    own. Existing variables win. .env is gitignored, and no value read from
    it is logged or printed.
    """
    load_dotenv(ENV_PATH, override=False)

# The state carries our own Pydantic models, so the checkpointer is told
# exactly which types it may reconstruct rather than left permissive. The
# list has to be complete: a type left off it comes back from a checkpoint as
# a plain dict rather than raising, so a missing entry shows up later as an
# attribute error on a resumed turn. Enums are listed alongside the models
# that hold them for the same reason.
_CHECKPOINT_TYPES = (
    SiteState,
    Measurement,
    Confidence,
    Provenance,
    RankedIntervention,
    PropagationResult,
    PathContribution,
    Tradeoff,
    Claim,
    RetrievedChunk,
)


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)


def ask_node(state: ConversationState) -> dict:
    """Put the pending question to the user and stop until it is answered.

    The resumed value re-enters the graph as a user message so intake parses
    it the same way it parses anything else. There is no separate answer
    parser, which is deliberate: an answer to "how steep is the land" is the
    same kind of statement as an unprompted "the slope is about 5%", and two
    parsers would drift.
    """
    question = state.get("pending_question")
    answer = interrupt({"question": question})
    return {
        "messages": [
            {"role": "assistant", "content": question},
            {"role": "user", "content": str(answer)},
        ],
        "pending_question": None,
    }


def _after_intake(state: ConversationState) -> str:
    """Greetings and meta questions skip the pipeline entirely.

    intake has already parsed the message by this point, and it sets the
    reply only when nothing measurable came out of it, so this cannot divert
    a turn that carried site data.
    """
    return "smalltalk" if state.get("smalltalk_reply") else "acquire"


def _after_gap_analysis(state: ConversationState) -> str:
    return "ask" if state.get("pending_question") else "diagnose"


def _after_critic(state: ConversationState) -> str:
    # The cap is enforced inside critic_node, which records its decision, so
    # this reads that decision rather than re-deriving it from a count it
    # would see at a different moment.
    return "synthesise" if needs_revision(state) else END


def build_agent_graph(checkpointer: MemorySaver | None = None):
    """Compile the state machine.

    The checkpointer is a MemorySaver keyed on thread_id, which is the
    working-memory tier: an interrupted conversation resumes with everything
    it had established. Persistent memory is separate and lives in
    data/derived/sites.
    """
    builder = StateGraph(ConversationState)

    builder.add_node("intake", intake_node)
    builder.add_node("smalltalk", smalltalk_node)
    builder.add_node("acquire", acquire_node)
    builder.add_node("gap_analysis", gap_analysis_node)
    builder.add_node("ask", ask_node)
    builder.add_node("diagnose", diagnose_node)
    builder.add_node("plan", plan_node)
    builder.add_node("belief_revision", belief_revision_node)
    builder.add_node("bind", bind_node)
    builder.add_node("synthesise", synthesise_node)
    builder.add_node("critic", critic_node)

    builder.set_entry_point("intake")
    builder.add_conditional_edges(
        "intake", _after_intake, {"smalltalk": "smalltalk", "acquire": "acquire"}
    )
    builder.add_edge("smalltalk", END)
    builder.add_edge("acquire", "gap_analysis")
    builder.add_conditional_edges(
        "gap_analysis", _after_gap_analysis, {"ask": "ask", "diagnose": "diagnose"}
    )
    builder.add_edge("ask", "intake")
    builder.add_edge("diagnose", "plan")
    builder.add_edge("plan", "belief_revision")
    builder.add_edge("belief_revision", "bind")
    builder.add_edge("bind", "synthesise")
    builder.add_edge("synthesise", "critic")
    builder.add_conditional_edges("critic", _after_critic, {"synthesise": "synthesise", END: END})

    return builder.compile(checkpointer=checkpointer or MemorySaver(serde=_serializer()))


class Conversation:
    """One thread of conversation over a compiled graph.

    Thin on purpose: it owns the thread_id and whether the graph is currently
    parked on an interrupt, and nothing else. All state lives in the
    checkpointer.
    """

    def __init__(self, site: SiteState, thread_id: str = "default", app=None) -> None:
        self.app = app if app is not None else build_agent_graph()
        self.config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        self._site = site
        self._started = False
        self._waiting = False

    def send(self, text: str) -> dict:
        """One user turn. Returns the raw graph result."""
        if self._waiting:
            payload = Command(resume=text)
        else:
            payload = (
                {**initial_state(self._site), "messages": [{"role": "user", "content": text}]}
                if not self._started
                else {"messages": [{"role": "user", "content": text}]}
            )
        result = self.app.invoke(payload, self.config)
        self._started = True
        self._waiting = bool(result.get("__interrupt__"))
        return result

    def question(self, result: dict) -> str | None:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return None
        value = interrupts[0].value
        return value.get("question") if isinstance(value, dict) else str(value)

    def state(self) -> ConversationState:
        return self.app.get_state(self.config).values


DEMO_TURNS = [
    "Biodiversity is declining on my land",
    "soil organic carbon 0.35%, rainfall low, wheat monoculture",
    "17.85, 75.42",
    "rainfall is actually 340mm",
    # The fifth turn is not extra scope: it answers the question the agent
    # asks on the fourth. Once rainfall stops being the band "low" and
    # becomes 340mm, the site is semi-arid, plant available water becomes the
    # binding constraint, and slope decides whether contour bunding does
    # anything, so slope goes from worthless to the highest-value unknown.
    # The agent asks, and the belief revision diff arrives with the answer.
    "slope is about 5%",
]


def run_demo(thread_id: str = "demo") -> None:
    # Load the embedding model, the cross-encoder and the indexes before the
    # conversation starts. They are lazily opened on first use, so without
    # this the first turn that retrieves anything absorbs about forty seconds
    # of model load and reports it as thinking time. The cost is real either
    # way; charging it to startup rather than to a turn is the difference
    # between a slow launch and an agent that appears to hang mid-sentence.
    started = time.perf_counter()
    warmup()
    print(f"(warmed retrieval models and indexes in {time.perf_counter() - started:.1f}s)\n")

    conversation = Conversation(SiteState(site_id="demo_deccan"), thread_id=thread_id)

    for index, text in enumerate(DEMO_TURNS, start=1):
        print("=" * 78)
        print(f"TURN {index} - USER: {text}")
        print("=" * 78)
        started = time.perf_counter()
        result = conversation.send(text)
        elapsed = time.perf_counter() - started

        question = conversation.question(result)
        if question is not None:
            print(f"\nAGENT (clarifying question, {elapsed:.1f}s):")
            print(question)
            print()
            continue

        draft = result.get("draft")
        print(f"\nAGENT ({elapsed:.1f}s):")
        print(draft if draft else "(no draft produced)")
        print()

    final = conversation.state()
    print("=" * 78)
    print("CONVERSATION STATE AT CLOSE")
    print("=" * 78)
    print(f"turns: {final.get('turn')}")
    print(f"asked about: {final.get('asked_about')}")
    print(f"critic passes: {final.get('critic_passes')}")
    coverage = final.get("grounding_coverage")
    if coverage is None:
        print("grounding: none")
    else:
        print(coverage_report(final.get("claims") or [], coverage))
    print(f"episodic summary: {final.get('summary')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Darukaa.Earth agent graph")
    parser.add_argument("--demo", action="store_true", help="replay the scripted conversation")
    parser.add_argument("--thread-id", default="demo")
    args = parser.parse_args(argv)
    load_env()

    if not args.demo:
        parser.print_help()
        return 1
    run_demo(args.thread_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
