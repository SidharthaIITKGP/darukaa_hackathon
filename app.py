"""Streamlit front end for the Darukaa.Earth AI environmental scientist.

A thin wrapper over src.agents.graph. Nothing here computes an effect size,
ranks an intervention, or builds a citation string: every number and every
citation on screen is produced by the propagation engine and the render
module that the command line demo also uses, so the interface and the demo
cannot drift apart.

What the interface does add is visibility. The causal chain behind each
recommendation and the passages retrieved for each edge are put on screen in
expanders rather than left in the log, because the reasoning is the thing
being assessed here and a chat transcript alone hides it.

Deployment notes, which drive most of the structure below:

  Memory   Streamlit Community Cloud gives roughly 1GB. The precise
           cross-encoder, bge-reranker-v2-m3, is 2.2GB on its own and cannot
           load there at all, so precise reranking is never requested. With
           DARUKAA_LOW_MEMORY=1 no cross-encoder loads at all and retrieval
           falls back to the fused RRF order.
  Locking  A local Qdrant holds an exclusive lock on its storage directory,
           and Streamlit serves concurrent sessions from one process. Every
           model and index is therefore built inside @st.cache_resource so
           one instance is shared per process, and the lock error is caught
           and explained rather than shown as a traceback.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st

st.set_page_config(page_title="Darukaa.Earth", layout="wide")


# ============================== configuration ==============================

# Set before anything under src is imported. The retrieval layer reads
# DARUKAA_LOW_MEMORY on each call rather than at import, but the model
# credentials are read by litellm at call time from os.environ only, and
# st.secrets is not visible to it.
def _secret(name: str, default: str | None = None) -> str | None:
    """Read a setting from st.secrets, falling back to the environment.

    st.secrets raises rather than returning empty when no secrets file
    exists, which is the normal case when running locally against .env, so
    the lookup is guarded.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - no secrets file is not an error here
        pass
    return os.environ.get(name, default)


def _configure_env() -> None:
    # The deployed build has no .env, so the key arrives through st.secrets.
    # Loading .env as well means a local run needs no secrets file.
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)

    groq_key = _secret("GROQ_API_KEY")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    # A Groq key is useless unless the configured model is a Groq model, and
    # the default in nodes.py is Anthropic. Set together or not at all.
    model = _secret("DARUKAA_MODEL")
    if model:
        os.environ["DARUKAA_MODEL"] = model
    elif groq_key and "DARUKAA_MODEL" not in os.environ:
        os.environ["DARUKAA_MODEL"] = "groq/openai/gpt-oss-120b"

    low_memory = _secret("DARUKAA_LOW_MEMORY")
    if low_memory:
        os.environ["DARUKAA_LOW_MEMORY"] = low_memory


_configure_env()

from src.agents import render  # noqa: E402
from src.agents.graph import Conversation, build_agent_graph  # noqa: E402
from src.agents.nodes import RENDER_TOP_N, coverage_report, model_name  # noqa: E402
from src.demo import DEMO_SITES  # noqa: E402
from src.graph.edges import build_graph  # noqa: E402
from src.graph.propagate import (  # noqa: E402
    DELTA_REF,
    PRIORITY_TOLERANCE,
    TIER_1_MIN_EFFECT,
    TRANSMISSION,
)
from src.retrieval.search import RERANK_FLOOR, low_memory  # noqa: E402

# Opening lines for the three preset sites. Each states the concern in the
# user's own terms plus the measurements a landholder would actually know,
# which is what the intake node parses.
PRESET_PROMPTS: dict[str, tuple[str, str]] = {
    "deccan_semiarid": (
        "Deccan, semi-arid",
        "Biodiversity is declining on my land. Soil organic carbon 0.35%, "
        "rainfall 340mm, pH 8.1, wheat monoculture, slope about 5%. "
        "Coordinates 17.85, 75.42.",
    ),
    "indo_gangetic": (
        "Indo-Gangetic plain",
        "My yields are falling and the soil is getting harder. Soil organic "
        "carbon 0.55%, rainfall 620mm, pH 7.6, rice-wheat rotation, flat "
        "land. Coordinates 29.15, 76.32.",
    ),
    "western_ghats": (
        "Western Ghats, steep",
        "I am losing soil off the slopes every monsoon. Soil organic carbon "
        "1.2%, rainfall 2100mm, pH 6.2, slope 12%, cropland. "
        "Coordinates 15.60, 74.05.",
    ),
}


# ============================ cached resources =============================


@st.cache_resource(show_spinner="Building the causal graph...")
def causal_graph():
    """The curated causal graph, built once per process."""
    return build_graph()


@st.cache_resource(show_spinner="Compiling the agent graph...")
def agent_app():
    """The compiled LangGraph state machine, shared across sessions.

    Safe to share: all conversation state lives in the checkpointer and is
    keyed on thread_id, and every session below gets its own thread_id.
    """
    return build_agent_graph()


@st.cache_resource(show_spinner="Loading embedding model and search indexes...")
def retrieval_status() -> tuple[bool, str | None]:
    """Open the indexes and load the models once, reporting failure as data.

    Returns (ready, message). The Qdrant lock is the failure worth naming:
    another process holding the storage directory produces an error that
    says nothing useful to a user looking at a chat box, and a traceback in
    its place would be worse. Cached, so the cost is paid once per process
    and a failure is not retried on every rerun.
    """
    from src.retrieval.search import warmup

    try:
        warmup(precise=False)
    except Exception as error:  # noqa: BLE001 - reported to the user verbatim below
        text = str(error)
        lowered = text.lower()
        if "already accessed" in lowered or "lock" in lowered:
            return False, (
                "The search index is locked by another process. Local Qdrant allows "
                "one reader at a time, so a demo script or a second app instance "
                "running against data/derived/qdrant will hold it. Stop that process "
                "and reload this page."
            )
        if "missing" in lowered or "does not exist" in lowered:
            return False, (
                f"The search index could not be opened: {text} Build it with "
                "`python -m src.retrieval.index`."
            )
        return False, f"Retrieval could not start: {text}"
    return True, None


def new_conversation() -> Conversation:
    """One conversation thread for this browser session."""
    site_id = st.session_state.get("site_id", "app_site")
    return Conversation(
        DEMO_SITES.get(site_id) or _blank_site(site_id),
        thread_id=st.session_state["thread_id"],
        app=agent_app(),
    )


def _blank_site(site_id: str):
    from src.graph.schemas import SiteState

    return SiteState(site_id=site_id)


# ================================ rendering ================================


def recommendation_blocks(state: dict) -> list[dict]:
    """Pull the rendered recommendations plus their reasoning out of state.

    Each block carries the rendered text, the causal chain that text
    narrates, and the passages bound to each edge on that chain. All three
    come from the same explanatory path, so the citations shown under a
    recommendation are the ones its mechanism sentence actually rests on.
    """
    ranked = state.get("ranked") or []
    diagnosis = state.get("diagnosis")
    if not ranked or not diagnosis:
        return []

    graph = causal_graph()
    limiting_var = diagnosis[0]
    evidence = state.get("evidence") or {}

    blocks: list[dict] = []
    for index, item in enumerate(ranked[:RENDER_TOP_N], start=1):
        items = render.reportable(item)
        path = render.explanatory_path(items)
        edges = render.path_edges(graph, path) if path else []

        chain: list[dict] = []
        for edge in edges:
            key = f"{edge.source}->{edge.target}"
            chain.append(
                {
                    "key": key,
                    "source": render.label(edge.source),
                    "target": render.label(edge.target),
                    "sign": edge.sign,
                    "metric": edge.metric.value,
                    "effect": edge.effect,
                    "lag_years": edge.lag_years,
                    "strength": edge.strength.value,
                    "confidence": edge.confidence.value,
                    "mechanism": edge.mechanism,
                    "contested": edge.contested,
                    "contested_note": edge.contested_note,
                    "refs": [(ref.source_id, ref.role) for ref in edge.evidence],
                    "chunks": evidence.get(key, []),
                }
            )

        blocks.append(
            {
                "index": index,
                "intervention": item.intervention,
                "text": "\n".join(
                    render.render_recommendation(graph, index, item, limiting_var)
                ),
                "path": [render.label(node) for node in path],
                "chain": chain,
            }
        )
    return blocks


def draw_recommendation(block: dict) -> None:
    st.code(block["text"], language=None)

    if not block["chain"]:
        return

    with st.expander(f"Causal chain behind recommendation {block['index']}"):
        st.markdown("**Path narrated above**")
        st.write(" -> ".join(block["path"]))
        st.caption(
            "Effects are composed multiplicatively in log space along this path, "
            "with each edge after the first attenuated by TRANSMISSION."
        )
        for step in block["chain"]:
            effect = step["effect"]
            st.markdown(f"**{step['source']} -> {step['target']}**  (`{step['sign']}`)")
            st.write(
                f"{step['metric']}, {effect.family} on "
                f"[{effect.ci_low:g}, {effect.ci_high:g}] at {effect.ci_level:.0%} CI, "
                f"lag {render.years(step['lag_years'])}, evidence {step['strength']}, "
                f"confidence {step['confidence']}"
            )
            st.write(step["mechanism"])
            if step["contested"]:
                st.warning(
                    f"Contested edge. {step['contested_note'] or ''} The disagreement is "
                    "carried as a wider distribution rather than reconciled to one value."
                )
            refs = ", ".join(
                f"{render.CITATIONS[sid]} [{role}]"
                for sid, role in step["refs"]
                if sid in render.CITATIONS
            )
            if refs:
                st.caption(f"Registered evidence: {refs}")

    with st.expander(f"Retrieved evidence for recommendation {block['index']}"):
        any_chunks = False
        for step in block["chain"]:
            chunks = step["chunks"]
            if not chunks:
                continue
            any_chunks = True
            st.markdown(f"**{step['source']} -> {step['target']}**")
            for chunk in chunks:
                score = (
                    f"rerank {chunk.rerank_score:.2f}"
                    if chunk.rerank_score is not None
                    else f"RRF {chunk.rrf_score:.4f}, no cross-encoder"
                )
                pages = ", ".join(str(p) for p in chunk.pages)
                flags = []
                if chunk.supporting_only:
                    flags.append("supporting only, not this edge's citation")
                if chunk.out_of_scope:
                    flags.append("out of scope")
                suffix = f" - {'; '.join(flags)}" if flags else ""
                st.caption(f"{chunk.citation} - pages {pages} - {score}{suffix}")
                st.text(chunk.body.strip()[:1200])
                if chunk.scope_note:
                    st.warning(chunk.scope_note)
        if not any_chunks:
            st.write(
                "No passage above the support floor was bound to these edges. The "
                "edges keep their registered citations; this corpus does not discuss "
                "them directly."
            )


# ================================= sidebar =================================


def draw_sidebar() -> str | None:
    """Presets and methodology. Returns a preset prompt when one was clicked."""
    clicked: str | None = None

    with st.sidebar:
        st.subheader("Demo sites")
        st.caption(
            "Each loads a different binding constraint, so the three get different "
            "diagnoses and different top recommendations."
        )
        for site_id, (label, prompt) in PRESET_PROMPTS.items():
            if st.button(label, key=f"preset_{site_id}", use_container_width=True):
                st.session_state["site_id"] = site_id
                st.session_state["thread_id"] = f"{site_id}-{uuid.uuid4().hex[:8]}"
                st.session_state["history"] = []
                st.session_state.pop("conversation", None)
                clicked = prompt

        if st.button("Reset conversation", use_container_width=True):
            st.session_state["thread_id"] = uuid.uuid4().hex[:12]
            st.session_state["history"] = []
            st.session_state.pop("conversation", None)
            st.rerun()

        st.divider()
        st.subheader("Methodology constants")
        st.caption(
            "Modelling assumptions, not empirical constants. No published study "
            "supports any of these values; they are choices about how evidence is "
            "combined, listed so the numbers can be argued with."
        )
        st.markdown(
            f"""
| constant | value |
| --- | --- |
| `TRANSMISSION` | {TRANSMISSION:g} |
| `DELTA_REF` | {DELTA_REF:g} |
| `PRIORITY_TOLERANCE` | {PRIORITY_TOLERANCE:.0%} |
| `RERANK_FLOOR` | {RERANK_FLOOR:g} |
| `TIER_1_MIN_EFFECT` | {TIER_1_MIN_EFFECT:.0%} |
"""
        )
        with st.expander("What each one does"):
            st.text("\n".join(render.methodology_lines()[1:]))

        st.divider()
        st.caption(f"Model: `{model_name()}`")
        if low_memory():
            st.caption(
                "Low memory mode: no cross-encoder. Retrieval returns the fused RRF "
                "order and support is judged on retriever agreement, which is a "
                "weaker test than a cross-encoder score."
            )

    return clicked


# ================================== main ===================================


def main() -> None:
    st.title("Darukaa.Earth")
    st.caption(
        "An AI environmental scientist. Describe your land and its problem; it "
        "diagnoses the binding constraint and recommends interventions, each with "
        "a quantified effect, a time horizon, a confidence level and a citation to "
        "a real study."
    )

    st.session_state.setdefault("thread_id", uuid.uuid4().hex[:12])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("site_id", "app_site")

    preset = draw_sidebar()

    ready, message = retrieval_status()
    if not ready:
        st.error(message)
        st.stop()

    for entry in st.session_state["history"]:
        with st.chat_message(entry["role"]):
            if entry.get("text"):
                st.write(entry["text"])
            for block in entry.get("blocks", []):
                draw_recommendation(block)
            if entry.get("grounding"):
                with st.expander("Grounding"):
                    st.text(entry["grounding"])

    typed = st.chat_input("Describe your site, or answer the question above")
    user_text = typed or preset
    if not user_text:
        return

    st.session_state["history"].append({"role": "user", "text": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = new_conversation()
    conversation: Conversation = st.session_state["conversation"]

    with st.chat_message("assistant"):
        with st.spinner("Diagnosing: propagating interventions and binding evidence..."):
            try:
                result = conversation.send(user_text)
            except Exception as error:  # noqa: BLE001 - surfaced to the user
                st.error(f"The agent could not complete this turn: {error}")
                st.session_state["history"].append(
                    {"role": "assistant", "text": f"Failed: {error}"}
                )
                return

        question = conversation.question(result)
        if question is not None:
            st.write(question)
            st.session_state["history"].append({"role": "assistant", "text": question})
            return

        state = conversation.state()
        draft = result.get("draft") or "No draft was produced for this turn."
        blocks = recommendation_blocks(state)

        grounding = None
        coverage = state.get("grounding_coverage")
        if coverage is not None:
            grounding = coverage_report(state.get("claims") or [], coverage)

        st.write(draft)
        for block in blocks:
            draw_recommendation(block)
        if grounding:
            with st.expander("Grounding"):
                st.text(grounding)

        st.session_state["history"].append(
            {
                "role": "assistant",
                "text": draft,
                "blocks": blocks,
                "grounding": grounding,
            }
        )


if __name__ == "__main__":
    main()
