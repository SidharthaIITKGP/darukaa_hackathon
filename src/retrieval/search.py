"""Hybrid search: dense plus BM25, fused by reciprocal rank, optionally
cross-encoder reranked, with evidence scope checked against the site.

Nothing here invents a citation. The citation string on every returned chunk
is looked up in sources.yaml by source_id, so a chunk whose source is not
registered cannot be returned with a citation at all.

Reranking is off by default. A cross-encoder is worth its latency only where
a passage is about to be attached to a claim: bind_evidence, the critic, and
the eval harness turn a specific passage into a specific citation, and there
precision is the whole point. A conversational turn or an intervention
ranking is only surfacing context, and RRF over dense plus BM25 already puts
the right document in the first few results. Paying a cross-encoder forward
pass per candidate on every turn buys ordering nobody reads and costs the
response budget.

Two rerankers, selected per call site rather than globally:

  RERANK_MODEL_FAST     bge-reranker-base, 278M parameters, English only.
  RERANK_MODEL_PRECISE  bge-reranker-v2-m3, 568M, multilingual.

Which one runs is the `precise` argument, so the citation paths can buy
accuracy with latency without imposing it on the conversation.
"""

from __future__ import annotations

import atexit
import pickle
import time
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.graph.schemas import SiteState
from src.retrieval.index import (
    BM25_PATH,
    BUILD_COMMAND,
    COLLECTION,
    QDRANT_PATH,
    embedding_model,
    tokenize,
)

_ROOT = Path(__file__).resolve().parents[2]
SOURCES_YAML_PATH = _ROOT / "sources.yaml"

RERANK_MODEL_FAST = "BAAI/bge-reranker-base"
RERANK_MODEL_PRECISE = "BAAI/bge-reranker-v2-m3"

# Candidates each retriever contributes before fusion.
CANDIDATE_DEPTH = 30
# How many fused candidates the cross-encoder scores. It is roughly two
# orders of magnitude slower per pair than a dot product, so the fused
# ranking is used to decide what is worth that cost. RRF has already done
# most of the filtering by this point: a chunk sitting 12th after fusion
# almost never wins the rerank, so scoring it is latency spent on an
# outcome that does not change.
RERANK_DEPTH = 8
# Reciprocal rank fusion constant. 60 is the value from Cormack et al. 2009
# and damps the influence of the very top rank enough that one retriever
# cannot unilaterally decide the fused order.
RRF_K = 60

# Cross-encoder score below which a passage is treated as no support at all.
# Chosen from the observed score distribution on this corpus, where a passage
# that genuinely answers the query scores around 0.99 and the best available
# match for a topic the corpus does not cover scores under 0.2. It is a
# threshold on an uncalibrated model output, not a probability, and it is a
# modelling assumption in exactly the sense the methodology section means.
RERANK_FLOOR = 0.30

# Query results cached per process. Conversational flows re-ask the same
# question as site state fills in, and an edge's bound evidence is requested
# again on every re-rank of the intervention list, so the same (query,
# filters) tuple recurs often. Cached entries are copied out, never handed to
# the caller, because flag_scope writes to the chunks it is given and a
# shared cached object would carry one site's scope note into the next.
QUERY_CACHE_SIZE = 256

# Latitude band treated as tropical when checking temperate-scoped evidence.
_TROPIC_LAT = 23.5
# Rough bounding box for Europe, used only to flag Europe-scoped sources.
_EUROPE_LAT = (35.0, 71.0)
_EUROPE_LON = (-25.0, 45.0)


class RetrievedChunk(BaseModel):
    """One passage returned by search, with the provenance of how it was found.

    supporting_only marks a chunk retrieved from outside the registered
    evidence refs of the edge it was retrieved for. Such a passage is
    contextual support: it may inform an answer, but the edge does not rest
    on it, and it must never be emitted as that edge's citation. Only
    bind_evidence sets it; a bare search leaves it False because there is no
    edge to be outside of.
    """

    chunk_id: str
    source_id: str
    citation: str  # resolved from sources.yaml, never constructed
    tier: str
    tags: list[str]
    pages: list[int]
    body: str
    dense_rank: int | None
    bm25_rank: int | None
    rrf_score: float
    rerank_score: float | None
    out_of_scope: bool
    scope_note: str | None
    # Set by bind_evidence when a passage was retrieved from outside the
    # edge's own evidence refs. Such a passage may inform, but it is not what
    # the edge cites, and the two must not be conflated in output.
    supporting_only: bool = False


@lru_cache(maxsize=1)
def _citations() -> dict[str, str]:
    data = yaml.safe_load(SOURCES_YAML_PATH.read_text())
    return {
        source_id: info["citation"]
        for source_id, info in data.get("sources", {}).items()
        if isinstance(info, dict) and "citation" in info
    }


@lru_cache(maxsize=1)
def _scopes() -> dict[str, str]:
    data = yaml.safe_load(SOURCES_YAML_PATH.read_text())
    return {
        source_id: (info.get("scope") or "")
        for source_id, info in data.get("sources", {}).items()
        if isinstance(info, dict)
    }


@lru_cache(maxsize=1)
def _qdrant():
    from qdrant_client import QdrantClient

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant index missing at {QDRANT_PATH}. Build it with: {BUILD_COMMAND}"
        )
    client = QdrantClient(path=str(QDRANT_PATH))
    # A local Qdrant holds a lock on its directory and closes itself from
    # __del__, which runs late enough during interpreter shutdown that the
    # import machinery is already gone and the traceback is printed to
    # stderr. Closing at exit instead keeps that noise out of demo output.
    atexit.register(client.close)
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION}' does not exist at {QDRANT_PATH}. "
            f"Build it with: {BUILD_COMMAND}"
        )
    return client


@lru_cache(maxsize=1)
def _bm25():
    """Return (BM25Okapi, chunk_ids, source_ids, tiers, tags)."""
    from rank_bm25 import BM25Okapi

    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {BM25_PATH}. Build it with: {BUILD_COMMAND}"
        )
    with BM25_PATH.open("rb") as f:
        payload = pickle.load(f)
    return (
        BM25Okapi(payload["tokens"]),
        payload["chunk_ids"],
        payload["source_ids"],
        payload["tiers"],
        payload["tags"],
    )


@lru_cache(maxsize=2)
def _reranker(precise: bool):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL_PRECISE if precise else RERANK_MODEL_FAST)


def _passes_filter(
    tier_value: str,
    tag_values: list[str],
    source_value: str,
    tier: str | None,
    tags: list[str] | None,
    source_ids: list[str] | None,
) -> bool:
    if tier is not None and tier_value != tier:
        return False
    if source_ids is not None and source_value not in source_ids:
        return False
    if tags:
        # Any-of. Chunk tags are descriptive rather than a controlled
        # taxonomy, so requiring all of them would usually return nothing.
        if not set(tags) & set(tag_values):
            return False
    return True


def _dense_candidates(
    query: str, tier: str | None, tags: list[str] | None, source_ids: list[str] | None
) -> list[dict]:
    from qdrant_client import models

    conditions = []
    if tier is not None:
        conditions.append(models.FieldCondition(key="tier", match=models.MatchValue(value=tier)))
    if source_ids is not None:
        conditions.append(
            models.FieldCondition(key="source_id", match=models.MatchAny(any=list(source_ids)))
        )
    if tags:
        conditions.append(models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags))))
    query_filter = models.Filter(must=conditions) if conditions else None

    vector = embedding_model().encode(query, normalize_embeddings=True).tolist()
    response = _qdrant().query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=CANDIDATE_DEPTH,
        query_filter=query_filter,
        with_payload=True,
    )
    return [point.payload for point in response.points]


def _bm25_candidates(
    query: str, tier: str | None, tags: list[str] | None, source_ids: list[str] | None
) -> list[str]:
    bm25, chunk_ids, all_source_ids, tiers, all_tags = _bm25()
    scores = bm25.get_scores(tokenize(query))
    allowed = [
        i
        for i in range(len(chunk_ids))
        if _passes_filter(tiers[i], all_tags[i], all_source_ids[i], tier, tags, source_ids)
    ]
    allowed.sort(key=lambda i: scores[i], reverse=True)
    return [chunk_ids[i] for i in allowed[:CANDIDATE_DEPTH] if scores[i] > 0]


class SearchTiming(BaseModel):
    """Wall-clock milliseconds per stage of one uncached search."""

    dense_ms: float
    bm25_ms: float
    rrf_ms: float
    rerank_ms: float
    total_ms: float


def search(
    query: str,
    k: int = 8,
    tier: str | None = None,
    tags: list[str] | None = None,
    rerank: bool = False,
    precise: bool = False,
    source_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Hybrid retrieval over the ingested corpus.

    tier, tags and source_ids, when given, are hard filters applied
    identically to both retrievers, so fusion always compares two rankings
    over the same candidate pool. source_ids exists for bind_evidence, which
    has to restrict retrieval to the sources an edge actually cites.

    rerank defaults to False. Turn it on where a passage is being attached to
    a claim, and set precise=True there as well if the citation has to be
    right rather than merely plausible.
    """
    cached = _search_cached(
        query,
        k,
        tier,
        tuple(tags) if tags is not None else None,
        tuple(source_ids) if source_ids is not None else None,
        rerank,
        precise,
    )
    # Deep copies, so a caller's flag_scope call cannot write scope notes
    # into the cache and leak them to the next site asking the same question.
    return [chunk.model_copy(deep=True) for chunk in cached]


def search_with_timing(
    query: str,
    k: int = 8,
    tier: str | None = None,
    tags: list[str] | None = None,
    rerank: bool = False,
    precise: bool = False,
    source_ids: list[str] | None = None,
) -> tuple[list[RetrievedChunk], SearchTiming]:
    """Run a search that bypasses the query cache and report per-stage timing.

    Deliberately uncached: a measurement served from an lru_cache measures
    the lru_cache.
    """
    chunks, timing = _search_uncached(query, k, tier, tags, source_ids, rerank, precise)
    return [chunk.model_copy(deep=True) for chunk in chunks], timing


@lru_cache(maxsize=QUERY_CACHE_SIZE)
def _search_cached(
    query: str,
    k: int,
    tier: str | None,
    tags: tuple[str, ...] | None,
    source_ids: tuple[str, ...] | None,
    rerank: bool,
    precise: bool,
) -> tuple[RetrievedChunk, ...]:
    chunks, _ = _search_uncached(
        query,
        k,
        tier,
        list(tags) if tags is not None else None,
        list(source_ids) if source_ids is not None else None,
        rerank,
        precise,
    )
    return tuple(chunks)


def _search_uncached(
    query: str,
    k: int,
    tier: str | None,
    tags: list[str] | None,
    source_ids: list[str] | None,
    rerank: bool,
    precise: bool,
) -> tuple[list[RetrievedChunk], SearchTiming]:
    started = time.perf_counter()

    dense_start = time.perf_counter()
    dense_payloads = _dense_candidates(query, tier, tags, source_ids)
    dense_ms = (time.perf_counter() - dense_start) * 1000

    bm25_start = time.perf_counter()
    bm25_ids = _bm25_candidates(query, tier, tags, source_ids)
    bm25_ms = (time.perf_counter() - bm25_start) * 1000

    rrf_start = time.perf_counter()

    payload_by_id = {p["chunk_id"]: p for p in dense_payloads}
    dense_rank = {p["chunk_id"]: i + 1 for i, p in enumerate(dense_payloads)}
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ids)}

    # Reciprocal rank fusion rather than normalising and adding the two
    # scores. Dense cosine similarity lives in a narrow band near 1 while
    # BM25 is an unbounded sum of term weights, so their scales are not
    # comparable and any normalisation would need per-query calibration that
    # a rank-based fusion simply does not require. RRF only asks which
    # retriever put a document higher.
    fused: dict[str, float] = {}
    for cid in set(dense_rank) | set(bm25_rank):
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (RRF_K + dense_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[cid])
        fused[cid] = score

    ordered = sorted(fused, key=lambda cid: (-fused[cid], cid))

    # BM25 can surface a chunk the dense retriever did not return, so its
    # payload has to be fetched before it can be turned into a result.
    missing = [cid for cid in ordered if cid not in payload_by_id]
    if missing:
        payload_by_id.update(_payloads_by_chunk_id(missing))

    results = [
        RetrievedChunk(
            chunk_id=cid,
            source_id=payload_by_id[cid]["source_id"],
            citation=_citation_for(payload_by_id[cid]["source_id"]),
            tier=payload_by_id[cid]["tier"],
            tags=payload_by_id[cid]["tags"],
            pages=payload_by_id[cid]["pages"],
            body=payload_by_id[cid]["body"],
            dense_rank=dense_rank.get(cid),
            bm25_rank=bm25_rank.get(cid),
            rrf_score=fused[cid],
            rerank_score=None,
            out_of_scope=False,
            scope_note=None,
        )
        for cid in ordered
        if cid in payload_by_id
    ]
    rrf_ms = (time.perf_counter() - rrf_start) * 1000

    rerank_start = time.perf_counter()
    if rerank and results:
        head = results[:RERANK_DEPTH]
        tail = results[RERANK_DEPTH:]
        # A cross-encoder feeds query and document through one transformer,
        # so every query token can attend to every document token. Unlike a
        # bi-encoder, which embeds each side independently and can only
        # compare the two summaries afterwards, it makes no independence
        # assumption between them. The cost is that it cannot be precomputed:
        # every pair is a forward pass at query time, which is why only the
        # top RERANK_DEPTH fused candidates get scored.
        #
        # One predict() call per query, and deliberately not one call across
        # many queries. Batching every claim's candidates into a single call
        # was implemented and measured on this corpus: 14 claim queries took
        # 21.8s sequentially and 23.5s batched, i.e. 0.93x, with identical
        # scores. On CPU this model is compute-bound rather than
        # call-overhead-bound, so there is no fixed cost to amortise, and a
        # wide batch pads every sequence to the longest one in it and spends
        # the saving on padding. The batched version was reverted.
        scores = _reranker(precise).predict([(query, c.body) for c in head])
        for chunk, score in zip(head, scores):
            chunk.rerank_score = float(score)
        head.sort(key=lambda c: c.rerank_score, reverse=True)
        results = head + tail
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    timing = SearchTiming(
        dense_ms=dense_ms,
        bm25_ms=bm25_ms,
        rrf_ms=rrf_ms,
        rerank_ms=rerank_ms,
        total_ms=(time.perf_counter() - started) * 1000,
    )
    return results[:k], timing


def has_support(chunks: list[RetrievedChunk]) -> bool:
    """True when at least one chunk clears RERANK_FLOOR.

    Retrieval always returns k results. That is a property of retrieval, not
    evidence that the corpus says anything about the query: asked about
    contour bunding, a corpus containing no bunding meta-analysis still
    returns its three least-bad matches. This separates evidence found from
    least-bad match returned, so the critic can decline to make a claim
    rather than cite a passage that does not support it.

    Only meaningful on reranked results. Chunks with no rerank_score have not
    been scored against the query by anything that could answer this, so they
    count as no support.
    """
    return any(c.rerank_score is not None and c.rerank_score >= RERANK_FLOOR for c in chunks)


def warmup(precise: bool = False) -> None:
    """Load the models and run one throwaway query.

    Without this the first real query pays several seconds of model load and
    lazy index open, which in a conversation reads as a hang rather than as
    work.
    """
    _qdrant()
    _bm25()
    embedding_model()
    _reranker(precise)
    _search_uncached("soil organic carbon", 3, None, None, None, True, precise)


def _payloads_by_chunk_id(chunk_ids: list[str]) -> dict[str, dict]:
    from qdrant_client import models

    response = _qdrant().scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids))]
        ),
        limit=len(chunk_ids),
        with_payload=True,
    )
    return {point.payload["chunk_id"]: point.payload for point in response[0]}


def _citation_for(source_id: str) -> str:
    citations = _citations()
    if source_id not in citations:
        raise ValueError(
            f"source_id '{source_id}' is not registered in {SOURCES_YAML_PATH}; "
            "it must not appear in output"
        )
    return citations[source_id]


def _scope_violation(scope: str, site: SiteState) -> str | None:
    """Return a note when the site falls outside a source's stated scope.

    Conservative by design: it reports only violations it can actually
    establish from the site record, and says nothing when the site lacks the
    field the scope talks about. A missing note means unchecked, not cleared.
    """
    lowered = scope.lower()
    if not lowered:
        return None

    notes: list[str] = []
    zone = site.climate_zone()

    # "global" does not clear a scope on its own. Joshi 2023 is registered as
    # "corn rotations, global, mostly temperate": worldwide in principle, but
    # its evidence base is neither semi-arid nor non-corn, and a site that is
    # both sits outside what the pooled estimate actually measured.
    if "temperate" in lowered:
        outside = None
        if zone in ("arid", "semi_arid"):
            outside = f"this site is {zone}"
        elif site.lat is not None and abs(site.lat) < _TROPIC_LAT:
            outside = f"this site lies at {site.lat:.2f} degrees latitude, inside the tropics"
        if outside is not None:
            notes.append(
                f"Evidence scope is '{scope}' and {outside}. Applying it here is an "
                "extrapolation beyond the climates the underlying studies sampled."
            )

    if "europe" in lowered and site.lat is not None and site.lon is not None:
        in_europe = (
            _EUROPE_LAT[0] <= site.lat <= _EUROPE_LAT[1]
            and _EUROPE_LON[0] <= site.lon <= _EUROPE_LON[1]
        )
        if not in_europe:
            notes.append(
                f"Evidence scope is '{scope}' and this site is outside Europe. "
                "Applying it here is an extrapolation."
            )

    if "corn" in lowered and site.crop is not None and site.crop.lower() not in ("corn", "maize"):
        notes.append(
            f"Evidence scope is '{scope}' and the cropping system here is {site.crop}."
        )

    return " ".join(notes) if notes else None


def flag_scope(chunks: list[RetrievedChunk], site: SiteState) -> None:
    """Mark, in place, every chunk whose source was not validated for this site.

    Out-of-scope chunks are flagged, never dropped. A scientist asked about
    cover crops on a semi-arid Indian site still cites the temperate
    meta-analysis, because it is the best evidence that exists, and states
    plainly that applying it there is an extrapolation. Silently withholding
    it would leave the recommendation looking better supported than it is.
    """
    scopes = _scopes()
    for chunk in chunks:
        note = _scope_violation(scopes.get(chunk.source_id, ""), site)
        chunk.out_of_scope = note is not None
        chunk.scope_note = note
