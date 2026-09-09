"""Bind a causal edge to the passages that back it.

This is where the two halves of the system meet. The graph asserts that
legume cover crops raise soil organic carbon by some distribution; this
module produces the paragraphs from Joshi 2023 and McClelland 2021 that the
assertion rests on. Retrieval serves the graph here, not the other way
round: the query is built from an edge that already exists, and the edge's
own evidence refs decide where to look first.
"""

from __future__ import annotations

from src.graph.schemas import CausalEdge, SiteState
from src.retrieval.search import RetrievedChunk, flag_scope, search

# Cap on how much mechanism prose enters the query. The whole mechanism text
# runs to several hundred words on some edges, most of it conditions and
# caveats that pull retrieval away from the relationship itself.
_MECHANISM_QUERY_CHARS = 400


def _query_for(edge: CausalEdge) -> str:
    source = edge.source.replace("_", " ")
    target = edge.target.replace("_", " ")
    mechanism = edge.mechanism.strip()[:_MECHANISM_QUERY_CHARS]
    return f"{source} effect on {target}. {mechanism}"


def bind_evidence(
    edge: CausalEdge, site: SiteState, k: int = 3, precise: bool = False
) -> list[RetrievedChunk]:
    """Retrieve up to k passages supporting one causal edge.

    Reranking is on here, unlike a conversational search, because this is
    where a passage becomes a citation and which passage is returned is the
    answer rather than context around it.

    precise selects the heavier cross-encoder and defaults to off. Measured
    on the three demo queries, bge-reranker-base and bge-reranker-v2-m3
    return different chunk orderings but the same sources, and agree on
    which queries the corpus supports at all: about 0.99 for cover crops and
    agroforestry, under 0.2 for contour bunding under both models. base is
    roughly twelve times faster. There is no observed accuracy to buy at
    that price, so the default does not buy it. The argument stays exposed
    because three queries is a thin basis for the claim, and a caller that
    finds a case where it matters can escalate without touching this
    module.

    The edge's own evidence source_ids are searched first. Those passages are
    what the edge cites. Only if that yields fewer than k does the search
    widen to the whole corpus, and anything it picks up there is marked
    supporting_only: it may be relevant context, but the edge does not rest
    on it and it must not be presented as the edge's citation.

    Every returned chunk is scope-checked against the site, so a temperate
    meta-analysis bound to an edge applied on a semi-arid site comes back
    carrying its extrapolation note.
    """
    query = _query_for(edge)
    cited_ids = [ref.source_id for ref in edge.evidence]

    results = search(query, k=k, source_ids=cited_ids, rerank=True, precise=precise)

    if len(results) < k:
        seen = {chunk.chunk_id for chunk in results}
        # Ask for extra candidates because the cited chunks already returned
        # will appear again in the unrestricted search and be dropped.
        widened = search(query, k=k + len(results), rerank=True, precise=precise)
        for chunk in widened:
            if len(results) >= k:
                break
            if chunk.chunk_id in seen:
                continue
            if chunk.source_id in cited_ids:
                # Same sources, so still cited rather than supporting only.
                results.append(chunk)
            else:
                chunk.supporting_only = True
                results.append(chunk)
            seen.add(chunk.chunk_id)

    flag_scope(results, site)
    return results
