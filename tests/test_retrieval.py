"""Tests for hybrid retrieval: indexes, filters, fusion, reranking, scope."""

from __future__ import annotations

import pytest

from src.graph.edges import EDGES
from src.graph.schemas import CausalEdge, Confidence, Measurement, Provenance, SiteState
from src.retrieval import search as search_module
from src.retrieval.bind import bind_evidence
from src.retrieval.index import BM25_PATH, COLLECTION, build_indexes, load_chunks
from src.retrieval.search import (
    RERANK_FLOOR,
    RRF_K,
    flag_scope,
    has_support,
    search,
    search_with_timing,
    warmup,
)

RERANK_QUERIES = [
    "cover crops soil organic carbon percentage increase",
    "agroforestry biodiversity species richness",
    "contour bunding erosion control semi-arid",
]


@pytest.fixture(scope="session", autouse=True)
def indexes() -> None:
    build_indexes()


def _deccan_site() -> SiteState:
    """Semi-arid Deccan plateau site. Outside every temperate-scoped source."""
    return SiteState(
        site_id="deccan_semi_arid",
        lat=17.38,
        lon=78.49,
        annual_rainfall_mm=Measurement(
            value=620.0,
            unit="mm",
            provenance=Provenance.API_NASA_POWER,
            confidence=Confidence.HIGH,
        ),
        soil_organic_carbon_pct=Measurement(
            value=0.42,
            unit="%",
            provenance=Provenance.API_SOILGRIDS,
            confidence=Confidence.MODERATE,
        ),
    )


def _legume_soc_edge() -> CausalEdge:
    for edge in EDGES:
        if edge.source == "legume_cover_crop" and edge.target == "soil_organic_carbon":
            return edge
    raise AssertionError("legume_cover_crop -> soil_organic_carbon edge is missing")


def test_indexes_built_over_every_chunk() -> None:
    n_chunks = len(load_chunks())
    assert n_chunks == 542

    client = search_module._qdrant()
    assert client.collection_exists(COLLECTION)
    assert client.count(COLLECTION).count == n_chunks
    assert BM25_PATH.exists()


def test_tier_filter_returns_only_meta_analysis() -> None:
    results = search("cover crops soil organic carbon effect size", tier="meta_analysis")
    assert results
    assert all(c.tier == "meta_analysis" for c in results)

    top_sources = {c.source_id for c in results[:3]}
    assert top_sources & {"Joshi_2023_covercrops_SOC", "McClelland_2020_covercrops_SOC"}


def test_tier_filter_never_leaks_assessment_chunks() -> None:
    """The filter has to hold for queries whose best matches are assessments."""
    for query in [
        "response options tradeoffs feasibility land degradation",
        "soil biodiversity mechanism microbial",
        "sustainable soil management guidelines",
    ]:
        results = search(query, k=10, tier="meta_analysis", rerank=False)
        assert results, query
        assert all(c.tier == "meta_analysis" for c in results), query


def test_rrf_rewards_agreement_between_retrievers() -> None:
    """A chunk both retrievers rank must outscore one only a single retriever
    ranks no lower. That is the whole point of rank fusion: agreement between
    two independent rankings is evidence, a single high rank is not.
    """
    results = search("cover crops soil organic carbon percentage increase", k=40, rerank=False)

    both = [c for c in results if c.dense_rank is not None and c.bm25_rank is not None]
    single = [c for c in results if (c.dense_rank is None) != (c.bm25_rank is None)]
    assert both, "no chunk was retrieved by both retrievers, so fusion is untested"
    assert single, "no chunk was retrieved by exactly one retriever, so fusion is untested"

    compared = 0
    for agreed in both:
        best_agreed_rank = min(agreed.dense_rank, agreed.bm25_rank)
        for lone in single:
            lone_rank = lone.dense_rank if lone.dense_rank is not None else lone.bm25_rank
            if lone_rank < best_agreed_rank:
                continue
            compared += 1
            assert agreed.rrf_score > lone.rrf_score
            assert agreed.rrf_score == pytest.approx(
                1.0 / (RRF_K + agreed.dense_rank) + 1.0 / (RRF_K + agreed.bm25_rank)
            )
    assert compared > 0, "no comparable pair found"


@pytest.mark.slow
def test_reranking_changes_the_ordering() -> None:
    changed = False
    for query in RERANK_QUERIES:
        fused = search(query, k=5, rerank=False)
        reranked = search(query, k=5, rerank=True)
        assert reranked
        assert all(c.rerank_score is not None for c in reranked)
        if fused[0].chunk_id != reranked[0].chunk_id:
            changed = True
        else:
            spread = max(c.rerank_score for c in reranked) - min(c.rerank_score for c in reranked)
            if spread > 0.1:
                changed = True
    assert changed, "reranking altered neither the top result nor the score spread on any query"


def test_flag_scope_marks_temperate_evidence_on_a_semi_arid_site() -> None:
    site = _deccan_site()
    results = search("cover crops temperate soil organic carbon stocks", k=10, tier="meta_analysis")
    flag_scope(results, site)

    mcclelland = [c for c in results if c.source_id == "McClelland_2020_covercrops_SOC"]
    assert mcclelland, "expected at least one McClelland chunk for this query"
    for chunk in mcclelland:
        assert chunk.out_of_scope
        assert chunk.scope_note
        assert "temperate" in chunk.scope_note.lower()


def test_flag_scope_leaves_global_evidence_in_scope() -> None:
    site = _deccan_site()
    results = search("land degradation response options", k=10, tier="assessment")
    flag_scope(results, site)
    globals_ = [c for c in results if c.source_id.startswith(("IPCC", "IPBES", "FAO"))]
    assert globals_
    assert all(not c.out_of_scope and c.scope_note is None for c in globals_)


@pytest.mark.slow
def test_bind_evidence_uses_the_edges_own_sources() -> None:
    edge = _legume_soc_edge()
    cited = {ref.source_id for ref in edge.evidence}

    chunks = bind_evidence(edge, _deccan_site(), k=3)
    assert len(chunks) >= 2
    assert all(c.source_id in cited for c in chunks)
    assert all(not c.supporting_only for c in chunks)

    # Both of this edge's sources are cover-crop meta-analyses, so every
    # bound passage should be meta-analysis tier rather than assessment prose.
    assert all(c.tier == "meta_analysis" for c in chunks)


@pytest.mark.slow
def test_bind_evidence_flags_extrapolation_on_a_semi_arid_site() -> None:
    edge = _legume_soc_edge()
    chunks = bind_evidence(edge, _deccan_site(), k=3)
    mcclelland = [c for c in chunks if c.source_id == "McClelland_2020_covercrops_SOC"]
    if mcclelland:
        assert all(c.out_of_scope and c.scope_note for c in mcclelland)


def test_rerank_is_off_by_default() -> None:
    """The conversational path must not pay the cross-encoder unasked."""
    results = search("cover crops soil organic carbon percentage increase", k=3)
    assert results
    assert all(c.rerank_score is None for c in results)


def test_rrf_only_search_is_well_under_a_second() -> None:
    """The whole point of the default. A conversational turn issues several
    of these, so the retrieval share of the response budget has to be small.
    """
    warmup(precise=False)
    _, timing = search_with_timing("cover crops soil organic carbon percentage increase", k=3)
    assert timing.rerank_ms < 1.0  # the skipped branch, not a forward pass
    assert timing.total_ms < 1000.0


@pytest.mark.slow
def test_has_support_separates_covered_from_uncovered_topics() -> None:
    """Both rerankers have to agree on this, because it is a claim about the
    corpus rather than about a model: the corpus has cover-crop
    meta-analyses and has nothing on contour bunding.
    """
    for precise in (False, True):
        covered = search(
            "cover crops soil organic carbon percentage increase",
            k=3,
            rerank=True,
            precise=precise,
        )
        assert has_support(covered), precise
        assert max(c.rerank_score for c in covered) > RERANK_FLOOR

        uncovered = search(
            "contour bunding erosion control semi-arid", k=3, rerank=True, precise=precise
        )
        assert uncovered, "retrieval still returns its least-bad matches"
        assert not has_support(uncovered), precise


def test_has_support_is_false_without_reranking() -> None:
    """An unreranked result has not been scored by anything that could
    answer the question, so it must not read as support.
    """
    assert not has_support(search("cover crops soil organic carbon", k=3))


def test_cache_hands_out_independent_objects() -> None:
    """flag_scope writes to the chunks it is given. If the cache handed back
    its own objects, one site's scope note would follow the next site's
    identical query.
    """
    query = "cover crops temperate soil organic carbon stocks"
    first = search(query, k=5, tier="meta_analysis")
    flag_scope(first, _deccan_site())
    assert any(c.out_of_scope for c in first)

    second = search(query, k=5, tier="meta_analysis")
    assert [c.chunk_id for c in second] == [c.chunk_id for c in first]
    assert all(not c.out_of_scope and c.scope_note is None for c in second)


def test_every_returned_citation_is_registered() -> None:
    registered = search_module._citations()
    seen = 0
    for query in RERANK_QUERIES:
        for chunk in search(query, k=8):
            assert chunk.citation
            assert chunk.citation == registered[chunk.source_id]
            seen += 1
    assert seen > 0
