"""Chunking for the ingestion pipeline.

Splits parsed page text into ~1000-token chunks (1 token ~= 0.75 words, a
plain word-count approximation, no tokenizer dependency), respecting
paragraph boundaries where possible, and prepends a contextual header to
each chunk's embedding text.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.ingest.parse import ParsedSource

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = REPO_ROOT / "sources.yaml"
MANIFEST_PATH = REPO_ROOT / "ingest_manifest.yaml"

WORDS_PER_TOKEN = 0.75

_HEADING_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)*\.?\s+\S.{0,80}|[A-Z][A-Za-z0-9 ,'\-/]{2,80})$"
)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    pages: list[int]
    tier: str
    tags: list[str]
    header: str
    body: str
    embed_text: str
    char_count: int


class _Paragraph(BaseModel):
    page: int
    heading: str
    text: str


_sources_cache: dict | None = None
_manifest_cache: dict | None = None


def _load_sources() -> dict:
    global _sources_cache
    if _sources_cache is None:
        with open(SOURCES_PATH) as f:
            _sources_cache = yaml.safe_load(f)["sources"]
    return _sources_cache


def _manifest_file_for(source_id: str) -> str | None:
    global _manifest_cache
    if _manifest_cache is None:
        with open(MANIFEST_PATH) as f:
            _manifest_cache = yaml.safe_load(f)["sources"]
    entry = _manifest_cache.get(source_id)
    return entry.get("file") if entry else None


def _citation_for(source_id: str, source_file: str | None) -> str:
    """Look up the citation string for a source_id in sources.yaml.

    ingest_manifest.yaml and sources.yaml disagree on the exact key for two
    entries (GCB_2025_agroforestry vs GCB_2025_agroforestry_multifunctionality,
    Mupepele_2021_agroforestry vs Mupepele_2021_agroforestry_biodiv). Fall
    back to matching on the `file` path when the direct key lookup misses.
    """
    sources = _load_sources()
    entry = sources.get(source_id)
    if entry is None and source_file is not None:
        for candidate in sources.values():
            if isinstance(candidate, dict) and candidate.get("file") == source_file:
                entry = candidate
                break
    if entry is None:
        return source_id
    return entry.get("citation", source_id)


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    word_count = len(stripped.split())
    if word_count > 12:
        return False
    return bool(_HEADING_RE.match(stripped))


def _paragraphs_from_pages(parsed: ParsedSource) -> list[_Paragraph]:
    """Split each page's text into paragraphs, tracking the nearest heading
    line seen so far (heading state carries across pages)."""
    paragraphs: list[_Paragraph] = []
    current_heading = "unknown"

    for page_text in parsed.pages:
        lines = page_text.text.split("\n")
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            joined = "\n".join(buffer).strip()
            if joined:
                paragraphs.append(
                    _Paragraph(page=page_text.page, heading=current_heading, text=joined)
                )
            buffer = []

        for line in lines:
            if not line.strip():
                flush()
                continue
            if _is_heading(line) and not buffer:
                current_heading = line.strip()
                continue
            buffer.append(line)
        flush()

    return paragraphs


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_source(parsed: ParsedSource, defaults: dict) -> list[Chunk]:
    chunk_tokens = defaults.get("chunk_tokens", 1000)
    chunk_overlap = defaults.get("chunk_overlap", 150)
    min_chunk_chars = defaults.get("min_chunk_chars", 250)

    target_words = int(chunk_tokens * WORDS_PER_TOKEN)
    overlap_words = int(chunk_overlap * WORDS_PER_TOKEN)

    paragraphs = _paragraphs_from_pages(parsed)
    source_file = _manifest_file_for(parsed.source_id)
    citation = _citation_for(parsed.source_id, source_file)

    chunks: list[Chunk] = []
    chunk_index = 0

    current_paragraphs: list[_Paragraph] = []
    current_word_count = 0

    def emit(paragraphs_for_chunk: list[_Paragraph]) -> None:
        nonlocal chunk_index
        if not paragraphs_for_chunk:
            return
        body = "\n\n".join(p.text for p in paragraphs_for_chunk)
        if len(body) < min_chunk_chars:
            return
        pages = sorted({p.page for p in paragraphs_for_chunk})
        heading = next((p.heading for p in paragraphs_for_chunk if p.heading != "unknown"), "unknown")
        header = f"[Source: {citation} | Section: {heading} | Tier: {parsed.tier}]"
        embed_text = header + "\n" + body
        chunk_id = f"{parsed.source_id}:p{pages[0]}:c{chunk_index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_id=parsed.source_id,
                pages=pages,
                tier=parsed.tier,
                tags=list(parsed.tags),
                header=header,
                body=body,
                embed_text=embed_text,
                char_count=len(body),
            )
        )
        chunk_index += 1

    def _split_long_paragraph(paragraph: _Paragraph) -> list[_Paragraph]:
        words = paragraph.text.split()
        pieces: list[_Paragraph] = []
        start = 0
        while start < len(words):
            end = min(start + target_words, len(words))
            piece_text = " ".join(words[start:end])
            pieces.append(_Paragraph(page=paragraph.page, heading=paragraph.heading, text=piece_text))
            if end == len(words):
                break
            start = end - overlap_words
        return pieces

    expanded_paragraphs: list[_Paragraph] = []
    for para in paragraphs:
        if _word_count(para.text) > target_words:
            expanded_paragraphs.extend(_split_long_paragraph(para))
        else:
            expanded_paragraphs.append(para)

    for para in expanded_paragraphs:
        para_words = _word_count(para.text)

        if current_word_count + para_words > target_words and current_paragraphs:
            emit(current_paragraphs)

            overlap_paragraphs: list[_Paragraph] = []
            overlap_count = 0
            for p in reversed(current_paragraphs):
                p_words = _word_count(p.text)
                if overlap_count + p_words > overlap_words:
                    break
                overlap_paragraphs.insert(0, p)
                overlap_count += p_words

            current_paragraphs = overlap_paragraphs
            current_word_count = overlap_count

        current_paragraphs.append(para)
        current_word_count += para_words

    emit(current_paragraphs)

    return chunks
