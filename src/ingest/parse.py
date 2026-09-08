"""PDF parsing for the ingestion pipeline.

Reads only the page ranges listed in ingest_manifest.yaml for each source,
extracting text with PyMuPDF and, where flagged, tables with pdfplumber.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "ingest_manifest.yaml"
SOURCES_PATH = REPO_ROOT / "sources.yaml"
CHUNKS_PATH = REPO_ROOT / "data/derived/chunks.jsonl"
TABLES_PATH = REPO_ROOT / "data/derived/tables.jsonl"


class PageText(BaseModel):
    source_id: str
    page: int
    text: str


class ExtractedTable(BaseModel):
    source_id: str
    page: int
    table_index: int
    rows: list[list[str | None]]
    caption: str | None


class ParsedSource(BaseModel):
    source_id: str
    tier: str
    tags: list[str]
    pages: list[PageText]
    tables: list[ExtractedTable]


def _pages_to_read(pages_spec: list[list[int]]) -> list[int]:
    """Expand manifest [start, end] ranges (1-indexed, inclusive) to a sorted
    list of unique page numbers."""
    result: set[int] = set()
    for start, end in pages_spec:
        result.update(range(start, end + 1))
    return sorted(result)


def _nearest_caption(page_text: str, table_bbox_top_lines: list[str]) -> str | None:
    """Best-effort caption: nearest non-empty line above the table on the page."""
    for line in reversed(table_bbox_top_lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _extract_tables_for_page(pdf_path: Path, page_number_1idx: int, source_id: str) -> list[ExtractedTable]:
    """Run pdfplumber's table extraction on a single page."""
    tables: list[ExtractedTable] = []
    with pdfplumber.open(pdf_path, pages=[page_number_1idx]) as pdf:
        page = pdf.pages[0]
        page_text = page.extract_text() or ""
        lines_before_table_cache = page_text.split("\n")
        found_tables = page.find_tables()
        for idx, table in enumerate(found_tables):
            rows = table.extract()
            if not rows:
                continue
            top_y = table.bbox[1]
            words_above = [w for w in page.extract_words() if w["bottom"] <= top_y]
            words_above.sort(key=lambda w: (w["top"], w["x0"]))
            caption_lines: list[str] = []
            if words_above:
                current_line_top = None
                current_line_words: list[str] = []
                for w in words_above:
                    if current_line_top is None or abs(w["top"] - current_line_top) > 2:
                        if current_line_words:
                            caption_lines.append(" ".join(current_line_words))
                        current_line_words = [w["text"]]
                        current_line_top = w["top"]
                    else:
                        current_line_words.append(w["text"])
                if current_line_words:
                    caption_lines.append(" ".join(current_line_words))
            caption = _nearest_caption(page_text, caption_lines) if caption_lines else None
            tables.append(
                ExtractedTable(
                    source_id=source_id,
                    page=page_number_1idx,
                    table_index=idx,
                    rows=[[cell for cell in row] for row in rows],
                    caption=caption,
                )
            )
    return tables


def parse_source(source_id: str, manifest: dict) -> ParsedSource:
    """Parse one source per its manifest entry, reading only listed pages."""
    entry = manifest["sources"][source_id]
    pdf_path = REPO_ROOT / entry["file"]
    page_numbers = _pages_to_read(entry["pages"])
    extract_tables_flag = entry.get("extract_tables", False)

    pages: list[PageText] = []
    tables: list[ExtractedTable] = []

    doc = fitz.open(pdf_path)
    try:
        for page_num in page_numbers:
            fitz_page = doc.load_page(page_num - 1)
            text = fitz_page.get_text()
            pages.append(PageText(source_id=source_id, page=page_num, text=text))
    finally:
        doc.close()

    if extract_tables_flag:
        for page_num in page_numbers:
            tables.extend(_extract_tables_for_page(pdf_path, page_num, source_id))

    return ParsedSource(
        source_id=source_id,
        tier=entry["tier"],
        tags=entry.get("tags", []),
        pages=pages,
        tables=tables,
    )


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_sources() -> dict:
    with open(SOURCES_PATH) as f:
        return yaml.safe_load(f)


def main() -> None:
    from src.ingest.chunk import chunk_source

    manifest = load_manifest()
    defaults = manifest.get("defaults", {})
    source_ids = list(manifest["sources"].keys())

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    with open(CHUNKS_PATH, "w") as chunks_f, open(TABLES_PATH, "w") as tables_f:
        for source_id in source_ids:
            parsed = parse_source(source_id, manifest)
            chunks = chunk_source(parsed, defaults)

            for chunk in chunks:
                chunks_f.write(chunk.model_dump_json() + "\n")
            for table in parsed.tables:
                tables_f.write(table.model_dump_json() + "\n")

            summary_rows.append(
                {
                    "source_id": source_id,
                    "pages_read": len(parsed.pages),
                    "chunks": len(chunks),
                    "tables": len(parsed.tables),
                }
            )

    header = f"{'source_id':<32} {'pages':>6} {'chunks':>7} {'tables':>7}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['source_id']:<32} {row['pages_read']:>6} {row['chunks']:>7} {row['tables']:>7}")


if __name__ == "__main__":
    main()
