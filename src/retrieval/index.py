"""Index construction for hybrid retrieval.

Two indexes over the same 542 chunks in data/derived/chunks.jsonl:

  dense   bge-m3 embeddings of each chunk's embed_text, in a local Qdrant
          collection. Catches paraphrase, so "carbon storage" finds a
          passage that only ever says "SOC stock".
  sparse  BM25 over the same text. Catches exact tokens a dense model
          blurs away: source names, units, numbers like "7.3%".

Neither alone is enough for this corpus, which mixes prose mechanism with
numeric effect sizes. Fusion happens in search.py, not here.

Model choice is a deployment tradeoff, not a fixed property of the design.
DENSE_MODEL here and RERANK_MODEL in search.py are the only two places a
different embedder or reranker has to be named. On a GPU deployment the
higher-capacity pair is BAAI/bge-m3 for the dense side (set EMBED_DIM to
1024 with it) and BAAI/bge-reranker-v2-m3 for the reranker.

Run as a module to build both:  python -m src.retrieval.index
"""

from __future__ import annotations

import pickle
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = _ROOT / "data" / "derived" / "chunks.jsonl"
QDRANT_PATH = _ROOT / "data" / "derived" / "qdrant"
BM25_PATH = _ROOT / "data" / "derived" / "bm25.pkl"

COLLECTION = "darukaa"

# bge-small-en-v1.5: 33M parameters, 384-dim, English only. The alternative,
# bge-m3, is 568M parameters and 1024-dim, built for large multilingual
# corpora, and roughly 15x slower on CPU. For 542 English chunks inside one
# technical domain the retrieval quality difference is marginal, and the
# cross-encoder reranker does most of the work on final precision anyway.
# Paying half an hour of CPU per rebuild for that margin is a bad trade at
# this corpus size.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
EMBED_BATCH_SIZE = 32

# Chunks average about 3700 characters, roughly 900 tokens, so embedding at
# full length costs most of the build. Truncating at 512 tokens keeps the
# contextual header and the opening of the body, which is where the topical
# signal that makes a chunk findable sits. The tail is not lost to the
# system: it is still in the payload body, still scored by BM25, and still
# read in full by the cross-encoder at rerank time.
MAX_SEQ_LENGTH = 512

BUILD_COMMAND = "python -m src.retrieval.index"


class ChunkRecord(BaseModel):
    """One row of chunks.jsonl. Mirrors what the ingest stage wrote."""

    chunk_id: str
    source_id: str
    pages: list[int]
    tier: str
    tags: list[str]
    header: str
    body: str
    embed_text: str
    char_count: int


def load_chunks() -> list[ChunkRecord]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} is missing. Run the ingestion stage before building indexes."
        )
    with CHUNKS_PATH.open() as f:
        return [ChunkRecord.model_validate_json(line) for line in f if line.strip()]


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping decimals and hyphenated forms
    intact so "7.3" and "no-tillage" survive as single tokens. Numbers are
    kept deliberately: half the value of the sparse side of this index is
    matching a published figure the dense model would smooth over.
    """
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def embedding_model():
    """Load the dense encoder once per process. Imported lazily so that
    importing this module stays cheap.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def indexes_exist() -> bool:
    return _qdrant_collection_ready() and BM25_PATH.exists()


def _qdrant_collection_ready() -> bool:
    if not QDRANT_PATH.exists():
        return False
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(path=str(QDRANT_PATH))
    except Exception:
        # An already-open client holds the local lock. Treat that as built:
        # the caller is a process that has the collection open already.
        return True
    try:
        return client.collection_exists(COLLECTION)
    finally:
        client.close()


def build_indexes(force: bool = False) -> None:
    """Build the dense and sparse indexes from chunks.jsonl.

    Skips when both already exist unless force=True. Progress is printed at
    every stage: the first run downloads model weights before it embeds
    anything, and silence is indistinguishable from a hang.
    """
    if indexes_exist() and not force:
        print(f"indexes already present at {QDRANT_PATH} and {BM25_PATH}; pass force=True to rebuild")
        return

    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    _build_bm25(chunks)
    _build_dense(chunks, force=force)
    print("done")


def _build_bm25(chunks: list[ChunkRecord]) -> None:
    """Persist the tokenised corpus and its chunk_id ordering.

    The BM25 object itself is not pickled. rank_bm25 rebuilds its statistics
    from the token lists in well under a second, and storing the tokens keeps
    the artifact readable by anything that can unpickle a list of lists
    rather than tied to one library version.
    """
    print("tokenising for BM25 ...", flush=True)
    payload = {
        "chunk_ids": [c.chunk_id for c in chunks],
        "source_ids": [c.source_id for c in chunks],
        "tiers": [c.tier for c in chunks],
        "tags": [c.tags for c in chunks],
        "tokens": [tokenize(c.embed_text) for c in chunks],
    }
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {BM25_PATH} ({len(chunks)} documents)")


def _build_dense(chunks: list[ChunkRecord], force: bool) -> None:
    from qdrant_client import QdrantClient, models

    print(f"loading embedding model {DENSE_MODEL} (first run downloads the weights) ...", flush=True)
    model = embedding_model()

    print(f"embedding {len(chunks)} chunks ...", flush=True)
    started = time.time()
    vectors = model.encode(
        [c.embed_text for c in chunks],
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"embedded in {time.time() - started:.0f}s, shape {vectors.shape}")

    if force and QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_PATH))
    try:
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "source_id": c.source_id,
                    "tier": c.tier,
                    "tags": c.tags,
                    "pages": c.pages,
                    "body": c.body,
                },
            )
            for i, c in enumerate(chunks)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        count = client.count(COLLECTION).count
        print(f"upserted {count} points into collection '{COLLECTION}' at {QDRANT_PATH}")
    finally:
        client.close()


if __name__ == "__main__":
    build_indexes()
