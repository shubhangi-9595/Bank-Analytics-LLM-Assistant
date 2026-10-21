"""Build and load the index over the data dictionary.

    python -m bank_assistant.rag.index                       # structured chunks, TF-IDF
    python -m bank_assistant.rag.index --embedder local      # free local embedding model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bank_assistant.config import Settings, get_settings
from bank_assistant.rag.chunking import Chunk, chunk_all
from bank_assistant.rag.documents import load_knowledge
from bank_assistant.rag.embeddings import Embedder, get_embedder, load_fitted
from bank_assistant.rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from bank_assistant.rag.store import VectorStore

DEFAULT_CHUNKING = {"structured": (256, 0), "recursive": (64, 8), "fixed": (64, 8)}


def make_chunks(knowledge_dir: Path, strategy: str = "structured") -> list[Chunk]:
    size, overlap = DEFAULT_CHUNKING[strategy]
    return chunk_all(load_knowledge(knowledge_dir), strategy, size, overlap)


def build_index(chunks: list[Chunk], out_dir: Path, embedder: Embedder) -> VectorStore:
    texts = [c.text for c in chunks]
    if embedder.needs_fit:
        embedder.fit(texts)
    store = VectorStore(chunks, embedder.embed(texts), embedder.name)
    store.save(out_dir)
    if embedder.needs_fit:
        embedder.save(out_dir / "embedder.pkl")
    return store


def retriever_from_store(store: VectorStore, embedder: Embedder, kind: str = "hybrid"):
    dense = DenseRetriever(store, embedder)
    if kind == "dense":
        return dense
    sparse = BM25Retriever(store.chunks)
    if kind == "bm25":
        return sparse
    return HybridRetriever(dense, sparse)


def load_retriever(index_dir: Path, kind: str = "hybrid", settings: Settings | None = None):
    s = settings or get_settings()
    store = VectorStore.load(index_dir)
    if (index_dir / "embedder.pkl").exists():
        embedder = load_fitted(index_dir / "embedder.pkl")
    else:
        embedder = get_embedder(s, provider=store.embedder_name.split(":")[0])
    return retriever_from_store(store, embedder, kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structured", choices=list(DEFAULT_CHUNKING))
    ap.add_argument("--embedder", default=None, help="tfidf | local | azure (default: EMBEDDING_PROVIDER)")
    args = ap.parse_args()
    s = get_settings()
    chunks = make_chunks(s.knowledge_dir, args.strategy)
    store = build_index(chunks, s.index_dir, get_embedder(s, args.embedder))
    (s.index_dir / "config.json").write_text(json.dumps({"strategy": args.strategy, "embedder": store.embedder_name}))
    print(f"indexed {len(chunks)} dictionary chunks ({args.strategy}) with {store.embedder_name} into {s.index_dir}")


if __name__ == "__main__":
    main()
