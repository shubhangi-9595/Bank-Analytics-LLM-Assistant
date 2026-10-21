"""Retrieval: find the chunks most likely to answer a question.

- DenseRetriever : vector similarity. Good at meaning and paraphrase.
- BM25Retriever  : keyword scoring. Good at exact terms, ids, names ("Zelle", "case-9f2a...").
- HybridRetriever: runs both and merges with Reciprocal Rank Fusion (RRF), which
                   only uses ranks, so the two very different score scales don't matter.

Upgrade beyond this: a reranker (cross-encoder or a small LLM call) that rescores
the top ~20 hybrid results. See docs/learning/04-retrieval.md.
"""
from __future__ import annotations

import re

import numpy as np

from bank_assistant.rag.chunking import Chunk
from bank_assistant.rag.embeddings import Embedder
from bank_assistant.rag.store import VectorStore

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are was were be been it this that with at by from as i my me "
            "we our you your they their what which who how when where why do does did not no can".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class DenseRetriever:
    name = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store, self.embedder = store, embedder

    def retrieve(self, query: str, k: int = 5, where: dict | None = None) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed_query(query), k, where)


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokens(c.text) for c in chunks])

    def retrieve(self, query: str, k: int = 5, where: dict | None = None) -> list[tuple[Chunk, float]]:
        scores = np.asarray(self.bm25.get_scores(_tokens(query)), dtype=float)
        if where:
            mask = np.array([all(c.metadata.get(f) == v for f, v in where.items()) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, rrf_k: int = 60, pool: int = 20,
                 diversify: bool = True):
        self.dense, self.sparse, self.rrf_k, self.pool = dense, sparse, rrf_k, pool
        self.diversify = diversify

    def retrieve(self, query: str, k: int = 5, where: dict | None = None) -> list[tuple[Chunk, float]]:
        fused = self._fuse(query, where)
        if self.diversify:
            return mmr(fused[: self.pool], self.dense.store, k)
        return fused[:k]

    def _fuse(self, query: str, where: dict | None) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for results in (self.dense.retrieve(query, self.pool, where), self.sparse.retrieve(query, self.pool, where)):
            for rank, (chunk, _) in enumerate(results):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[chunk.chunk_id] = chunk
        best = sorted(fused, key=fused.get, reverse=True)
        return [(by_id[i], fused[i]) for i in best]


def mmr(results: list[tuple[Chunk, float]], store: VectorStore, k: int, lambda_: float = 0.6) -> list[tuple[Chunk, float]]:
    """Maximal Marginal Relevance: pick results that are relevant AND different from
    what's already picked. Without it, several near-identical chunks (e.g. overlapping fixed-size
    windows of the same dictionary entry) can fill every slot and push out a different entry the question also needs.

    score = lambda * relevance - (1 - lambda) * max_similarity_to_already_selected
    """
    if len(results) <= 1:
        return results[:k]
    index = {c.chunk_id: i for i, c in enumerate(store.chunks)}
    vecs = np.stack([store.vectors[index[c.chunk_id]] for c, _ in results])
    rel = np.array([s for _, s in results], dtype=float)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)  # put fused/BM25/cosine scores on one scale
    chosen = [0]
    while len(chosen) < min(k, len(results)):
        sim_to_chosen = (vecs @ vecs[chosen].T).max(axis=1)
        score = lambda_ * rel - (1 - lambda_) * sim_to_chosen
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [results[i] for i in chosen]


def dedupe_by_document(results: list[tuple[Chunk, float]], k: int) -> list[tuple[Chunk, float]]:
    """Keep the best chunk per document so 5 slots aren't spent on one long PDF."""
    seen, out = set(), []
    for chunk, score in results:
        if chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            out.append((chunk, score))
    return out[:k]
