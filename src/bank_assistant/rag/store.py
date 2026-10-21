"""A minimal vector store: a matrix of vectors + metadata, searched by dot product.

This is what Chroma / LanceDB / Azure AI Search do at heart. Exact search over a
few thousand chunks takes about a millisecond, so we don't need an ANN index
(HNSW, IVF) until roughly a million vectors. Writing it ourselves makes the
concept visible; swapping in LanceDB later is a ~20 line change.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bank_assistant.rag.chunking import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        assert len(chunks) == len(vectors), "one vector per chunk"
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder_name = embedder_name

    def search(self, query_vec: np.ndarray, k: int = 5, where: dict | None = None) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vec.astype(np.float32)
        if where:  # metadata filter, applied before ranking ("pre-filtering")
            mask = np.array([all(c.metadata.get(f) == v for f, v in where.items()) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        (directory / "meta.json").write_text(json.dumps({"embedder": self.embedder_name, "n": len(self.chunks)}))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        with open(directory / "chunks.jsonl") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        meta = json.loads((directory / "meta.json").read_text())
        return cls(chunks, np.load(directory / "vectors.npy"), meta["embedder"])
