"""Embeddings: turn text into vectors so "similar meaning" becomes "nearby vectors".

Three interchangeable embedders:
- TfidfEmbedder   : sparse keyword weights (SVD-compressed). Free, offline, the baseline.
                    Only matches shared words: "won't refund" != "refused reimbursement".
- LocalEmbedder   : sentence-transformers model on your laptop. Free, real semantic vectors.
- AzureEmbedder   : text-embedding-3-small via Azure OpenAI. Best quality, costs credit
                    (~$0.02 per 1M tokens; embedding 3,000 complaints is well under a cent).

All return L2-normalised float32 vectors, so cosine similarity == dot product.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from bank_assistant.config import Settings, get_settings
from bank_assistant.llm.cache import DiskCache


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class Embedder:
    name = "base"
    needs_fit = False

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def save(self, path: Path) -> None:
        pass


class TfidfEmbedder(Embedder):
    """TF-IDF + truncated SVD (a.k.a. LSA). SVD turns 20k sparse dims into a few hundred
    dense ones and lets related words share dimensions a little."""

    name = "tfidf"
    needs_fit = True

    def __init__(self, dims: int = 256):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
        self.dims = dims
        self.svd = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD

        x = self.vec.fit_transform(texts)
        k = min(self.dims, x.shape[1] - 1, x.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(x) if k >= 2 else None
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        x = self.vec.transform(texts)
        return _normalise(self.svd.transform(x) if self.svd is not None else x.toarray())

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


class LocalEmbedder(Embedder):
    name = "local"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.name = f"local:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 500))


class AzureEmbedder(Embedder):
    name = "azure"

    def __init__(self, settings: Settings, budget=None):
        from bank_assistant.llm.budget import Budget
        from bank_assistant.llm.client import azure_v1_client

        self.client = azure_v1_client(settings)
        self.deployment = settings.azure_embedding_deployment
        self.budget = budget or Budget(settings.max_spend_usd)
        self.name = f"azure:{self.deployment}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):  # batch: fewer requests, kinder to rate limits
            self.budget.check()
            resp = self.client.embeddings.create(model=self.deployment, input=texts[i:i + 64])
            self.budget.record("text-embedding-3-small", resp.usage.prompt_tokens)
            out.extend(d.embedding for d in resp.data)
        return _normalise(np.array(out))


class CachedEmbedder(Embedder):
    """Wraps a paid/slow embedder: each unique text is embedded once, ever."""

    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        self.cache = DiskCache(cache_dir / "embeddings.sqlite")

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [DiskCache.key(self.inner.name, t) for t in texts]
        found = {i: self.cache.get(k) for i, k in enumerate(keys)}
        missing = [i for i, v in found.items() if v is None]
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing])
            for i, vec in zip(missing, fresh):
                found[i] = vec.astype(np.float32).tobytes()
                self.cache.set(keys[i], found[i])
        return np.stack([np.frombuffer(found[i], dtype=np.float32) for i in range(len(texts))])


def get_embedder(settings: Settings | None = None, provider: str | None = None, budget=None) -> Embedder:
    s = settings or get_settings()
    provider = provider or s.embedding_provider
    if provider == "tfidf":
        return TfidfEmbedder()
    if provider == "local":
        return CachedEmbedder(LocalEmbedder(s.local_embedding_model), s.cache_dir)
    if provider == "azure":
        return CachedEmbedder(AzureEmbedder(s, budget), s.cache_dir)
    raise ValueError(f"unknown embedding provider: {provider}")


def load_fitted(path: Path) -> Embedder:
    with open(path, "rb") as f:
        return pickle.load(f)
