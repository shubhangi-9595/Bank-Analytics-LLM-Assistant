"""Build a small, fully offline demo environment from SYNTHETIC data.

    python -m bank_assistant.demo
    DEMO=1 python -m bank_assistant.agent.assistant "How risky is loan 3?"

Used by tests and as a quick start before downloading the real data. Everything is fake.
"""
from __future__ import annotations

from pathlib import Path

from bank_assistant.config import Settings, demo_settings
from bank_assistant.data.build_db import build
from bank_assistant.data.sample import generate
from bank_assistant.model.train import fit_final
from bank_assistant.rag.embeddings import get_embedder
from bank_assistant.rag.index import build_index, make_chunks


def build_demo(base: Path | None = None, n_accounts: int = 300) -> Settings:
    import joblib

    s = demo_settings(base)
    build(generate(s.raw_dir / "berka", n_accounts=n_accounts), s.db_path)
    build_index(make_chunks(s.knowledge_dir), s.index_dir, get_embedder(s, "tfidf"))
    s.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fit_final(s.db_path, "logistic"), "kind": "logistic"}, s.model_dir / "model.joblib")
    return s


if __name__ == "__main__":
    s = build_demo()
    print(f"demo ready in {s.data_dir}\nTry (PowerShell: $env:DEMO=\"1\"):\n"
          "  DEMO=1 python -m bank_assistant.agent.assistant \"How risky is loan 3?\"")
