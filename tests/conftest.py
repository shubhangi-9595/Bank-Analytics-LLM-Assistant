"""Shared test fixtures, built once per test run from SYNTHETIC data (offline, a few seconds).

- `bank`: database + data dictionary index + a few sample benchmark questions
- `demo`: `bank` plus a trained default model (for model, assistant and LLM-path tests)

Imports happen inside the fixtures, so a test file only needs the modules it actually uses.
"""
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def bank(tmp_path_factory):
    from bank_assistant.config import demo_settings
    from bank_assistant.data.build_db import build
    from bank_assistant.data.sample import generate
    from bank_assistant.rag.embeddings import get_embedder
    from bank_assistant.rag.index import build_index, make_chunks

    s = demo_settings(tmp_path_factory.mktemp("bank"))
    build(generate(s.raw_dir / "berka", n_accounts=250), s.db_path)
    build_index(make_chunks(s.knowledge_dir), s.index_dir, get_embedder(s, "tfidf"))
    shutil.copy(FIXTURES / "bird_sample.json", s.bird_questions)
    return replace(s, llm_provider="offline", cache_dir=tmp_path_factory.mktemp("cache"))


@pytest.fixture(scope="session")
def demo(bank):
    import joblib

    from bank_assistant.model.train import fit_final

    bank.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fit_final(bank.db_path, "logistic"), "kind": "logistic"}, bank.model_dir / "model.joblib")
    return bank
