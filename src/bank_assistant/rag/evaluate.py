"""Retrieval evaluation on the BIRD questions, fully offline and free.

For each question that needs dictionary terms (see labels.py), retrieve top-k chunks for the
question text and measure:
- term recall@k   : share of needed terms whose definition is in the retrieved text
- full coverage@k : share of questions where ALL needed terms were retrieved
                    (the number that matters: one missing code is enough to get the SQL wrong)
- context tokens  : how much text we send to the LLM (cost)
Compared across chunking strategies x retrievers, with bootstrap confidence intervals.
"""
from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bank_assistant.data.bird import BirdQuestion
from bank_assistant.rag.chunking import count_tokens
from bank_assistant.rag.embeddings import get_embedder
from bank_assistant.rag.index import build_index, make_chunks, retriever_from_store
from bank_assistant.rag.labels import needed_terms, schema_from_db
from bank_assistant.stats import bootstrap_ci


@dataclass
class LabelledQuestion:
    qid: int
    question: str
    terms: set[str]


def labelled_questions(questions: list[BirdQuestion], db_path: Path) -> list[LabelledQuestion]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    schema = schema_from_db(con)
    con.close()
    out = []
    for q in questions:
        terms = needed_terms(q.gold_sql, schema)
        if terms:
            out.append(LabelledQuestion(q.qid, q.question, terms))
    return out


def score(retriever, items: list[LabelledQuestion], k: int) -> dict:
    recalls, full, tokens = [], [], []
    for it in items:
        text = "\n".join(c.text for c, _ in retriever.retrieve(it.question, k))
        found = sum(t in text for t in it.terms)
        recalls.append(found / len(it.terms))
        full.append(float(found == len(it.terms)))
        tokens.append(count_tokens(text))
    return {"term_recall": bootstrap_ci(recalls), "full_coverage": bootstrap_ci(full),
            "context_tokens": sum(tokens) / len(tokens), "n": len(items), "_full": full}


def retrieval_grid(knowledge_dir: Path, items: list[LabelledQuestion], ks=(2, 4), embedder: str = "tfidf") -> dict:
    grid = {}
    with tempfile.TemporaryDirectory() as tmp:
        for strategy in ("fixed", "recursive", "structured"):
            chunks = make_chunks(knowledge_dir, strategy)
            emb = get_embedder(provider=embedder)
            store = build_index(chunks, Path(tmp) / strategy, emb)
            for kind in ("dense", "bm25", "hybrid"):
                r = retriever_from_store(store, emb, kind)
                for k in ks:
                    grid[f"{strategy}/{kind}/k={k}"] = score(r, items, k)
    return grid
