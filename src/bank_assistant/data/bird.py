"""The BIRD benchmark questions about this bank.

Each item: question, evidence (a hint written by the annotators, e.g. "A11 refers to average
salary"), gold SQL and difficulty. We split them once into DEV (for trying prompt ideas) and
TEST (reported only at the end), so the headline number isn't tuned on the questions it's measured on.
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BirdQuestion:
    qid: int
    question: str
    evidence: str
    gold_sql: str
    difficulty: str
    split: str


def load_questions(path: str | Path, dev_fraction: float = 0.3, seed: int = 42) -> list[BirdQuestion]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = [q for q in raw if q.get("db_id", "financial") == "financial"]
    ids = sorted(q["question_id"] for q in raw)
    rng = random.Random(seed)
    dev = set(rng.sample(ids, round(len(ids) * dev_fraction)))
    return [BirdQuestion(q["question_id"], q["question"], q.get("evidence", ""), q["SQL"],
                         q.get("difficulty", "unknown"), "dev" if q["question_id"] in dev else "test")
            for q in sorted(raw, key=lambda q: q["question_id"])]


def run_gold(con: sqlite3.Connection, sql: str):
    return con.execute(sql).fetchall()


def check_gold_sql(db_path: str | Path, questions_path: str | Path) -> tuple[int, list[tuple[int, str]]]:
    """Every reference query should run on our database (a schema-compatibility check)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ok, failed = 0, []
    for q in load_questions(questions_path):
        try:
            run_gold(con, q.gold_sql)
            ok += 1
        except sqlite3.Error as e:
            failed.append((q.qid, str(e)))
    con.close()
    return ok, failed


def compare_with_bird_db(ours: str | Path, theirs: str | Path, questions_path: str | Path) -> tuple[int, list[int]]:
    """Optional check against BIRD's own SQLite file: do the reference queries give the same answers?"""
    a = sqlite3.connect(f"file:{ours}?mode=ro", uri=True)
    b = sqlite3.connect(f"file:{theirs}?mode=ro", uri=True)
    same, different = 0, []
    for q in load_questions(questions_path):
        try:
            ra, rb = set(run_gold(a, q.gold_sql)), set(run_gold(b, q.gold_sql))
        except sqlite3.Error:
            ra, rb = None, ()
        if ra == rb:
            same += 1
        else:
            different.append(q.qid)
    return same, different
