"""Text-to-SQL evaluation on the BIRD benchmark's 106 questions about this bank.

    python -m bank_assistant.evals.bird --modes none,retrieved --split dev     # try ideas (cheap)
    python -m bank_assistant.evals.bird --modes none,retrieved,full,oracle --split test
    python -m bank_assistant.evals.bird --score-only                           # re-score saved runs, free

Metric: execution accuracy (EX), as defined by BIRD: a prediction is correct when running it
returns the same SET of rows as the reference query. Different SQL that returns the same
answer counts as correct.

Every run saves its predictions to evals/predictions/<model>__<mode>__<split>.json. They're
small, so commit them: CI and anyone reading the repo can re-score them without calling the LLM.

Statistics: 95% bootstrap CIs per mode, and PAIRED bootstrap differences between modes on the
same questions (e.g. "retrieved notes vs no notes: +0.21 [+0.10, +0.32]").
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from bank_assistant.config import ROOT, Settings, get_settings
from bank_assistant.data.bird import BirdQuestion, load_questions
from bank_assistant.rag.chunking import count_tokens
from bank_assistant.rag.index import load_retriever, make_chunks
from bank_assistant.sql.executor import SQLExecutor, schema_ddl
from bank_assistant.sql.text_to_sql import LLMTextToSQL
from bank_assistant.stats import bootstrap_ci, paired_bootstrap

PREDICTIONS = ROOT / "evals" / "predictions"
MODES = ("none", "retrieved", "full", "oracle")


def execution_match(con: sqlite3.Connection, predicted_rows, gold_sql: str) -> bool:
    gold = set(con.execute(gold_sql).fetchall())
    return predicted_rows is not None and set(predicted_rows) == gold


def run_mode(t2s: LLMTextToSQL, gold_con: sqlite3.Connection, questions: list[BirdQuestion], mode: str) -> list[dict]:
    out = []
    for i, q in enumerate(questions, 1):
        attempt = t2s.generate_and_run(q.question, mode=mode, oracle=q.evidence, row_limit=None)
        rows = attempt.result.rows if attempt.result is not None and attempt.result.ok else None
        out.append({"qid": q.qid, "split": q.split, "difficulty": q.difficulty, "sql": attempt.sql,
                    "error": attempt.result.error if attempt.result is not None else "no SQL returned",
                    "repaired": attempt.repaired, "notes_tokens": count_tokens(attempt.notes) if attempt.notes else 0,
                    "correct": execution_match(gold_con, rows, q.gold_sql)})
        print(f"\r  {mode}: {i}/{len(questions)}  correct so far {sum(p['correct'] for p in out)}", end="", flush=True)
    print()
    return out


def score_runs(runs: dict[str, list[dict]]) -> dict:
    """runs: {mode: predictions}. All runs must cover the same questions."""
    report = {"modes": {}, "comparisons": {}}
    for mode, preds in runs.items():
        preds = sorted(preds, key=lambda p: p["qid"])
        by_diff = {}
        for d in ("simple", "moderate", "challenging"):
            sub = [p["correct"] for p in preds if p["difficulty"] == d]
            if sub:
                by_diff[d] = bootstrap_ci(sub).as_dict()
        report["modes"][mode] = {"ex": bootstrap_ci([p["correct"] for p in preds]).as_dict(), "by_difficulty": by_diff,
                                 "n": len(preds), "self_corrected": sum(p["repaired"] for p in preds),
                                 "errors": sum(bool(p["error"]) for p in preds),
                                 "avg_notes_tokens": sum(p["notes_tokens"] for p in preds) / max(len(preds), 1)}
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa = [p["correct"] for p in sorted(runs[a], key=lambda p: p["qid"])]
            pb = [p["correct"] for p in sorted(runs[b], key=lambda p: p["qid"])]
            if len(pa) == len(pb):
                report["comparisons"][f"{b} vs {a}"] = str(paired_bootstrap(pa, pb))
    return report


def to_markdown(report: dict, title: str) -> str:
    f = lambda e: f"{e['value']:.2f} [{e['low']:.2f}, {e['high']:.2f}]"  # noqa: E731
    lines = [f"## {title}", "", "| notes in prompt | execution accuracy | simple | moderate | challenging | "
             "avg note tokens | self-corrected |", "|---|---|---|---|---|---|---|"]
    for mode, m in report["modes"].items():
        d = m["by_difficulty"]
        lines.append(f"| {mode} | {f(m['ex'])} | " + " | ".join(f(d[k]) if k in d else "-" for k in
                     ("simple", "moderate", "challenging")) + f" | {m['avg_notes_tokens']:.0f} | {m['self_corrected']} |")
    if report["comparisons"]:
        lines += ["", "Paired differences (same questions, 95% CI, p-value):", ""]
        lines += [f"- {k}: {v}" for k, v in report["comparisons"].items()]
    return "\n".join(lines) + "\n"


def load_saved(split: str, model: str | None = None) -> dict[str, list[dict]]:
    runs = {}
    for path in sorted(PREDICTIONS.glob(f"*__*__{split}.json")):
        m, mode, _ = path.stem.split("__")
        if model is None or m == model:
            runs[mode] = json.loads(path.read_text())
    return dict(sorted(runs.items(), key=lambda kv: MODES.index(kv[0]) if kv[0] in MODES else 99))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="none,retrieved")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--score-only", action="store_true", help="re-score saved predictions without calling the LLM")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    args = ap.parse_args()
    s = get_settings()

    if not args.score_only:
        from bank_assistant.llm.client import LLMClient

        s = replace(s, llm_provider="azure") if s.llm_provider == "offline" else s
        llm = LLMClient(s)
        questions = [q for q in load_questions(s.bird_questions) if args.split == "all" or q.split == args.split]
        questions = questions[: args.limit] if args.limit else questions
        t2s = LLMTextToSQL(llm, SQLExecutor(s.db_path, row_limit=None, timeout_s=30), schema_ddl(s.db_path),
                           load_retriever(s.index_dir, "hybrid", s), [c.text for c in make_chunks(s.knowledge_dir)],
                           s.top_k)
        gold_con = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        for mode in args.modes.split(","):
            preds = run_mode(t2s, gold_con, questions, mode)
            suffix = args.split if not args.limit else f"{args.split}-first{args.limit}"
            (PREDICTIONS / f"{s.azure_chat_model}__{mode}__{suffix}.json").write_text(json.dumps(preds, indent=1))
        print(f"LLM cost: {llm.budget.summary()}")

    report = score_runs(load_saved(args.split))
    if report["modes"]:
        print(to_markdown(report, f"BIRD `financial` questions, {args.split} split"))
    else:
        print(f"no saved predictions for split '{args.split}' in {PREDICTIONS}")


if __name__ == "__main__":
    main()
