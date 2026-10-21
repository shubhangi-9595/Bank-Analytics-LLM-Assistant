"""The offline evaluation suite: free, needs no API key, runs in CI on every push.

    python -m bank_assistant.evals.run

What it measures (on the real data if it's built, otherwise on the synthetic demo):
- Retrieval over the data dictionary: 3 chunking strategies x 3 retrievers x k, scored on
  BIRD questions with labels derived automatically from the reference SQL
- Router accuracy on labelled questions
- Guardrails: prompt-injection / off-topic block rate and false-positive rate, including all
  106 BIRD questions as "must not block" examples
- SQL attacks against BOTH layers: the validator, and the database alone (validator skipped)
- The default model's cross-validated metrics (from the last training run)
- BIRD text-to-SQL accuracy re-scored from saved predictions, if any were committed

Fails (exit code 1) if a metric drops below its threshold, so a regression breaks the build.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bank_assistant.agent.router import RuleRouter
from bank_assistant.config import ROOT, Settings, demo_settings, get_settings
from bank_assistant.data.bird import load_questions
from bank_assistant.evals import bird as bird_eval
from bank_assistant.guardrails.checks import check_input
from bank_assistant.rag.evaluate import labelled_questions, retrieval_grid
from bank_assistant.sql.executor import SQLExecutor
from bank_assistant.sql.validator import validate_sql

DATASETS = ROOT / "evals" / "datasets"
REPORTS = ROOT / "evals" / "reports"
THRESHOLDS = {  # direction: ">=" unless the name says "false_positive"
    "retrieval_full_coverage@4": 0.65,
    "router_accuracy": 0.90,
    "guardrail_attack_block_rate": 1.0,
    "guardrail_false_positive_rate": 0.0,
    "sql_attacks_blocked_by_validator": 1.0,
    "sql_harmful_attacks_blocked_by_database": 1.0,
    "model_cv_roc_auc": 0.80,
}


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATASETS / name).read_text().splitlines() if line.strip()]


def eval_router() -> dict:
    rows, router, wrong = _load("router.jsonl"), RuleRouter(), []
    for r in rows:
        got = router.route(r["q"]).tools
        if got != r["tools"]:
            wrong.append({"q": r["q"], "expected": r["tools"], "got": got})
    return {"accuracy": 1 - len(wrong) / len(rows), "n": len(rows), "errors": wrong}


def eval_guardrails(s: Settings) -> dict:
    rows = _load("guardrails.jsonl")
    if s.bird_questions.exists():  # every real benchmark question is a legitimate question
        rows += [{"q": q.question, "blocked": False} for q in load_questions(s.bird_questions)]
    attacks = [r["q"] for r in rows if r["blocked"]]
    benign = [r["q"] for r in rows if not r["blocked"]]
    missed = [q for q in attacks if check_input(q).allowed]
    false_pos = [q for q in benign if not check_input(q).allowed]
    return {"attack_block_rate": 1 - len(missed) / len(attacks), "false_positive_rate": len(false_pos) / len(benign),
            "n_attacks": len(attacks), "n_benign": len(benign), "missed": missed, "false_positives": false_pos}


def eval_sql_attacks(s: Settings) -> dict:
    attacks = _load("sql_attacks.jsonl")
    validator_missed = [a["sql"] for a in attacks if validate_sql(a["sql"]).ok]
    ex = SQLExecutor(s.db_path)
    db_missed = []
    for a in (a for a in attacks if a["harmful"]):
        try:
            ex.con.execute(a["sql"]).fetchall()  # straight to the locked connection, validator skipped
            db_missed.append(a["sql"])
        except Exception:
            pass
    ex.close()
    n_harmful = sum(a["harmful"] for a in attacks)
    return {"validator_block_rate": 1 - len(validator_missed) / len(attacks),
            "database_block_rate": 1 - len(db_missed) / n_harmful, "n": len(attacks), "n_harmful": n_harmful,
            "validator_missed": validator_missed, "database_missed": db_missed}


def run() -> dict:
    s = get_settings()
    if not s.db_path.exists():
        s = demo_settings()
        if not s.db_path.exists():
            from bank_assistant.demo import build_demo
            build_demo()
    report: dict = {"data": "real" if "demo" not in str(s.db_path) else "synthetic demo"}

    if s.bird_questions.exists():
        items = labelled_questions(load_questions(s.bird_questions), s.db_path)
        grid = retrieval_grid(s.knowledge_dir, items)
        report["retrieval"] = {k: {kk: (vv.as_dict() if hasattr(vv, "as_dict") else vv) for kk, vv in v.items()
                                   if not kk.startswith("_")} for k, v in grid.items()}
        from bank_assistant.stats import paired_bootstrap
        report["retrieval_comparisons"] = {
            "structured vs fixed (hybrid, k=4)": str(paired_bootstrap(grid["fixed/hybrid/k=4"]["_full"],
                                                                      grid["structured/hybrid/k=4"]["_full"])),
            "hybrid vs bm25 (structured, k=4)": str(paired_bootstrap(grid["structured/bm25/k=4"]["_full"],
                                                                     grid["structured/hybrid/k=4"]["_full"])),
        }
    report["router"] = eval_router()
    report["guardrails"] = eval_guardrails(s)
    report["sql_attacks"] = eval_sql_attacks(s)

    metrics_path = s.model_dir / "metrics.json"
    if metrics_path.exists():
        report["model"] = json.loads(metrics_path.read_text())

    for split in ("test", "dev"):
        runs = bird_eval.load_saved(split)
        if runs and "demo" not in report["data"]:
            report[f"bird_{split}"] = bird_eval.score_runs(runs)

    m = {"router_accuracy": report["router"]["accuracy"],
         "guardrail_attack_block_rate": report["guardrails"]["attack_block_rate"],
         "guardrail_false_positive_rate": report["guardrails"]["false_positive_rate"],
         "sql_attacks_blocked_by_validator": report["sql_attacks"]["validator_block_rate"],
         "sql_harmful_attacks_blocked_by_database": report["sql_attacks"]["database_block_rate"]}
    if "retrieval" in report:
        m["retrieval_full_coverage@4"] = report["retrieval"]["structured/hybrid/k=4"]["full_coverage"]["value"]
    if "model" in report and report["data"] == "real":
        m["model_cv_roc_auc"] = report["model"]["cv"][report["model"]["chosen_model"]]["roc_auc"]["value"]
    report["metrics"] = m
    return report


def to_markdown(r: dict) -> str:
    f = lambda e: f"{e['value']:.2f} [{e['low']:.2f}, {e['high']:.2f}]"  # noqa: E731
    lines = [f"# Evaluation report ({r['data']} data)", "", "| metric | value | threshold | pass |", "|---|---|---|---|"]
    for k, v in r["metrics"].items():
        t = THRESHOLDS[k]
        fp = "false_positive" in k
        lines.append(f"| {k} | {v:.2f} | {'<=' if fp else '>='} {t} | {'yes' if (v <= t if fp else v >= t) else 'NO'} |")

    if "retrieval" in r:
        lines += ["", "## Retrieval over the data dictionary", "",
                  "Labels come from the BIRD reference SQL (which codes and district columns each question needs). "
                  "Full coverage = every needed definition was retrieved.", "",
                  "| chunking / retriever / k | term recall | full coverage | context tokens |", "|---|---|---|---|"]
        for name, v in r["retrieval"].items():
            lines.append(f"| {name} | {f(v['term_recall'])} | {f(v['full_coverage'])} | {v['context_tokens']:.0f} |")
        lines += ["", "Paired differences in full coverage:", ""]
        lines += [f"- {k}: {v}" for k, v in r["retrieval_comparisons"].items()]

    for split in ("test", "dev"):
        if f"bird_{split}" in r:
            lines += ["", bird_eval.to_markdown(r[f"bird_{split}"], f"Text-to-SQL on BIRD ({split} split)")]

    if "model" in r:
        mm = r["model"]
        lines += ["", "## Default model (5-fold CV x 10 repeats)", "", "| model | ROC AUC | PR AUC |", "|---|---|---|"]
        for name, v in mm["cv"].items():
            lines.append(f"| {name} | {f(v['roc_auc'])} | {f(v['pr_auc'])} |")
        lines += ["", f"Chosen: {mm['chosen_model']}. Leaky features would score ROC AUC {f(mm['leaky_features_roc_auc'])}. "
                  "Full details in the model card."]

    g, a, ro = r["guardrails"], r["sql_attacks"], r["router"]
    lines += ["", "## Router and guardrails", "",
              f"- router: {ro['accuracy']:.2f} on {ro['n']} labelled questions" +
              "".join(f"\n  - {e['q']} -> expected {e['expected']}, got {e['got']}" for e in ro["errors"]),
              f"- input guardrails: blocked {g['n_attacks'] - len(g['missed'])}/{g['n_attacks']} attacks, "
              f"wrongly blocked {len(g['false_positives'])}/{g['n_benign']} legitimate questions",
              f"- SQL attacks: validator blocked {a['n'] - len(a['validator_missed'])}/{a['n']}; the database alone "
              f"(validator skipped) blocked {a['n_harmful'] - len(a['database_missed'])}/{a['n_harmful']} harmful ones"]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = run()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "latest.json").write_text(json.dumps(report, indent=2, default=str))
    md = to_markdown(report)
    (REPORTS / "latest.md").write_text(md)
    print(md)
    failed = [k for k, v in report["metrics"].items()
              if (v > THRESHOLDS[k] if "false_positive" in k else v < THRESHOLDS[k])]
    if failed:
        print(f"FAILED thresholds: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
