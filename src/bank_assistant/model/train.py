"""Train and evaluate the loan default model.

    python -m bank_assistant.model.train

Evaluation protocol (682 loans, 76 defaults, so every number gets a confidence interval):
1. Repeated stratified 5-fold cross-validation (10 repeats). Each loan gets an out-of-fold
   prediction averaged over repeats; metrics and 95% bootstrap CIs are computed on those.
2. Models: a one-feature rule (lowest balance in the past year), logistic regression, and
   gradient boosting. Models are compared with a PAIRED bootstrap on the same loans.
3. A time-based check: train on loans granted up to 1996, test on 1997. (1998 loans are too
   recent to have defaulted yet: 4 defaults in 158, so they'd make any model look bad.)
4. A leakage check: the same pipeline on features that peek at transactions after the loan.
5. Calibration: does a predicted 30% mean roughly 30% of those loans default?

Model choice rule, decided before looking at results: use logistic regression (simpler,
explainable per loan) unless boosting is better with a CI that excludes zero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_assistant.config import ROOT, get_settings
from bank_assistant.model.features import CATEGORICAL, NUMERIC, build_features, feature_matrix
from bank_assistant.stats import bootstrap_metric, paired_bootstrap_metric

SEED = 0


def make_model(kind: str) -> Pipeline:
    if kind == "logistic":
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
        ])
        return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.3, max_iter=2000))])
    if kind == "boosting":
        pre = ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ], sparse_threshold=0)
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                             l2_regularization=1.0, random_state=SEED)
        return Pipeline([("pre", pre), ("clf", clf)])
    raise ValueError(kind)


def out_of_fold(kind: str, X: pd.DataFrame, y: np.ndarray, repeats: int = 10) -> np.ndarray:
    preds = np.zeros((repeats, len(y)))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = make_model(kind).fit(X.iloc[tr], y[tr])
        preds[i // 5, te] = m.predict_proba(X.iloc[te])[:, 1]
    return preds.mean(axis=0)


def metrics(y, p, probabilities: bool = True) -> dict:
    top = np.argsort(-p)[: int(round(0.2 * len(p)))]
    return {
        "roc_auc": bootstrap_metric(y, p, roc_auc_score).as_dict(),
        "pr_auc": bootstrap_metric(y, p, average_precision_score).as_dict(),
        "brier": float(brier_score_loss(y, p)) if probabilities else None,
        "defaults_caught_in_riskiest_20pct": float(y[top].sum() / y.sum()),
    }


def calibration_table(y, p, bins: int = 5) -> list[dict]:
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"predicted": float(p[chunk].mean()), "observed": float(y[chunk].mean()), "n": int(len(chunk))})
    return out


def evaluate(db_path: Path, repeats: int = 10) -> dict:
    df = build_features(db_path)
    X, y = feature_matrix(df), df["default"].to_numpy()
    rule = -df["balance_min"].fillna(df["balance_min"].median()).to_numpy()  # lower balance = riskier
    oof = {"rule_min_balance": rule, "logistic": out_of_fold("logistic", X, y, repeats),
           "boosting": out_of_fold("boosting", X, y, repeats)}

    report = {"n_loans": int(len(y)), "n_defaults": int(y.sum()), "default_rate": float(y.mean()),
              "cv": {name: metrics(y, p, probabilities=name != "rule_min_balance")  # the rule only ranks
                     for name, p in oof.items()}}

    cmp = lambda a, b: str(paired_bootstrap_metric(y, oof[a], oof[b], roc_auc_score))  # noqa: E731
    report["comparisons_roc_auc"] = {"logistic_vs_rule": cmp("rule_min_balance", "logistic"),
                                     "boosting_vs_logistic": cmp("logistic", "boosting")}
    diff = paired_bootstrap_metric(y, oof["logistic"], oof["boosting"], roc_auc_score)
    report["chosen_model"] = "boosting" if diff.significant and diff.diff > 0 else "logistic"
    report["calibration"] = {k: calibration_table(y, oof[k]) for k in ("logistic", "boosting")}

    # Time-based check: train <= 1996, test 1997
    train, test = df["loan_year"] <= 1996, df["loan_year"] == 1997
    report["time_split"] = {"train_loans": int(train.sum()), "test_loans": int(test.sum()),
                            "test_defaults": int(y[test].sum()),
                            "test_status_counts": df.loc[test, "status"].value_counts().to_dict(),
                            "all_status_counts": df["status"].value_counts().to_dict()}
    for kind in ("logistic", "boosting"):
        m = make_model(kind).fit(X[train], y[train])
        report["time_split"][kind] = bootstrap_metric(y[test], m.predict_proba(X[test])[:, 1], roc_auc_score).as_dict()

    # Leakage check
    leaky = build_features(db_path, leaky=True)
    report["leaky_features_roc_auc"] = bootstrap_metric(
        y, out_of_fold(report["chosen_model"], feature_matrix(leaky), y, max(2, repeats // 5)), roc_auc_score).as_dict()
    return report


def fit_final(db_path: Path, kind: str) -> Pipeline:
    df = build_features(db_path)
    return make_model(kind).fit(feature_matrix(df), df["default"].to_numpy())


def write_model_card(report: dict, path: Path) -> None:
    r, cv = report, report["cv"]
    f = lambda e: f"{e['value']:.3f} [{e['low']:.3f}, {e['high']:.3f}]"  # noqa: E731
    lines = [
        "# Model card: loan default model", "",
        f"**Data:** {r['n_loans']} loans from the Czech bank (1993-1998), {r['n_defaults']} defaults "
        f"({r['default_rate']:.1%}). Default = status B (finished, not paid) or D (running, in debt).",
        "**Features:** loan terms, account and owner attributes, district demographics, and the account's "
        "transactions in the 365 days BEFORE the loan. Nothing after the loan date.",
        f"**Chosen model:** {r['chosen_model']} (rule fixed in advance: logistic unless boosting is "
        "significantly better).", "",
        "## Cross-validated performance (5-fold x 10 repeats, 95% bootstrap CI)", "",
        "| model | ROC AUC | PR AUC | Brier | defaults caught in riskiest 20% |", "|---|---|---|---|---|",
    ]
    for name, m in cv.items():
        brier = f"{m['brier']:.3f}" if m["brier"] is not None else "n/a"
        lines.append(f"| {name} | {f(m['roc_auc'])} | {f(m['pr_auc'])} | {brier} | "
                     f"{m['defaults_caught_in_riskiest_20pct']:.0%} |")
    lines += ["", f"PR AUC of a random model = default rate = {r['default_rate']:.3f}.", "",
              "Paired comparisons (difference in ROC AUC, 95% CI, p-value):", ""]
    lines += [f"- {k}: {v}" for k, v in r["comparisons_roc_auc"].items()]
    ts = r["time_split"]
    lines += ["", "## Time-based check", "",
              f"Trained on {ts['train_loans']} loans granted up to 1996, tested on {ts['test_loans']} loans from 1997 "
              f"({ts['test_defaults']} defaults): logistic ROC AUC {f(ts['logistic'])}, boosting {f(ts['boosting'])}.",
              "1998 loans are excluded from testing: they are too recent to have defaulted (label not mature).",
              "",
              f"Why this can beat cross-validation: the 1997 defaults are mostly status D "
              f"({ts['test_status_counts'].get('D', 0)} D vs {ts['test_status_counts'].get('B', 0)} B), i.e. clients already "
              "in debt early in the loan, whose weak pre-loan history makes them easier to spot. "
              f"Over all years it's {ts['all_status_counts'].get('D', 0)} D vs {ts['all_status_counts'].get('B', 0)} B. "
              "So the higher score reflects an easier test mix, not a model that improves over time; "
              "with this few defaults the interval is also wide.",
              "", "## Leakage check", "",
              f"Same model on features that include transactions AFTER the loan: ROC AUC {f(r['leaky_features_roc_auc'])}. "
              "The jump is the outcome leaking into the features, which is why the window stops at the loan date.",
              "", "## Calibration (out-of-fold, 5 equal-size bins)", "", "| model | predicted | observed | n |",
              "|---|---|---|---|"]
    for name, rows in r["calibration"].items():
        lines += [f"| {name} | {b['predicted']:.2f} | {b['observed']:.2f} | {b['n']} |" for b in rows]
    lines += ["", "## Limitations", "",
              "- 76 defaults is a small sample: intervals are wide, and small differences between models are noise.",
              "- Running loans (status C) may still default, so some 'good' labels are wrong, more so for recent loans.",
              "- One bank, 1990s Czech Republic: not representative of other populations or today.",
              "- Not for real credit decisions; gender is in the features, which real lending law may prohibit."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    s = get_settings()
    repeats = int(sys.argv[sys.argv.index("--repeats") + 1]) if "--repeats" in sys.argv else 10
    report = evaluate(s.db_path, repeats)
    s.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fit_final(s.db_path, report["chosen_model"]), "kind": report["chosen_model"]},
                s.model_dir / "model.joblib")
    (s.model_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    write_model_card(report, s.model_dir / "model_card.md")
    if "demo" not in str(s.db_path):  # real data: keep a copy in docs/ so it's visible on GitHub
        (ROOT / "docs").mkdir(exist_ok=True)
        write_model_card(report, ROOT / "docs" / "model_card.md")
    print((s.model_dir / "model_card.md").read_text())


if __name__ == "__main__":
    main()
