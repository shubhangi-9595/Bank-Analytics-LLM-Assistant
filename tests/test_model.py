import sqlite3
import shutil

from bank_assistant.model.explain import LoanScorer
from bank_assistant.model.features import build_features
from bank_assistant.model.train import evaluate


def test_features_ignore_everything_after_the_loan(demo, tmp_path):
    db = tmp_path / "bank.sqlite"
    shutil.copy(demo.db_path, db)
    before = build_features(db).set_index("loan_id")
    con = sqlite3.connect(db)
    loan_id, account_id, date = con.execute("SELECT loan_id, account_id, date FROM loan LIMIT 1").fetchone()
    # add a disaster AFTER the loan: a big negative balance with sanction interest
    con.execute("INSERT INTO trans VALUES (999999, ?, date(?, '+30 days'), 'VYDAJ', '', 50000, -40000, "
                "'SANKC. UROK', NULL, NULL)", (account_id, date))
    con.commit()
    after = build_features(db).set_index("loan_id")
    cols = ["balance_min", "n_sanction_interest", "n_negative_balance", "balance_last"]
    assert before.loc[loan_id, cols].equals(after.loc[loan_id, cols])
    leaky = build_features(db, leaky=True).set_index("loan_id")
    assert leaky.loc[loan_id, "n_sanction_interest"] > before.loc[loan_id, "n_sanction_interest"]


def test_evaluation_report_has_intervals_and_beats_chance(demo):
    r = evaluate(demo.db_path, repeats=2)
    auc = r["cv"]["logistic"]["roc_auc"]
    assert auc["low"] <= auc["value"] <= auc["high"] and auc["value"] > 0.6
    assert r["chosen_model"] in ("logistic", "boosting")
    assert r["leaky_features_roc_auc"]["value"] >= auc["value"] - 0.05


def test_scorer_explains_a_loan(demo):
    s = LoanScorer(demo.model_dir / "model.joblib", demo.db_path)
    loan_id = int(s.features.index[0])
    result = s.score(loan_id)
    assert 0 <= result.probability <= 1 and len(result.drivers) == 4
    assert "default risk" in result.summary()
    assert s.score_account(int(s.features.loc[loan_id, "account_id"])).loan_id == loan_id
