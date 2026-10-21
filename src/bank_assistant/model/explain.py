"""Score a loan and explain the score: the tool the assistant calls.

For logistic regression the explanation is exact: the log-odds are a sum of
coefficient x (standardised feature value), so each feature's term is its contribution
relative to an average loan. For boosting we fall back to the global permutation-importance
ranking (SHAP would be the per-loan upgrade, see docs/learning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bank_assistant.model.features import CATEGORICAL, NUMERIC, build_features, feature_matrix

READABLE = {
    "amount": "loan amount", "duration": "loan duration", "payments": "monthly payment",
    "account_age_days": "account age", "owner_age": "owner's age", "n_account_users": "number of account users",
    "n_cards_before_loan": "credit cards held", "district_avg_salary": "district average salary",
    "district_unemployment_96": "district unemployment", "district_entrepreneurs": "district entrepreneurs",
    "district_crime_rate_96": "district crime rate", "n_trans": "number of transactions",
    "balance_mean": "average balance", "balance_min": "lowest balance", "balance_max": "highest balance",
    "balance_last": "balance just before the loan", "n_negative_balance": "times overdrawn",
    "n_sanction_interest": "penalty interest charges", "monthly_credits": "monthly income",
    "monthly_debits": "monthly spending", "payment_to_income": "loan payment / income",
    "cash_withdrawal_share": "share of cash withdrawals", "has_pension": "receives a pension",
}


@dataclass
class LoanScore:
    loan_id: int
    probability: float
    status: str
    drivers: list[dict] = field(default_factory=list)   # [{"feature", "value", "effect"}]

    def summary(self) -> str:
        lines = [f"Loan {self.loan_id}: estimated default risk {self.probability:.0%} "
                 f"(actual status in the data: {self.status})."]
        for d in self.drivers:
            arrow = "raises" if d["effect"] > 0 else "lowers"
            lines.append(f"- {d['feature']} = {d['value']} {arrow} the risk")
        return "\n".join(lines)


class LoanScorer:
    def __init__(self, model_path: Path, db_path: Path):
        bundle = joblib.load(model_path)
        self.model, self.kind = bundle["model"], bundle["kind"]
        self.features = build_features(db_path).set_index("loan_id")

    def score(self, loan_id: int, top: int = 4) -> LoanScore:
        if loan_id not in self.features.index:
            raise KeyError(f"no loan with id {loan_id}")
        row = self.features.loc[[loan_id]]
        X = feature_matrix(row.reset_index())
        p = float(self.model.predict_proba(X)[0, 1])
        return LoanScore(int(loan_id), p, str(row["status"].iloc[0]), self._drivers(X, top))

    def score_account(self, account_id: int) -> LoanScore:
        match = self.features.index[self.features["account_id"] == account_id]
        if len(match) == 0:
            raise KeyError(f"account {account_id} has no loan")
        return self.score(int(match[0]))

    def _drivers(self, X: pd.DataFrame, top: int) -> list[dict]:
        if self.kind != "logistic":
            return []
        pre, clf = self.model.named_steps["pre"], self.model.named_steps["clf"]
        z = pre.transform(X)[0]
        contrib = z * clf.coef_[0]
        names = NUMERIC + list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
        order = np.argsort(-np.abs(contrib))[:top]
        out = []
        for i in order:
            name = names[i]
            raw = X.iloc[0][name] if name in X.columns else 1
            value = f"{raw:,.0f}" if isinstance(raw, (int, float, np.number)) and abs(raw) >= 10 else (
                f"{raw:.2f}" if isinstance(raw, (float, np.floating)) else str(raw))
            level = "" if name not in NUMERIC else (" (above average)" if z[i] > 0.25 else
                                                    " (below average)" if z[i] < -0.25 else " (about average)")
            out.append({"feature": READABLE.get(name, name.replace("_", " ")), "value": value + level,
                        "effect": float(contrib[i])})
        return out
