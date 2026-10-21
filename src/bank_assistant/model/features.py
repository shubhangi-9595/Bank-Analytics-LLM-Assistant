"""Loan-level features for default prediction, built in SQL from the bank database.

THE rule: a feature may only use information available on the day the loan was granted.
Transactions are restricted to the 365 days BEFORE loan.date. Using the account's whole
history leaks the outcome: a client who later defaults gets charged sanction interest
after the loan (on average 6 times vs 0.08 for good loans), so a "leaky" model looks
excellent and is useless. `build_features(..., leaky=True)` builds that version on purpose,
to measure how much leakage inflates the scores.

Not used, on purpose:
- permanent orders (`order`): the table has no dates, and every loan account has an order
  for paying the loan itself (k_symbol 'UVER'), so its timing can't be checked
- loan.status: it IS the label
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 365

FEATURES_SQL = """
WITH owner AS (
    SELECT d.account_id, c.gender, c.birth_date
    FROM disp d JOIN client c USING (client_id)
    WHERE d.type = 'OWNER'
),
history AS (
    SELECT l.loan_id,
           count(*)                                              AS n_trans,
           avg(t.balance)                                        AS balance_mean,
           min(t.balance)                                        AS balance_min,
           max(t.balance)                                        AS balance_max,
           sum(t.balance < 0)                                    AS n_negative_balance,
           sum(t.k_symbol = 'SANKC. UROK')                       AS n_sanction_interest,
           sum(CASE WHEN t.type = 'PRIJEM' THEN t.amount ELSE 0 END) AS credits_total,
           sum(CASE WHEN t.type != 'PRIJEM' THEN t.amount ELSE 0 END) AS debits_total,
           sum(t.operation = 'VYBER')                            AS n_cash_withdrawals,
           sum(t.k_symbol = 'DUCHOD')                            AS n_pension,
           max(julianday(l.date) - julianday(t.date))            AS history_days
    FROM loan l JOIN trans t ON t.account_id = l.account_id
    WHERE {window}
    GROUP BY l.loan_id
),
last_balance AS (
    SELECT l.loan_id,
           (SELECT t.balance FROM trans t
            WHERE t.account_id = l.account_id AND t.date < l.date
            ORDER BY t.date DESC, t.trans_id DESC LIMIT 1) AS balance_last
    FROM loan l
)
SELECT l.loan_id, l.account_id, l.date AS loan_date, l.status,
       l.amount, l.duration, l.payments,
       a.frequency,
       julianday(l.date) - julianday(a.date)                                AS account_age_days,
       o.gender,
       (julianday(l.date) - julianday(o.birth_date)) / 365.25               AS owner_age,
       (SELECT count(*) FROM disp d WHERE d.account_id = l.account_id)      AS n_account_users,
       (SELECT count(*) FROM card k JOIN disp d USING (disp_id)
         WHERE d.account_id = l.account_id AND k.issued < l.date)           AS n_cards_before_loan,
       di.A11 AS district_avg_salary, di.A13 AS district_unemployment_96,
       di.A14 AS district_entrepreneurs, di.A16 * 1000.0 / di.A4 AS district_crime_rate_96,
       h.*, lb.balance_last
FROM loan l
JOIN account a USING (account_id)
JOIN owner o USING (account_id)
JOIN district di ON di.district_id = a.district_id
LEFT JOIN history h USING (loan_id)
LEFT JOIN last_balance lb USING (loan_id)
ORDER BY l.loan_id
"""

SAFE_WINDOW = f"t.date < l.date AND t.date >= date(l.date, '-{WINDOW_DAYS} days')"
LEAKY_WINDOW = "1 = 1"  # all transactions, including after the loan

NUMERIC = ["amount", "duration", "payments", "account_age_days", "owner_age", "n_account_users",
           "n_cards_before_loan", "district_avg_salary", "district_unemployment_96", "district_entrepreneurs",
           "district_crime_rate_96", "n_trans", "balance_mean", "balance_min", "balance_max", "balance_last",
           "n_negative_balance", "n_sanction_interest", "monthly_credits", "monthly_debits",
           "payment_to_income", "cash_withdrawal_share", "has_pension"]
CATEGORICAL = ["frequency", "gender"]


def build_features(db_path: str | Path, leaky: bool = False) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = FEATURES_SQL.format(window=LEAKY_WINDOW if leaky else SAFE_WINDOW)
    df = pd.read_sql_query(sql, con)
    con.close()
    df = df.loc[:, ~df.columns.duplicated()]

    months = (df["history_days"].clip(lower=30) / 30.44).fillna(1)
    df["monthly_credits"] = df["credits_total"].fillna(0) / months
    df["monthly_debits"] = df["debits_total"].fillna(0) / months
    df["payment_to_income"] = df["payments"] / df["monthly_credits"].replace(0, np.nan)
    df["cash_withdrawal_share"] = df["n_cash_withdrawals"].fillna(0) / df["n_trans"].replace(0, np.nan)
    df["has_pension"] = (df["n_pension"].fillna(0) > 0).astype(int)
    df["default"] = df["status"].isin(["B", "D"]).astype(int)
    df["loan_year"] = df["loan_date"].str[:4].astype(int)
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC + CATEGORICAL]
