"""Generate a small synthetic bank in the exact RAW Berka CSV format.

Tests and CI use it so they run in seconds without the 70 MB download, and so the CSV
parsing/decoding code (YYMMDD dates, birth numbers, quoting) is exercised end to end.
Defaulted loans get weaker account histories (lower balances, sanction interest), so the
model has a signal to find, like in the real data. The numbers mean nothing.
"""
from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

REGIONS = ["Prague", "central Bohemia", "south Bohemia", "west Bohemia", "north Bohemia", "east Bohemia",
           "south Moravia", "north Moravia"]
NAMES = ["Hl.m. Praha", "Benesov", "Pisek", "Klatovy", "Liberec", "Pardubice", "Brno - mesto", "Ostrava - mesto",
         "Sokolov", "Jicin"]
FREQ = ["POPLATEK MESICNE"] * 12 + ["POPLATEK TYDNE", "POPLATEK PO OBRATU"]


def _d(day: dt.date) -> str:
    return day.strftime("%y%m%d")


def _q(v) -> str:
    return f'"{v}"'


def generate(out_dir: str | Path, n_accounts: int = 300, seed: int = 7) -> Path:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {n: open(out / f"{n}.csv", "w", encoding="utf-8", newline="\n")
             for n in ["district", "account", "client", "disp", "card", "loan", "order", "trans"]}
    w = lambda name, *fields: files[name].write(";".join(map(str, fields)) + "\n")  # noqa: E731

    w("district", *[f"A{i}" for i in range(1, 17)])
    for i, name in enumerate(NAMES, start=1):
        a12 = "?" if i == 9 else f"{rng.uniform(0.3, 6):.2f}"  # the real file has one '?' too
        w("district", i, _q(name), _q(REGIONS[(i - 1) % len(REGIONS)]), rng.randint(40_000, 1_200_000),
          *[rng.randint(0, 80) for _ in range(4)], rng.randint(1, 15), f"{rng.uniform(30, 100):.1f}",
          rng.randint(8000, 12600), a12, f"{rng.uniform(0.4, 7):.2f}", rng.randint(80, 170),
          "?" if i == 9 else rng.randint(800, 85_000), rng.randint(900, 99_000))

    w("account", *map(_q, ["account_id", "district_id", "frequency", "date"]))
    w("client", *map(_q, ["client_id", "birth_number", "district_id"]))
    w("disp", *map(_q, ["disp_id", "client_id", "account_id", "type"]))
    w("card", *map(_q, ["card_id", "disp_id", "type", "issued"]))
    w("loan", *map(_q, ["loan_id", "account_id", "date", "amount", "duration", "payments", "status"]))
    w("order", *map(_q, ["order_id", "account_id", "bank_to", "account_to", "amount", "k_symbol"]))
    w("trans", *map(_q, ["trans_id", "account_id", "date", "type", "operation", "amount", "balance", "k_symbol",
                         "bank", "account"]))

    client_id = disp_id = card_id = loan_id = order_id = trans_id = 1
    for acc in range(1, n_accounts + 1):
        district = rng.randint(1, len(NAMES))
        opened = dt.date(1993, 1, 1) + dt.timedelta(days=rng.randint(0, 900))
        w("account", acc, district, _q(rng.choice(FREQ)), _d(opened))

        has_loan = rng.random() < 0.35
        defaults = has_loan and rng.random() < 0.25
        for owner in [True] + ([False] if rng.random() < 0.2 else []):
            female = rng.random() < 0.5
            born = dt.date(1920, 1, 1) + dt.timedelta(days=rng.randint(0, 365 * 60))
            month = born.month + (50 if female else 0)
            w("client", client_id, _q(f"{born:%y}{month:02d}{born:%d}"), district)
            w("disp", disp_id, client_id, acc, _q("OWNER" if owner else "DISPONENT"))
            if owner and rng.random() < 0.2:
                issued = opened + dt.timedelta(days=rng.randint(30, 600))
                w("card", card_id, disp_id, _q(rng.choice(["classic"] * 4 + ["junior", "gold"])),
                  f"{_d(issued)} 00:00:00")
                card_id += 1
            client_id += 1
            disp_id += 1

        income = rng.uniform(9000, 30000) * (0.6 if defaults else 1.0)
        balance = rng.uniform(500, 3000)
        day = opened
        loan_day = opened + dt.timedelta(days=rng.randint(200, 900)) if has_loan else None
        end = min(dt.date(1998, 12, 31), opened + dt.timedelta(days=1400))
        while day < end:
            balance += income
            w("trans", trans_id, acc, _d(day), _q("PRIJEM"), _q("VKLAD"), int(income), int(balance), _q(""), "", "")
            trans_id += 1
            for _ in range(rng.randint(1, 4)):
                spend = income * rng.uniform(0.15, 0.45) * (1.25 if defaults else 1.0)
                balance -= spend
                op = rng.choice(["VYBER", "PREVOD NA UCET", "VYBER KARTOU"])
                bank = rng.choice(["AB", "CD", "EF", "GH"]) if op == "PREVOD NA UCET" else ""
                w("trans", trans_id, acc, _d(day + dt.timedelta(days=rng.randint(1, 25))),
                  _q("VYBER" if op == "VYBER" and rng.random() < 0.05 else "VYDAJ"), _q(op), int(spend),
                  int(balance), _q(rng.choice(["SIPO", "", " ", "POJISTNE"])), _q(bank) if bank else "",
                  rng.randint(10**7, 10**8) if bank else "")
                trans_id += 1
            if balance < 0:
                w("trans", trans_id, acc, _d(day + dt.timedelta(days=27)), _q("VYDAJ"), _q(""), 20, int(balance - 20),
                  _q("SANKC. UROK"), "", "")
                trans_id += 1
            day += dt.timedelta(days=30)

        if has_loan and loan_day < dt.date(1998, 12, 1):
            amount = rng.choice([24, 36, 48, 60]) * rng.randint(1500, 6000)
            duration = rng.choice([12, 24, 36, 48, 60])
            finished = loan_day + dt.timedelta(days=30 * duration) < dt.date(1999, 1, 1)
            status = ("B" if finished else "D") if defaults else ("A" if finished else "C")
            w("loan", loan_id, acc, _d(loan_day), amount, duration, f"{amount / duration:.2f}", _q(status))
            loan_id += 1
        if rng.random() < 0.5:
            w("order", order_id, acc, _q(rng.choice(["AB", "YZ", "ST"])), _q(rng.randint(10**7, 10**8)),
              f"{rng.uniform(100, 9000):.2f}", _q(rng.choice(["SIPO", "UVER", "POJISTNE", "LEASING", " "])))
            order_id += 1

    for f in files.values():
        f.close()
    return out


if __name__ == "__main__":
    print(generate("data/demo/raw/berka"))
