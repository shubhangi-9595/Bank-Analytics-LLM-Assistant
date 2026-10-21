"""Build the bank database (SQLite) from the 8 raw Berka CSV files, with quality checks.

    python -m bank_assistant.data.build_db

The result follows the schema the BIRD benchmark uses for this bank, so BIRD's reference SQL
runs on it unchanged. What we have to fix in the raw files:

- semicolon-separated, with quoted strings; an unquoted empty field means missing (NULL)
- dates are YYMMDD integers (930101) -> ISO text '1993-01-01'; card.issued also has a time part
- client.birth_number packs gender into the month: women have 50 added (YY MM+50 DD)
  -> split into gender ('M'/'F') and birth_date
- district columns are anonymised A1..A16; A1 is the id -> renamed district_id
- one district has '?' instead of numbers in A12 and A15 -> NULL
- the transaction "type" column has a third, undocumented-in-most-places value 'VYBER' (cash)

Quality checks fail the build loudly instead of producing a subtly wrong database.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from bank_assistant.config import get_settings

SCHEMA = """
CREATE TABLE district (
    district_id INTEGER PRIMARY KEY,
    A2 TEXT, A3 TEXT, A4 INTEGER, A5 INTEGER, A6 INTEGER, A7 INTEGER, A8 INTEGER, A9 INTEGER,
    A10 REAL, A11 INTEGER, A12 REAL, A13 REAL, A14 INTEGER, A15 INTEGER, A16 INTEGER
);
CREATE TABLE account (
    account_id INTEGER PRIMARY KEY, district_id INTEGER REFERENCES district, frequency TEXT, date TEXT
);
CREATE TABLE client (
    client_id INTEGER PRIMARY KEY, gender TEXT, birth_date TEXT, district_id INTEGER REFERENCES district
);
CREATE TABLE disp (
    disp_id INTEGER PRIMARY KEY, client_id INTEGER REFERENCES client, account_id INTEGER REFERENCES account,
    type TEXT
);
CREATE TABLE card (
    card_id INTEGER PRIMARY KEY, disp_id INTEGER REFERENCES disp, type TEXT, issued TEXT
);
CREATE TABLE loan (
    loan_id INTEGER PRIMARY KEY, account_id INTEGER REFERENCES account, date TEXT, amount INTEGER,
    duration INTEGER, payments REAL, status TEXT
);
CREATE TABLE "order" (
    order_id INTEGER PRIMARY KEY, account_id INTEGER REFERENCES account, bank_to TEXT, account_to INTEGER,
    amount REAL, k_symbol TEXT
);
CREATE TABLE trans (
    trans_id INTEGER PRIMARY KEY, account_id INTEGER REFERENCES account, date TEXT, type TEXT,
    operation TEXT, amount INTEGER, balance INTEGER, k_symbol TEXT, bank TEXT, account INTEGER
);
CREATE INDEX trans_account_date ON trans(account_id, date);
CREATE INDEX disp_account ON disp(account_id);
CREATE INDEX disp_client ON disp(client_id);
CREATE INDEX loan_account ON loan(account_id);
"""

# Documented row counts for the PKDD'99 release. A mismatch means a broken download.
EXPECTED_ROWS = {"account": 4500, "client": 5369, "disp": 5369, "order": 6471, "trans": 1056320,
                 "loan": 682, "card": 892, "district": 77}


# ------------------------------------------------------------------------------------ parsing
def parse_line(line: str) -> list:
    """'"a";1;;"" ' -> ['a', '1', None, '']. Quoted = text (maybe empty), unquoted empty = NULL."""
    out = []
    for field in line.rstrip("\r\n").split(";"):
        if len(field) >= 2 and field[0] == '"' and field[-1] == '"':
            out.append(field[1:-1])
        elif field in ("", "?"):
            out.append(None)
        else:
            out.append(field)
    return out


def read_csv(path: Path) -> tuple[list[str], list[list]]:
    with open(path, encoding="utf-8") as f:
        header = [h.strip('"') for h in f.readline().rstrip("\r\n").split(";")]
        rows = []
        for n, line in enumerate(f, start=2):
            row = parse_line(line)
            if len(row) != len(header):
                raise ValueError(f"{path.name} line {n}: {len(row)} fields, expected {len(header)}")
            rows.append(row)
    return header, rows


def yymmdd(value) -> str | None:
    if value is None:
        return None
    s = str(value).split()[0]  # card.issued is '931107 00:00:00'
    return f"19{s[0:2]}-{s[2:4]}-{s[4:6]}"


def decode_birth_number(bn: str) -> tuple[str, str]:
    yy, mm, dd = int(bn[0:2]), int(bn[2:4]), int(bn[4:6])
    gender = "F" if mm > 50 else "M"
    mm = mm - 50 if mm > 50 else mm
    return gender, f"19{yy:02d}-{mm:02d}-{dd:02d}"


def _num(v, cast):
    return None if v is None else cast(float(v)) if cast is int else cast(v)


# ------------------------------------------------------------------------------------ build
def build(raw_dir: Path, db_path: Path) -> Path:
    raw_dir = Path(raw_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)

    def load(name):
        return read_csv(raw_dir / f"{name}.csv")[1]

    con.executemany("INSERT INTO district VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (int(r[0]), r[1], r[2], *[_num(v, int) for v in r[3:9]], _num(r[9], float), _num(r[10], int),
         _num(r[11], float), _num(r[12], float), _num(r[13], int), _num(r[14], int), _num(r[15], int))
        for r in load("district")])
    con.executemany("INSERT INTO account VALUES (?,?,?,?)",
                    [(int(r[0]), int(r[1]), r[2], yymmdd(r[3])) for r in load("account")])
    con.executemany("INSERT INTO client VALUES (?,?,?,?)",
                    [(int(r[0]), *decode_birth_number(r[1]), int(r[2])) for r in load("client")])
    con.executemany("INSERT INTO disp VALUES (?,?,?,?)",
                    [(int(r[0]), int(r[1]), int(r[2]), r[3]) for r in load("disp")])
    con.executemany("INSERT INTO card VALUES (?,?,?,?)",
                    [(int(r[0]), int(r[1]), r[2], yymmdd(r[3])) for r in load("card")])
    con.executemany("INSERT INTO loan VALUES (?,?,?,?,?,?,?)",
                    [(int(r[0]), int(r[1]), yymmdd(r[2]), int(r[3]), int(r[4]), float(r[5]), r[6])
                     for r in load("loan")])
    con.executemany('INSERT INTO "order" VALUES (?,?,?,?,?,?)',
                    [(int(r[0]), int(r[1]), r[2], _num(r[3], int), float(r[4]), r[5]) for r in load("order")])
    con.executemany("INSERT INTO trans VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(int(r[0]), int(r[1]), yymmdd(r[2]), r[3], r[4], _num(r[5], int), _num(r[6], int),
                      r[7], r[8], _num(r[9], int)) for r in load("trans")])
    con.commit()

    problems = quality_checks(con, strict_counts=(raw_dir / "trans.csv").stat().st_size > 50_000_000)
    con.close()
    if problems:
        tmp.unlink()
        raise SystemExit("quality checks failed:\n  - " + "\n  - ".join(problems))
    tmp.replace(db_path)
    return db_path


def quality_checks(con: sqlite3.Connection, strict_counts: bool = True) -> list[str]:
    problems = []
    q = lambda sql: con.execute(sql).fetchone()[0]  # noqa: E731
    if strict_counts:  # the synthetic test sample is smaller on purpose
        for table, n in EXPECTED_ROWS.items():
            got = q(f'SELECT count(*) FROM "{table}"')
            if got != n:
                problems.append(f"{table}: {got} rows, expected {n}")
    orphans = {
        "trans without account": "SELECT count(*) FROM trans t LEFT JOIN account a USING(account_id) WHERE a.account_id IS NULL",
        "loan without account": "SELECT count(*) FROM loan l LEFT JOIN account a USING(account_id) WHERE a.account_id IS NULL",
        "disp without client": "SELECT count(*) FROM disp d LEFT JOIN client c USING(client_id) WHERE c.client_id IS NULL",
        "card without disp": "SELECT count(*) FROM card c LEFT JOIN disp d USING(disp_id) WHERE d.disp_id IS NULL",
        "account without district": "SELECT count(*) FROM account a LEFT JOIN district d USING(district_id) WHERE d.district_id IS NULL",
    }
    for name, sql in orphans.items():
        if (n := q(sql)):
            problems.append(f"{n} {name}")
    if (n := q("SELECT count(*) FROM client WHERE birth_date NOT GLOB '19[0-9][0-9]-[01][0-9]-[0-3][0-9]' "
               "OR substr(birth_date, 6, 2) NOT BETWEEN '01' AND '12'")):
        problems.append(f"{n} clients with an impossible birth date (birth_number decoding bug?)")
    if (n := q("SELECT count(*) FROM loan WHERE status NOT IN ('A','B','C','D')")):
        problems.append(f"{n} loans with an unknown status")
    if (n := q("SELECT count(*) FROM loan l JOIN account a USING(account_id) WHERE l.date < a.date")):
        problems.append(f"{n} loans granted before their account was opened")
    if (n := q("SELECT count(*) FROM (SELECT account_id FROM loan GROUP BY 1 HAVING count(*) > 1)")):
        problems.append(f"{n} accounts with more than one loan (documented: at most one)")
    return problems


def main() -> None:
    s = get_settings()
    path = build(s.raw_dir / "berka", s.db_path)
    con = sqlite3.connect(path)
    counts = {t: con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in EXPECTED_ROWS}
    print(f"built {path}\n  " + ", ".join(f"{t} {n:,}" for t, n in counts.items()))
    if s.bird_questions.exists():
        from bank_assistant.data.bird import check_gold_sql
        ok, failed = check_gold_sql(path, s.bird_questions)
        print(f"BIRD reference SQL: {ok} of {ok + len(failed)} run on this database")
        for qid, err in failed:
            print(f"  question {qid}: {err}", file=sys.stderr)
        bird_db = s.raw_dir / "bird_financial.sqlite"
        if bird_db.exists():
            from bank_assistant.data.bird import compare_with_bird_db
            same, different = compare_with_bird_db(path, bird_db, s.bird_questions)
            print(f"same answers as BIRD's own database: {same} of {same + len(different)}"
                  + (f" (different: {different})" if different else ""))


if __name__ == "__main__":
    main()
