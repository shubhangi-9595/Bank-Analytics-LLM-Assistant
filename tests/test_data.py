import sqlite3

import pytest

from bank_assistant.data.build_db import build, decode_birth_number, parse_line, yymmdd
from bank_assistant.data.sample import generate


def test_parse_line_distinguishes_empty_text_from_missing():
    assert parse_line('695247;2378;930101;"PRIJEM";"";;\n') == ["695247", "2378", "930101", "PRIJEM", "", None, None]
    assert parse_line('1;"x";?') == ["1", "x", None]  # the real district file has '?' for missing


def test_dates_and_birth_numbers():
    assert yymmdd("930101") == "1993-01-01"
    assert yymmdd("931107 00:00:00") == "1993-11-07"
    assert decode_birth_number("706213") == ("F", "1970-12-13")  # month 62 = December + 50 (female)
    assert decode_birth_number("450204") == ("M", "1945-02-04")


def test_build_runs_quality_checks(tmp_path):
    db = build(generate(tmp_path / "raw", n_accounts=120), tmp_path / "bank.sqlite")
    con = sqlite3.connect(db)
    assert con.execute("SELECT count(*) FROM client WHERE gender NOT IN ('M','F')").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM loan").fetchone()[0] > 10
    assert con.execute("SELECT A12, A15 FROM district WHERE district_id = 9").fetchone() == (None, None)


def test_build_fails_loudly_on_broken_data(tmp_path):
    raw = generate(tmp_path / "raw", n_accounts=30)
    lines = (raw / "loan.csv").read_text().splitlines()
    lines[1] = lines[1].rsplit(";", 1)[0] + ';"Z"'  # an unknown loan status
    (raw / "loan.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="unknown status"):
        build(raw, tmp_path / "bank.sqlite")
    assert not (tmp_path / "bank.sqlite").exists()  # no half-built database left behind
