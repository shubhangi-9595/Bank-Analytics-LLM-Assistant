import pytest

from bank_assistant.sql.executor import SQLExecutor
from bank_assistant.sql.validator import validate_sql

ATTACKS = ["DROP TABLE loan", "DELETE FROM trans", "SELECT 1; DELETE FROM loan", "INSERT INTO loan SELECT * FROM loan",
           "ATTACH 'x.db' AS x", "PRAGMA table_info(loan)", "SELECT * FROM pragma_table_info('loan')",
           "SELECT name FROM sqlite_master", "SELECT load_extension('evil')", "SELECT readfile('/etc/passwd')",
           "VACUUM INTO '/tmp/copy.db'", "WITH x AS (SELECT * FROM sqlite_master) SELECT * FROM x",
           "SELECT * FROM main.loan"]


@pytest.mark.parametrize("sql", [
    "SELECT status, COUNT(*) FROM loan GROUP BY status",
    "SELECT * FROM `order` LIMIT 1, 1",
    "WITH d AS (SELECT * FROM loan WHERE status = 'D') SELECT COUNT(*) FROM d",
    "SELECT DISTINCT IIF(AVG(A13) > AVG(A12), '1996', '1995') FROM district",
])
def test_allows_reads(sql):
    assert validate_sql(sql).ok


@pytest.mark.parametrize("sql", ATTACKS)
def test_validator_blocks_attacks(sql):
    assert not validate_sql(sql).ok


@pytest.mark.parametrize("sql", [a for a in ATTACKS if a != "SELECT * FROM main.loan"])
def test_database_blocks_attacks_even_without_validator(bank, sql):
    ex = SQLExecutor(bank.db_path)
    with pytest.raises(Exception):
        ex.con.execute(sql).fetchall()  # skips validate_sql on purpose
    ex.close()


def test_row_cap_and_timeout(bank):
    ex = SQLExecutor(bank.db_path, row_limit=10, timeout_s=0.5)
    r = ex.run("SELECT * FROM trans")
    assert len(r.rows) == 10 and r.truncated
    assert "longer than" in ex.run("SELECT count(*) FROM trans a, trans b, trans c").error
    assert ex.run("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 5) "
                  "SELECT count(*) FROM n").rows == [(5,)]


def test_validator_does_not_rewrite_the_query(bank):
    ex = SQLExecutor(bank.db_path)
    sql = "SELECT COUNT(*) FROM loan WHERE status = 'A'"
    assert ex.run(sql).sql == sql
