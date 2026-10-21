"""Guardrail #2 for SQL: run queries on a connection that physically can't do harm.

- `mode=ro`      : SQLite opens the file read-only; any write fails
- an authorizer  : SQLite asks us about EVERY operation while compiling the query. We allow
                   only reading the 8 bank tables and calling ordinary functions. ATTACH,
                   PRAGMA, writes, reading sqlite_master, load_extension... are all denied.
- a progress handler interrupts queries running longer than the timeout
- results are fetched with a cap, so a huge result can't flood memory or the LLM's context
- Python's sqlite3 refuses to execute two statements in one call (a third, free layer)
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from bank_assistant.sql.validator import ALLOWED_TABLES, FORBIDDEN_FUNCTIONS, validate_sql


@dataclass
class QueryResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def as_markdown(self, max_rows: int = 20) -> str:
        if self.error:
            return f"(error: {self.error})"
        if not self.rows:
            return "(no rows)"
        head = "| " + " | ".join(self.columns) + " |\n|" + "---|" * len(self.columns) + "\n"
        body = "\n".join("| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows[:max_rows])
        more = len(self.rows) - max_rows
        tail = f"\n... {more} more rows" if more > 0 else ""
        return head + body + tail + ("\n(result truncated)" if self.truncated else "")


def _fmt(v) -> str:
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if db_name is None:  # reading a CTE or subquery the query itself defined, not a stored table
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_OK if db_name == "main" and (arg1 or "").lower() in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or "").lower() in FORBIDDEN_FUNCTIONS else sqlite3.SQLITE_OK
    if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):  # WITH RECURSIVE
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY  # everything else: writes, ATTACH, PRAGMA, transactions, ...


class SQLExecutor:
    def __init__(self, db_path: str | Path, row_limit: int | None = 200, timeout_s: float = 10.0):
        self.row_limit, self.timeout_s = row_limit, timeout_s
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.con.enable_load_extension(False) if hasattr(self.con, "enable_load_extension") else None
        self.con.set_authorizer(_authorizer)
        self._lock = threading.Lock()

    def run(self, sql: str, row_limit: int | None = -1) -> QueryResult:
        limit = self.row_limit if row_limit == -1 else row_limit
        check = validate_sql(sql)
        if not check.ok:
            return QueryResult(sql, error=f"rejected by validator: {check.reason}")
        with self._lock:
            deadline = time.monotonic() + self.timeout_s
            self.con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            try:
                cur = self.con.execute(sql)
                columns = [d[0] for d in (cur.description or [])]
                if limit is None:
                    rows, truncated = cur.fetchall(), False
                else:
                    rows = cur.fetchmany(limit + 1)
                    truncated = len(rows) > limit
                    rows = rows[:limit]
            except sqlite3.OperationalError as e:
                msg = str(e)
                if "interrupted" in msg:
                    msg = f"query took longer than {self.timeout_s:g}s"
                return QueryResult(sql, error=msg)
            except (sqlite3.DatabaseError, sqlite3.Warning, sqlite3.ProgrammingError) as e:
                return QueryResult(sql, error=str(e).splitlines()[0])
            finally:
                self.con.set_progress_handler(None, 0)
        return QueryResult(sql, columns, rows, truncated)

    def close(self) -> None:
        self.con.close()


def schema_ddl(db_path: str | Path) -> str:
    """The schema as CREATE TABLE statements, read by trusted code (not via the LLM's connection)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute("SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    con.close()
    return "\n".join(r[0].strip() + ";" for r in rows if r[0])
