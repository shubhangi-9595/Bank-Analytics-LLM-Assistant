"""Guardrail #1 for SQL: inspect the model's query BEFORE it runs.

Why parse instead of regex? `"DELETE" in sql` is fooled by comments, casing, string literals
and stacked statements. Parsing into a syntax tree (sqlglot) shows what the query really does.

Rules:
- exactly one statement, and it must be a read query (SELECT / UNION / WITH ... SELECT)
- no write, DDL or admin nodes anywhere in the tree (catches INSERT ... SELECT, VACUUM INTO, ATTACH)
- only the 8 bank tables, plus the query's own CTE names
- no table-valued functions (pragma_table_info(...)) and no dangerous functions (load_extension, readfile)

The validator only CHECKS. The original text is what runs, so a pass can't change the
query's meaning. Defence in depth: the executor also opens SQLite read-only with an
authorizer that rejects anything but reads, so a validator bug alone can't cause damage.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"account", "card", "client", "disp", "district", "loan", "order", "trans"}
FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
                   exp.Merge, exp.TruncateTable, exp.Copy, exp.Set, exp.Pragma, exp.Use, exp.Attach,
                   exp.Detach, exp.Transaction, exp.Commit, exp.Rollback)
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zipfile",
                       "sqlite_compileoption_get", "sqlite_compileoption_used", "sqlite_source_id",
                       "sqlite_offset", "randomblob", "zeroblob"}


@dataclass
class Validation:
    ok: bool
    reason: str = ""


def validate_sql(sql: str, allowed: set[str] = ALLOWED_TABLES) -> Validation:
    text = sql.strip().rstrip(";").strip()
    if not text:
        return Validation(False, "empty query")
    try:
        statements = [s for s in sqlglot.parse(text, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        return Validation(False, f"could not parse SQL: {str(e).splitlines()[0]}")
    if len(statements) != 1:
        return Validation(False, "only one statement is allowed")

    tree = statements[0]
    if not isinstance(tree, exp.Query):
        return Validation(False, f"only SELECT queries are allowed (got {type(tree).__name__})")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return Validation(False, f"forbidden operation: {type(node).__name__}")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS or name.startswith("pragma_"):
                return Validation(False, f"function not allowed: {name}")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return Validation(False, "table-valued functions are not allowed")
        if table.db or table.catalog:
            return Validation(False, "schema-qualified tables are not allowed")
        name = table.name.lower()
        if name not in allowed and name not in ctes:
            return Validation(False, f"table '{table.name}' is not allowed")
    return Validation(True)
