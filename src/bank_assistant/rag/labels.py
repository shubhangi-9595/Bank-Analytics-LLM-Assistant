"""Automatic relevance labels for retrieval, derived from BIRD's reference SQL.

Hand-labelling which dictionary entries each question needs is slow and biased (you label
what your system finds). Instead we read each reference query and list the opaque things it
relies on:
- a coded value compared with a coded column      ->  "'POPLATEK PO OBRATU' ="
- an anonymised district column (A2..A16)        ->  "A11 ="
A retrieval result "covers" a term if the retrieved text contains that definition.
This works for any chunking strategy, so strategies can be compared fairly.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

CODED_COLUMNS = {("account", "frequency"), ("trans", "type"), ("trans", "operation"), ("trans", "k_symbol"),
                 ("order", "k_symbol"), ("loan", "status"), ("disp", "type"), ("card", "type"),
                 ("client", "gender")}
_A_COL = re.compile(r"^A([2-9]|1[0-6])$", re.IGNORECASE)  # qualify() lowercases names


def needed_terms(sql: str, schema: dict[str, dict[str, str]]) -> set[str]:
    try:
        tree = qualify(sqlglot.parse_one(sql, read="sqlite"), schema=schema, dialect="sqlite",
                       validate_qualify_columns=False, identify=False)
    except Exception:
        tree = sqlglot.parse_one(sql, read="sqlite")
    alias_to_table = {t.alias_or_name: t.name for t in tree.find_all(exp.Table)}

    def table_of(col: exp.Column) -> str:
        return alias_to_table.get(col.table, col.table)

    terms = set()
    for col in tree.find_all(exp.Column):
        if _A_COL.match(col.name) and table_of(col) in ("district", ""):
            terms.add(f"{col.name.upper()} =")
    for node in tree.find_all(exp.EQ, exp.NEQ, exp.In, exp.Like):
        cols = [c for c in node.find_all(exp.Column) if (table_of(c), c.name) in CODED_COLUMNS]
        lits = [lit.this for lit in node.find_all(exp.Literal) if lit.is_string]
        if cols and lits:
            terms |= {f"'{v}' =" for v in lits}
    return terms


def schema_from_db(con) -> dict[str, dict[str, str]]:
    schema = {}
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        schema[table] = {r[1]: (r[2] or "TEXT") for r in con.execute(f'PRAGMA table_info("{table}")')}
    return schema
