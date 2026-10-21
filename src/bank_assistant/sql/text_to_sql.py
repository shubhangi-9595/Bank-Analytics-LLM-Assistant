"""Text-to-SQL: question in English -> SQLite query over the bank database.

The prompt always contains the schema (CREATE TABLE statements) and three worked examples
written for this project (not taken from the benchmark). What varies is the NOTES section,
and that's the experiment the evals measure:

    none       schema only: can the model guess what A11 or 'POPLATEK TYDNE' mean?
    retrieved  the top-k data dictionary entries for this question (RAG)
    full       the whole data dictionary pasted in (accurate but many more tokens)
    oracle     BIRD's human-written hint for this question (upper bound; eval only)

After generating, the query is run through the guarded executor. If it fails (syntax error,
unknown column, rejected by the validator), the error goes back to the model for one retry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from bank_assistant.sql.executor import QueryResult, SQLExecutor

FEW_SHOT = [
    ("What was the average monthly payment of loans in north Moravia that finished without being paid back?",
     "- loan.status 'B' = contract finished, loan not paid\n- district A3 = region",
     "SELECT AVG(l.payments) FROM loan AS l JOIN account AS a ON l.account_id = a.account_id "
     "JOIN district AS d ON a.district_id = d.district_id WHERE d.A3 = 'north Moravia' AND l.status = 'B'"),
    ("How many old-age pension payments were credited in 1998?",
     "- trans.k_symbol 'DUCHOD' = old-age pension",
     "SELECT COUNT(*) FROM trans WHERE k_symbol = 'DUCHOD' AND STRFTIME('%Y', date) = '1998'"),
    ("Which three clients holding a classic card are the oldest? Give their ids and birth dates.",
     "- card links to client through disp (card.disp_id = disp.disp_id)",
     "SELECT c.client_id, c.birth_date FROM client AS c JOIN disp AS d ON c.client_id = d.client_id "
     "JOIN card AS k ON k.disp_id = d.disp_id WHERE k.type = 'classic' ORDER BY c.birth_date ASC LIMIT 3"),
]

SYSTEM = """You write SQLite queries for analysts at a Czech bank (data from 1993-1998).

Schema:
{schema}

Rules:
- Write ONE read-only SELECT statement in SQLite syntax (STRFTIME, IIF, CAST(x AS REAL) for division).
- Use only the tables and columns above. The table `order` must be quoted with backticks.
- Many values are Czech codes and district columns are named A2..A16: rely on the notes to map words to codes.
- Return only the columns the question asks for.
Reply with JSON only: {{"sql": "..."}}. If the question can't be answered from this data, reply {{"sql": null}}."""


@dataclass
class SQLAttempt:
    question: str
    sql: str | None
    result: QueryResult | None
    notes: str = ""
    repaired: bool = False
    history: list[str] = field(default_factory=list)


class LLMTextToSQL:
    def __init__(self, llm, executor: SQLExecutor, schema: str, retriever=None, knowledge: list[str] | None = None,
                 top_k: int = 4, max_repairs: int = 1):
        self.llm, self.executor, self.retriever = llm, executor, retriever
        self.knowledge, self.top_k, self.max_repairs = knowledge or [], top_k, max_repairs
        self.system = SYSTEM.format(schema=schema)
        self.examples = []
        for q, notes, sql in FEW_SHOT:
            self.examples += [{"role": "user", "content": self._user(q, notes)},
                              {"role": "assistant", "content": json.dumps({"sql": sql})}]

    # ---------------------------------------------------------------- notes
    def notes_for(self, question: str, mode: str, oracle: str = "") -> str:
        if mode == "none":
            return ""
        if mode == "oracle":
            return oracle
        if mode == "full":
            return "\n\n".join(self.knowledge)
        if mode == "retrieved":
            if self.retriever is None:
                raise ValueError("retrieved mode needs a retriever")
            return "\n\n".join(c.text for c, _ in self.retriever.retrieve(question, self.top_k))
        raise ValueError(f"unknown knowledge mode: {mode}")

    @staticmethod
    def _user(question: str, notes: str) -> str:
        return f"Question: {question}" + (f"\n\nNotes:\n{notes}" if notes else "")

    # ---------------------------------------------------------------- generate
    def generate_and_run(self, question: str, mode: str = "retrieved", oracle: str = "",
                         row_limit: int | None = -1) -> SQLAttempt:
        notes = self.notes_for(question, mode, oracle)
        history = [{"role": "user", "content": self._user(question, notes)}]
        sql = self._ask(history)
        attempt = SQLAttempt(question, sql, None, notes)
        if not sql:
            return attempt
        result = self.executor.run(sql, row_limit=row_limit)
        for _ in range(self.max_repairs):  # self-correction loop
            if result.ok:
                break
            attempt.history.append(f"{sql}  ->  {result.error}")
            history += [{"role": "assistant", "content": json.dumps({"sql": sql})},
                        {"role": "user", "content": f"That query failed: {result.error}. Return a corrected query."}]
            fixed = self._ask(history)
            if not fixed:
                break
            sql, attempt.repaired = fixed, True
            result = self.executor.run(sql, row_limit=row_limit)
        attempt.sql, attempt.result = sql, result
        return attempt

    def _ask(self, history: list[dict]) -> str | None:
        resp = self.llm.chat([{"role": "system", "content": self.system}, *self.examples, *history],
                             json_mode=True, max_tokens=500)
        try:
            sql = json.loads(resp.text).get("sql")
        except (json.JSONDecodeError, AttributeError):
            return None
        return sql.strip() if isinstance(sql, str) and sql.strip() else None
