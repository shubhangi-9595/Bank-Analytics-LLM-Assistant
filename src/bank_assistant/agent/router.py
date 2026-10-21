"""Routing: decide which tool(s) a question needs.

Tools:
  query_database     numbers and lists from the bank's tables (text-to-SQL)
  search_dictionary  what a column or code means ("what is A11?", "what does status D mean?")
  score_loan         default risk of one loan or account, with the reasons

- LLMRouter : the model picks tools via function calling; it can pick several.
- RuleRouter: keyword rules. Free and predictable; it's the offline mode and the fallback
              when the LLM call fails (rate limit, spend limit, network).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LOAN_ID = re.compile(r"\bloan\s*(?:id|number|no\.?|#)?\s*:?\s*(\d+)", re.I)
_ACCOUNT_ID = re.compile(r"\baccount\s*(?:id|number|no\.?|#)?\s*:?\s*(\d+)", re.I)
_RISK = re.compile(r"\b(risk|risky|default|likely|probability|chance|score|explain|why)\b", re.I)
_DOCS = re.compile(r"\b(what (does|do|is|are)\b.*\b(mean|stand for|column|code)|meaning of|stand for|"
                   r"what is (column )?A\d{1,2}\b|definition|define|data dictionary|which column)", re.I)
_RAW_SQL = re.compile(r"^\s*(select|with)\b", re.I)


@dataclass
class Route:
    tools: list[str]
    reason: str
    sql_question: str = ""
    docs_query: str = ""
    loan_id: int | None = None
    account_id: int | None = None
    raw_sql: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "+".join(self.tools)


class RuleRouter:
    name = "rules"

    def route(self, question: str) -> Route:
        if _RAW_SQL.match(question):
            return Route(["query_database"], "the input is SQL", raw_sql=question.strip())
        loan, account = _LOAN_ID.search(question), _ACCOUNT_ID.search(question)
        if (loan or account) and _RISK.search(question):
            return Route(["score_loan"], "asks about the risk of a specific loan or account",
                         loan_id=int(loan.group(1)) if loan else None,
                         account_id=int(account.group(1)) if account and not loan else None)
        if _DOCS.search(question):
            return Route(["search_dictionary"], "asks what a column or code means", docs_query=question)
        return Route(["query_database"], "a question about the data", sql_question=question)


TOOLS = [
    {"type": "function", "function": {
        "name": "query_database",
        "description": "Answer questions about the bank's data (accounts, clients, loans, cards, transactions, "
                       "districts) by writing and running SQL: counts, totals, averages, rankings, lists, lookups.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The data question, self-contained"}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "search_dictionary",
        "description": "Look up what a column, code or table means in the data dictionary (e.g. A11, "
                       "'POPLATEK TYDNE', loan status 'D'), or how tables relate. Not for numbers.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look up"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "score_loan",
        "description": "Estimate the default risk of ONE loan, given its loan id or its account id, and explain "
                       "the main reasons. Use when the user asks how risky a specific loan or account is.",
        "parameters": {"type": "object", "properties": {
            "loan_id": {"type": "integer"}, "account_id": {"type": "integer"}}}}},
]

ROUTER_SYSTEM = ("You route questions for a bank data assistant. Call the tool or tools needed to answer. "
                 "Do not answer the question yourself.")


class LLMRouter:
    name = "llm"

    def __init__(self, llm, fallback: RuleRouter | None = None):
        self.llm, self.fallback = llm, fallback or RuleRouter()

    def route(self, question: str) -> Route:
        if _RAW_SQL.match(question):
            return self.fallback.route(question)
        try:
            resp = self.llm.chat([{"role": "system", "content": ROUTER_SYSTEM},
                                  {"role": "user", "content": question}], tools=TOOLS, tool_choice="required",
                                 max_tokens=200)
        except Exception as e:  # budget, network, quota: degrade gracefully
            r = self.fallback.route(question)
            r.reason += f" (LLM router unavailable: {type(e).__name__})"
            return r
        calls = {c.name: c.arguments for c in resp.tool_calls}
        if not calls:
            return self.fallback.route(question)
        route = Route([n for n in ("score_loan", "search_dictionary", "query_database") if n in calls],
                      "model chose: " + ", ".join(calls))
        route.sql_question = calls.get("query_database", {}).get("question", "")
        route.docs_query = calls.get("search_dictionary", {}).get("query", "")
        args = calls.get("score_loan", {})
        route.loan_id, route.account_id = args.get("loan_id"), args.get("account_id")
        return route
