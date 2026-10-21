"""Input and output checks around the whole assistant.

INPUT  (before any LLM call): length limit, PII masking, injection heuristics,
       a scope check so the bot doesn't become a free general chatbot on your credit.
OUTPUT (before the user sees it): PII masking, citation check (every [doc] id
       cited must have been retrieved), numeric grounding (numbers in a data
       answer should come from the SQL result).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bank_assistant.guardrails.injection import looks_like_injection
from bank_assistant.guardrails.pii import mask_pii

MAX_QUESTION_CHARS = 500
_RAW_SQL = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_IN_SCOPE = re.compile(
    r"account|client|customer|loan|default|risk|card|transaction|trans\b|withdraw|credit|debit|balance|deposit|"
    r"payment|order|statement|issuance|pension|interest|insurance|household|owner|disponent|disposition|"
    r"district|region|branch|bohemia|moravia|prague|salary|unemploy|crime|inhabitant|municipal|entrepreneur|"
    r"gender|female|male|\bwom[ae]n\b|\bm[ae]n\b|born|birth|\bages?\b|oldest|youngest|bank|amount|money|"
    r"how many|how much|average|total|percent|ratio|\brates?\b|\bshare\b|count|number of|\bsum\b|trend|month|year|19\d\d|"
    r"\bcode\b|column|table|\bmean(s|ing)?\b|dictionary|\bA\d{1,2}\b|status",
    re.IGNORECASE,
)


@dataclass
class InputCheck:
    allowed: bool
    question: str
    flags: list[str] = field(default_factory=list)
    message: str = ""


def check_input(question: str) -> InputCheck:
    q = question.strip()
    if not q:
        return InputCheck(False, q, ["empty"], "Please ask a question.")
    if len(q) > MAX_QUESTION_CHARS:
        return InputCheck(False, q[:MAX_QUESTION_CHARS], ["too_long"],
                          f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    if hit := looks_like_injection(q):
        return InputCheck(False, q, ["prompt_injection"],
                          "That looks like an attempt to change my instructions, so I won't run it. "
                          "Ask me about accounts, loans, cards, transactions or districts instead.")
    masked, pii = mask_pii(q)
    flags = [f"pii:{p}" for p in pii]
    if _RAW_SQL.match(masked):  # typed SQL goes to the SQL validator, which gives the precise reason
        return InputCheck(True, masked, flags + ["raw_sql"])
    if not _IN_SCOPE.search(masked):
        return InputCheck(False, masked, flags + ["off_topic"],
                          "I can only answer questions about the bank's data: accounts, clients, loans, cards, "
                          "transactions and districts.")
    return InputCheck(True, masked, flags)


_CITATION = re.compile(r"\[([\w.-]+#\d+)\]")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")  # skips digits inside ids like 00688cbc


@dataclass
class OutputCheck:
    answer: str
    flags: list[str] = field(default_factory=list)


def check_output(answer: str, retrieved_ids: set[str] | None = None, sql_rows: list[tuple] | None = None,
                 context_texts: list[str] | None = None) -> OutputCheck:
    """Numbers in the answer must appear somewhere in the context (SQL rows or retrieved text).

    A flag doesn't block the answer; it's logged and shown, and the eval suite
    counts how often it fires (a rising rate means the model is making numbers up).
    """
    answer, pii = mask_pii(answer)
    flags = [f"masked:{p}" for p in pii]

    if retrieved_ids is not None:
        bad = {c for c in _CITATION.findall(answer) if c not in retrieved_ids}
        for c in bad:
            answer = answer.replace(f"[{c}]", "")
        if bad:
            flags.append(f"removed_unknown_citations:{len(bad)}")

    if sql_rows is not None or context_texts:
        known: set[str] = set()
        for row in sql_rows or []:
            for v in row:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    for r in (v, round(v), round(v, 1), round(v, 2)):
                        known.add(_norm(r))
                    if 0 <= v <= 1:  # a rate may be shown as a percentage
                        known |= {_norm(round(v * 100, d)) for d in (0, 1, 2)}
                else:
                    known |= {_norm(n) for n in _NUMBER.findall(str(v))}
        for text in context_texts or []:
            known |= {_norm(n) for n in _NUMBER.findall(text)}
        claimed = {_norm(n) for n in _NUMBER.findall(_CITATION.sub("", answer))}  # ids like [40-district#2] aren't claims
        claimed = {n for n in claimed if n and float(n) > 31}  # small numbers (days, ranks) are too noisy
        if unsupported := sorted(claimed - known):
            flags.append(f"ungrounded_numbers:{','.join(unsupported[:5])}")
    return OutputCheck(answer, flags)


def _norm(x) -> str:
    try:
        f = float(str(x).replace(",", ""))
    except ValueError:
        return ""
    return f"{f:.2f}".rstrip("0").rstrip(".")
