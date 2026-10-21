"""PII masking for anything that goes INTO the LLM or OUT to the user.

Card numbers are detected as 13-19 digit runs (spaces/dashes allowed) that pass
the Luhn checksum, which avoids masking ordinary long numbers like amounts or ids.
"""
from __future__ import annotations

import re

_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def mask_card_id(value) -> str:
    s = str(value)
    return f"**** {s[-4:]}" if len(s) >= 8 else s


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, list of what was masked)."""
    found = []

    def card(m):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append("card_number")
            return f"[card **** {digits[-4:]}]"
        return m.group(0)

    text = _CARD.sub(card, text)
    for name, pat, repl in (("ssn", _SSN, "[ssn]"), ("email", _EMAIL, "[email]"), ("phone", _PHONE, "[phone]")):
        text, n = pat.subn(repl, text)
        found += [name] * n
    return text, found
