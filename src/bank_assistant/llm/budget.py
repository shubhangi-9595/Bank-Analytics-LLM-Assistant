"""Spend tracking so a loop bug can't drain a $100 student credit.

Prices are per 1M tokens in USD and are APPROXIMATE - check the Azure OpenAI
pricing page for your region and update them. They only need to be roughly
right for the guard to do its job.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

PRICES_PER_1M = {
    # model/deployment: (input, output)
    "gpt-5-mini": (0.25, 2.00),   # output includes hidden reasoning tokens
    "gpt-5-nano": (0.05, 0.40),
    "text-embedding-3-small": (0.02, 0.0),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(f"spend limit ${self.limit_usd:.2f} reached (spent ${self.spent_usd:.4f})")

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        price_in, price_out = PRICES_PER_1M.get(model, (1.0, 4.0))  # unknown model: assume expensive
        cost = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.spent_usd += cost
        return cost

    def summary(self) -> str:
        return (f"{self.calls} calls, {self.input_tokens:,} in / {self.output_tokens:,} out tokens, "
                f"~${self.spent_usd:.4f}")
