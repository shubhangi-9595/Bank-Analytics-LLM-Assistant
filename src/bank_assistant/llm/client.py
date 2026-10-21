"""Thin wrapper around Azure OpenAI chat, with caching, retries and a spend limit.

Everything that needs an LLM goes through `LLMClient.chat`, so caching, cost
tracking and retries live in one place.

Model notes (October 2026): gpt-4o-mini and gpt-4.1-mini are retired on Azure.
The default is gpt-5-mini, a REASONING model, which differs from older chat models:
- no `temperature` (always the default), so answers can vary slightly run to run
- `max_completion_tokens` instead of `max_tokens`, and the limit INCLUDES hidden
  reasoning tokens, so it must leave room for thinking
- `reasoning_effort` controls how much it thinks; "minimal"/"low" is cheapest
If a deployment rejects a parameter, the client drops it and retries once,
so the same code works with older and newer models.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from bank_assistant.config import Settings, get_settings
from bank_assistant.llm.budget import Budget
from bank_assistant.llm.cache import DiskCache

REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    cached: bool = False


def azure_v1_client(settings: Settings):
    """Azure's v1 endpoint works with the standard OpenAI client and needs no api-version."""
    from openai import OpenAI

    endpoint = settings.azure_endpoint.rstrip("/")
    base_url = endpoint if endpoint.endswith("/openai/v1") else f"{endpoint}/openai/v1"
    return OpenAI(base_url=base_url + "/", api_key=settings.azure_api_key, max_retries=2, timeout=60)


def is_reasoning_model(name: str) -> bool:
    return name.lower().startswith(REASONING_PREFIXES)


class LLMClient:
    def __init__(self, settings: Settings | None = None, budget: Budget | None = None, client=None):
        self.s = settings or get_settings()
        if client is None and not (self.s.azure_endpoint and self.s.azure_api_key):
            raise RuntimeError("Azure OpenAI is not configured: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        self.client = client or azure_v1_client(self.s)
        self.model = self.s.azure_chat_deployment
        self.budget = budget or Budget(self.s.max_spend_usd)
        self.cache = DiskCache(self.s.cache_dir / "llm.sqlite")

    def _params(self, max_tokens: int) -> dict:
        if is_reasoning_model(self.s.azure_chat_model or self.model):
            # reasoning tokens count toward the cap, so leave headroom for them
            return {"max_completion_tokens": max_tokens + 1500, "reasoning_effort": self.s.reasoning_effort}
        return {"max_tokens": max_tokens, "temperature": 0}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | dict | None = None,
             json_mode: bool = False, max_tokens: int = 600) -> LLMResponse:
        key = DiskCache.key(self.model, messages, tools, tool_choice, json_mode, max_tokens)
        if (hit := self.cache.get(key)) is not None:
            d = json.loads(hit)
            return LLMResponse(d["text"], [ToolCall(**t) for t in d["tool_calls"]], cached=True)

        self.budget.check()
        kwargs = dict(model=self.model, messages=messages, **self._params(max_tokens))
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._create(kwargs)
        usage = resp.usage
        self.budget.record(self.s.azure_chat_model or self.model, usage.prompt_tokens, usage.completion_tokens)
        msg = resp.choices[0].message
        calls = [ToolCall(tc.function.name, json.loads(tc.function.arguments or "{}")) for tc in (msg.tool_calls or [])]
        out = LLMResponse(msg.content or "", calls)
        if out.text or out.tool_calls:  # don't cache empty replies (e.g. token cap hit while reasoning)
            self.cache.set(key, json.dumps({"text": out.text, "tool_calls": [c.__dict__ for c in calls]}))
        return out

    def _create(self, kwargs: dict):
        dropped: set[str] = set()
        for attempt in range(5):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                text = str(e)
                status = getattr(e, "status_code", None)
                if status == 429 or "429" in text:  # rate limit: student quotas are small
                    time.sleep(2 ** min(attempt, 3) * 2)
                    continue
                bad = self._unsupported_param(text, kwargs)
                if status == 400 and bad and bad not in dropped:
                    dropped.add(bad)
                    value = kwargs.pop(bad)
                    if bad == "max_tokens":  # older/newer naming of the same limit
                        kwargs["max_completion_tokens"] = value
                    elif bad == "max_completion_tokens":
                        kwargs["max_tokens"] = value
                    continue
                raise
        raise RuntimeError("Azure OpenAI request kept failing after retries")

    @staticmethod
    def _unsupported_param(error_text: str, kwargs: dict) -> str | None:
        if not re.search(r"unsupported|not supported|unrecognized|does not support", error_text, re.I):
            return None
        for p in ("temperature", "reasoning_effort", "max_tokens", "max_completion_tokens", "response_format"):
            if p in kwargs and p in error_text:
                return p
        return None


def get_llm(settings: Settings | None = None, budget: Budget | None = None) -> LLMClient | None:
    """Returns None in offline mode: callers then use their rule-based fallback."""
    s = settings or get_settings()
    if s.llm_provider == "offline":
        return None
    return LLMClient(s, budget)
