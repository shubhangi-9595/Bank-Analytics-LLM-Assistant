"""The LLM code paths, tested with scripted fake models (no API calls, no cost).

These check OUR logic around the model: tool-call parsing, the SQL self-correction loop,
the knowledge modes, fallbacks when the API fails, caching, and what we send to Azure.
"""
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace as NS

from bank_assistant.agent.assistant import Assistant
from bank_assistant.agent.router import LLMRouter
from bank_assistant.config import Settings
from bank_assistant.data.bird import load_questions
from bank_assistant.evals.bird import run_mode, score_runs
from bank_assistant.llm.budget import Budget, BudgetExceeded
from bank_assistant.llm.client import LLMClient, LLMResponse, ToolCall, azure_v1_client
from bank_assistant.rag.index import load_retriever, make_chunks
from bank_assistant.sql.executor import SQLExecutor, schema_ddl
from bank_assistant.sql.text_to_sql import LLMTextToSQL


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _t2s(demo, llm):
    return LLMTextToSQL(llm, SQLExecutor(demo.db_path), schema_ddl(demo.db_path),
                        load_retriever(demo.index_dir, "hybrid", demo), [c.text for c in make_chunks(demo.knowledge_dir)])


def test_router_reads_tool_calls():
    llm = FakeLLM([LLMResponse(tool_calls=[ToolCall("score_loan", {"loan_id": 7}),
                                           ToolCall("search_dictionary", {"query": "status D"})])])
    r = LLMRouter(llm).route("why is loan 7 risky and what does status D mean")
    assert r.tools == ["score_loan", "search_dictionary"] and r.loan_id == 7
    assert llm.calls[0][1]["tool_choice"] == "required"


def test_router_falls_back_to_rules_when_api_fails():
    r = LLMRouter(FakeLLM([BudgetExceeded("limit")])).route("How risky is loan 12?")
    assert r.tools == ["score_loan"] and "unavailable" in r.reason


def test_text_to_sql_self_corrects_once(demo):
    llm = FakeLLM([LLMResponse(json.dumps({"sql": "SELECT COUNT(*) FROM loans"})),   # no such table
                   LLMResponse(json.dumps({"sql": "SELECT COUNT(*) FROM loan"}))])
    attempt = _t2s(demo, llm).generate_and_run("how many loans?", mode="none")
    assert attempt.repaired and attempt.result.ok and attempt.result.rows[0][0] > 0
    assert "failed" in llm.calls[1][0][-1]["content"]          # the error was fed back


def test_knowledge_modes_change_only_the_notes(demo):
    seen = {}
    for mode in ("none", "retrieved", "full", "oracle"):
        llm = FakeLLM([LLMResponse(json.dumps({"sql": "SELECT 1"}))])
        _t2s(demo, llm).generate_and_run("accounts with weekly statements", mode=mode, oracle="HINT")
        seen[mode] = llm.calls[0][0][-1]["content"]
    assert "Notes:" not in seen["none"]
    assert "POPLATEK TYDNE" in seen["retrieved"] and len(seen["full"]) > len(seen["retrieved"])
    assert seen["oracle"].endswith("HINT")


def test_bird_scoring_end_to_end_with_a_perfect_fake_model(demo):
    qs = load_questions(demo.bird_questions)
    gold = {q.question: q.gold_sql for q in qs}

    class Gold:
        def chat(self, messages, **kw):
            q = messages[-1]["content"].split("\n")[0].removeprefix("Question: ")
            return LLMResponse(json.dumps({"sql": gold[q]}))

    con = sqlite3.connect(f"file:{demo.db_path}?mode=ro", uri=True)
    preds = run_mode(_t2s(demo, Gold()), con, qs, "oracle")
    assert score_runs({"oracle": preds})["modes"]["oracle"]["ex"]["value"] == 1.0


def test_answer_prompt_fences_documents_and_masks_pii(demo):
    a = Assistant(demo)
    fake = FakeLLM([LLMResponse(tool_calls=[ToolCall("search_dictionary", {"query": "weekly"})]),
                    LLMResponse("Weekly means POPLATEK TYDNE [10-account-client#0]. Card 4111 1111 1111 1111 [made-up#9].")])
    a.llm, a.router = fake, LLMRouter(fake)
    ans = a.ask("What does POPLATEK TYDNE mean?")
    assert "<document id=" in fake.calls[1][0][1]["content"]
    assert "4111 1111 1111 1111" not in ans.answer and "[made-up#9]" not in ans.answer


# ------------------------------------------------------------------- the real client, fake Azure API
class APIError(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status_code = status


class FakeAzure:
    """Mimics chat.completions.create; rejects `temperature` like a reasoning model does."""

    def __init__(self, reject=("temperature",), rate_limits=0):
        self.reject, self.rate_limits, self.requests = set(reject), rate_limits, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(dict(kw))
        if self.rate_limits:
            self.rate_limits -= 1
            raise APIError("Error code: 429 - rate limit", 429)
        for p in self.reject:
            if p in kw:
                raise APIError(f"Unsupported parameter: '{p}' is not supported with this model.", 400)
        msg = NS(content='{"sql": "SELECT 1"}', tool_calls=None)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=100, completion_tokens=50))


def _client(tmp_path, fake, model="gpt-5-mini"):
    s = replace(Settings(), cache_dir=tmp_path, azure_chat_deployment=model, azure_chat_model=model,
                azure_endpoint="https://x.openai.azure.com/", azure_api_key="k")
    return LLMClient(s, Budget(1.0), client=fake)


def test_reasoning_model_gets_reasoning_params(tmp_path):
    fake = FakeAzure()
    _client(tmp_path, fake).chat([{"role": "user", "content": "hi"}], max_tokens=100)
    sent = fake.requests[0]
    assert "temperature" not in sent and "max_tokens" not in sent
    assert sent["max_completion_tokens"] > 100 and sent["reasoning_effort"] == "minimal"


def test_unsupported_param_is_dropped_and_retried(tmp_path):
    fake = FakeAzure(reject=("reasoning_effort",))
    assert _client(tmp_path, fake).chat([{"role": "user", "content": "hi"}]).text
    assert len(fake.requests) == 2 and "reasoning_effort" not in fake.requests[1]


def test_rate_limit_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("bank_assistant.llm.client.time.sleep", lambda s: None)
    fake = FakeAzure(rate_limits=2)
    assert _client(tmp_path, fake).chat([{"role": "user", "content": "hi"}]).text and len(fake.requests) == 3


def test_cache_makes_repeat_calls_free(tmp_path):
    fake = FakeAzure()
    c = _client(tmp_path, fake)
    c.chat([{"role": "user", "content": "same"}])
    assert c.chat([{"role": "user", "content": "same"}]).cached and len(fake.requests) == 1


def test_budget_blocks_after_limit():
    b = Budget(limit_usd=0.001)
    b.record("gpt-5-mini", input_tokens=10_000)
    try:
        b.check()
        raise AssertionError("should have raised")
    except BudgetExceeded:
        pass


def test_endpoint_becomes_v1_url():
    for ep in ("https://x.openai.azure.com/", "https://x.openai.azure.com", "https://x.openai.azure.com/openai/v1/"):
        s = replace(Settings(), azure_endpoint=ep, azure_api_key="k")
        assert str(azure_v1_client(s).base_url) == "https://x.openai.azure.com/openai/v1/"
