"""Central settings, read from environment variables (and .env if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    db_path: Path = ROOT / "data" / "processed" / "bank.sqlite"
    index_dir: Path = ROOT / "data" / "processed" / "index"
    model_dir: Path = ROOT / "data" / "processed" / "model"
    bird_questions: Path = ROOT / "data" / "raw" / "bird_dev.json"
    knowledge_dir: Path = PACKAGE / "knowledge"
    cache_dir: Path = ROOT / ".cache"

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "offline"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "tfidf"))
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    azure_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    # Deployment = the name YOU gave it in the portal. Model = the underlying model, used for
    # pricing and to know whether it's a reasoning model. They're often the same string.
    azure_chat_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini"))
    azure_chat_model: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-5-mini"))
    reasoning_effort: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_REASONING_EFFORT", "minimal"))
    azure_embedding_deployment: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    )

    max_spend_usd: float = field(default_factory=lambda: float(os.getenv("MAX_SPEND_USD", "0.50")))

    top_k: int = 4             # knowledge snippets per question
    sql_row_limit: int = 200   # rows returned to the user / LLM
    sql_timeout_s: float = 10.0


def demo_settings(base: Path | None = None) -> Settings:
    """Settings pointing at the small synthetic build (python -m bank_assistant.demo)."""
    base = Path(base or ROOT / "data" / "demo")
    return replace(Settings(), data_dir=base, raw_dir=base / "raw", db_path=base / "bank.sqlite",
                   index_dir=base / "index", model_dir=base / "model", bird_questions=base / "bird_dev.json")


def get_settings() -> Settings:
    return demo_settings() if os.getenv("DEMO") == "1" else Settings()
