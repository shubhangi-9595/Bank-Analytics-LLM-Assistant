"""A tiny persistent cache (SQLite) for LLM responses and embeddings.

While developing you ask the same questions and embed the same texts again and
again. Caching makes re-runs free and instant, and makes evals reproducible.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class DiskCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, k: str):
        with self._lock:
            row = self._con.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k: str, v) -> None:
        with self._lock:
            self._con.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, v))
            self._con.commit()
