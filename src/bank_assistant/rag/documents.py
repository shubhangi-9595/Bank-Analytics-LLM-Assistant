"""Documents for retrieval: the data dictionary in src/bank_assistant/knowledge/*.md.

Why RAG over a data dictionary? Real company schemas are full of codes and cryptic column
names (here: A11, 'POPLATEK PO OBRATU', status 'D'). A model can't guess what they mean, and
real catalogs are too big to paste into every prompt. Retrieving just the relevant entries
per question is how text-to-SQL assistants handle that, and it's measurable: see evals.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_knowledge(directory: str | Path) -> list[Document]:
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        docs.append(Document(doc_id=path.stem, source="data_dictionary", title=title, text=text))
    return docs


def save_jsonl(docs: list[Document], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")


def load_jsonl(path: str | Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]
