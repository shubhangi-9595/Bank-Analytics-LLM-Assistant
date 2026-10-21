"""Chunking: split documents into pieces small enough to embed and retrieve precisely.

Three strategies, from naive to better:
1. fixed      - every N tokens, with overlap. Simple, but cuts sentences in half.
2. recursive  - split on paragraphs, then sentences, then words, packing pieces up
                to N tokens. Keeps sentences whole.
3. structured - split on section headings first (the data dictionary has one per entry), recurse
                inside long sections, and prefix every chunk with its document
                title + section heading ("contextual chunk header"), so a chunk
                that just says "'B' = contract finished, loan not paid" still says
                it's about loan.status.

With "fixed" and "recursive", a document shorter than the chunk size stays whole:
splitting a short, single-topic text only loses context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bank_assistant.rag.documents import Document

_WORD = re.compile(r"\S+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_HEADING = re.compile(
    r"^(?:#{1,4}\s+.+|[A-Z][A-Z0-9 ,&/()\-]{6,80}|(?:\d+\.|[IVX]+\.)\s+[A-Z].{3,80})\s*$", re.MULTILINE
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    """Cheap approximation (~1.3 tokens per English word). Good enough for sizing.

    Swap for tiktoken if you need exact counts for a specific model.
    """
    return int(len(_WORD.findall(text)) * 1.3) + 1


def _words_to_tokens(n_tokens: int) -> int:
    return max(1, int(n_tokens / 1.3))


# --------------------------------------------------------------------------- 1. fixed
def split_fixed(text: str, size: int = 256, overlap: int = 32) -> list[str]:
    words = text.split()
    step_w, size_w = _words_to_tokens(size - overlap), _words_to_tokens(size)
    return [" ".join(words[i:i + size_w]) for i in range(0, max(len(words), 1), step_w) if words[i:i + size_w]]


# --------------------------------------------------------------------------- 2. recursive
def split_recursive(text: str, size: int = 256, overlap: int = 32) -> list[str]:
    """Pack paragraphs -> sentences -> words into chunks of <= size tokens."""
    units = _units(text, size)
    chunks, current = [], []
    for unit in units:
        if current and count_tokens(" ".join(current + [unit])) > size:
            chunks.append(" ".join(current))
            current = _tail_for_overlap(current, overlap)
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _units(text: str, size: int) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if count_tokens(para) <= size:
            out.append(para)
            continue
        for sent in _SENTENCE.split(para):
            if count_tokens(sent) <= size:
                out.append(sent)
            else:  # a monster sentence: fall back to fixed windows
                out.extend(split_fixed(sent, size, 0))
    return out


def _tail_for_overlap(units: list[str], overlap: int) -> list[str]:
    """Carry the last sentence(s) into the next chunk, up to `overlap` tokens."""
    tail, total = [], 0
    for u in reversed(units):
        total += count_tokens(u)
        if total > overlap:
            break
        tail.insert(0, u)
    return tail


# --------------------------------------------------------------------------- 3. structured
def split_structured(text: str, size: int = 256, overlap: int = 32) -> list[tuple[str, str]]:
    """Return (section_heading, chunk_text) pairs."""
    sections, last, heading = [], 0, ""
    for m in _HEADING.finditer(text):
        if text[last:m.start()].strip():
            sections.append((heading, text[last:m.start()]))
        heading, last = m.group(0).strip("# ").strip(), m.end()
    sections.append((heading, text[last:]))
    out = []
    for h, body in sections:
        for piece in split_recursive(body, size, overlap):
            out.append((h, piece))
    return out


# --------------------------------------------------------------------------- entry point
def chunk_document(doc: Document, strategy: str = "structured", size: int = 256, overlap: int = 32) -> list[Chunk]:
    base = {**doc.metadata, "source": doc.source, "title": doc.title, "strategy": strategy}
    # A short document is one idea: keep it whole. Except with "structured", whose whole point is
    # to split on headings (a data dictionary is short but has one entry per heading).
    if count_tokens(doc.text) <= size and strategy != "structured":
        return [Chunk(f"{doc.doc_id}#0", doc.doc_id, _with_header(doc.title, "", doc.text, strategy), base)]

    if strategy == "fixed":
        pieces = [("", t) for t in split_fixed(doc.text, size, overlap)]
    elif strategy == "recursive":
        pieces = [("", t) for t in split_recursive(doc.text, size, overlap)]
    elif strategy == "structured":
        pieces = split_structured(doc.text, size, overlap)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")

    return [Chunk(f"{doc.doc_id}#{i}", doc.doc_id, _with_header(doc.title, h, t, strategy),
                  {**base, "section": h, "key": h.split(":")[0].strip() if h else ""})
            for i, (h, t) in enumerate(pieces) if t.strip()]


def _with_header(title: str, section: str, text: str, strategy: str) -> str:
    if strategy != "structured":
        return text
    header = title + (f" > {section}" if section else "")
    return f"[{header}]\n{text}"


def chunk_all(docs: list[Document], strategy: str = "structured", size: int = 256, overlap: int = 32) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, strategy, size, overlap)]
