"""Download the raw data (about 70 MB). Run once:

    python -m bank_assistant.data.download

Sources, all public and with no login:
- Czech bank data (PKDD'99 "Berka" dataset): the 8 original CSV files, from a GitHub mirror.
- BIRD text-to-SQL benchmark, dev questions (CC BY-SA 4.0): we keep the 106 questions about the
  `financial` database, which is this bank. GitHub mirror first, Hugging Face as a backup.

Nothing here is committed to git (see .gitignore).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from bank_assistant.config import Settings, get_settings

BERKA_BASE = "https://raw.githubusercontent.com/Kusainov/czech-banking-fin-analysis/master/"
BERKA_FILES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

BIRD_SOURCES = [
    "https://raw.githubusercontent.com/1050727345hu-web/Bird_Pandas/main/Origin_dev_Bird.json",
    "https://huggingface.co/datasets/prem-research/birdbench/resolve/main/validation/validation.json?download=true",
]
# Optional: BIRD's own SQLite build of the same bank (71 MB). Only needed if you want to check
# that our database matches theirs exactly: python -m bank_assistant.data.download --bird-sqlite
BIRD_SQLITE = ("https://huggingface.co/datasets/prem-research/birdbench/resolve/main/"
               "validation/dev_databases/financial/financial.sqlite?download=true")

HEADERS = {"User-Agent": "bank-analytics-assistant (student project)"}


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip  {dest.name} (already downloaded)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"get   {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def download_berka(s: Settings) -> None:
    for name in BERKA_FILES:
        fetch(BERKA_BASE + f"{name}.csv", s.raw_dir / "berka" / f"{name}.csv")


def download_bird(s: Settings) -> None:
    if s.bird_questions.exists():
        print(f"skip  {s.bird_questions.name} (already downloaded)")
        return
    tmp = s.raw_dir / "bird_dev_full.json"
    for url in BIRD_SOURCES:
        try:
            fetch(url, tmp)
            break
        except Exception as e:  # mirrors move; try the next one
            print(f"warn  {url.split('/')[2]}: {e}", file=sys.stderr)
    else:
        raise SystemExit("could not download the BIRD questions from any source")
    questions = [q for q in json.loads(tmp.read_text(encoding="utf-8")) if q["db_id"] == "financial"]
    s.bird_questions.write_text(json.dumps(questions, indent=1), encoding="utf-8")
    tmp.unlink()
    print(f"kept  {len(questions)} BIRD questions about the bank -> {s.bird_questions.name}")


def main() -> None:
    s = get_settings()
    download_berka(s)
    download_bird(s)
    if "--bird-sqlite" in sys.argv:
        fetch(BIRD_SQLITE, s.raw_dir / "bird_financial.sqlite")


if __name__ == "__main__":
    main()
