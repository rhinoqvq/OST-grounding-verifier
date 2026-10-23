"""Shared loaders for RAGTruth task splits (pure data, no verifier imports).

Every version under comparison reads the inputs built from these loaders, so they all
see byte-identical passages.

Washing rules:
- QA: the passage list split on the "passage N:" markers.
- Summary: source_info is one news-article string. When these inputs were fixed the
  service capped a passage at 8000 chars, so articles above CAP are cut at newline boundaries (greedy
  packing; sentence-boundary fallback; hard cut as last resort). Chunks are
  verbatim slices — "".join(chunks) == article — so no text is altered.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# local copy of the RAGTruth repository's dataset/ folder (response.jsonl, source_info.jsonl);
# override with GV_RAGTRUTH_DIR
DATA_DIR = Path(os.environ.get("GV_RAGTRUTH_DIR") or Path(__file__).resolve().parents[2] / "var" / "ragtruth")
CAP = 7500  # headroom under the 8000-char service limit

_PASSAGE_SPLIT = re.compile(r"\n\n(?=passage \d+:)")
_PASSAGE_PREFIX = re.compile(r"^passage \d+:")


def split_passages(raw: str) -> list[str]:
    parts = _PASSAGE_SPLIT.split(raw.rstrip("\n"))
    return [_PASSAGE_PREFIX.sub("", p) for p in parts]


def split_article(text: str, cap: int = CAP) -> list[str]:
    """Cut a long article into verbatim chunks of at most `cap` chars.

    Per window: prefer the last newline, then the last sentence end, and only
    hard-cut when neither exists, so a cut never lands mid-sentence when a
    boundary is available.
    """
    if len(text) <= cap:
        return [text]
    newline_cuts = [m.end() for m in re.finditer(r"\n+", text)]
    sentence_cuts = [m.end() for m in re.finditer(r"[.!?][\"')\]]?\s+", text)]
    chunks: list[str] = []
    start = 0
    while len(text) - start > cap:
        cut = None
        for cuts in (newline_cuts, sentence_cuts):
            inside = [c for c in cuts if start < c <= start + cap]
            if inside:
                cut = inside[-1]
                break
        if cut is None:
            cut = start + cap
        chunks.append(text[start:cut])
        start = cut
    chunks.append(text[start:])
    assert "".join(chunks) == text, "washing must not alter the article"
    return chunks


def _test_rows(task_type: str) -> tuple[dict, list[dict]]:
    sources: dict = {}
    with open(DATA_DIR / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task_type"] == task_type:
                sources[row["source_id"]] = row
    rows = []
    with open(DATA_DIR / "response.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "test" and row["source_id"] in sources:
                rows.append(row)
    return sources, rows


def load_qa_test() -> list[dict]:
    sources, rows = _test_rows("QA")
    return [
        {
            "response_id": r["id"],
            "source_id": r["source_id"],
            "model": r["model"],
            "quality": r["quality"],
            "passages": split_passages(sources[r["source_id"]]["source_info"]["passages"]),
            "answer": r["response"],
            "gold_labels": r["labels"],
        }
        for r in rows
    ]


def load_summary_test() -> list[dict]:
    sources, rows = _test_rows("Summary")
    return [
        {
            "response_id": r["id"],
            "source_id": r["source_id"],
            "model": r["model"],
            "quality": r["quality"],
            "passages": split_article(sources[r["source_id"]]["source_info"]),
            "answer": r["response"],
            "gold_labels": r["labels"],
        }
        for r in rows
    ]
