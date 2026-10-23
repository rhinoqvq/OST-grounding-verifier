"""Build the RAGTruth test inputs (what the verifier sees) and gold (what the scorer reads).

Reads the RAGTruth dataset files (response.jsonl, source_info.jsonl from the RAGTruth
repository's dataset/ folder; directory set by GV_RAGTRUTH_DIR, default var/ragtruth) and
writes, for the 2,700 test responses (900 each of QA, Summary, Data2txt):

- inputs.jsonl: {"id", "task", "context_passages", "generated_answer"}; never contains labels
- gold.jsonl:   {"id", "task", "model", "labels"}; RAGTruth's own hallucination spans
                (start, end, label_type)

Source -> passages (label-blind, fixed before any full run):
- QA: passages split on the "passage N:" markers (tasks.split_passages).
- Summary: the article cut into verbatim chunks of at most 7,500 chars (tasks.split_article).
- Data2txt: one passage, the structured-data block exactly as the generator saw it: the prompt
  text after "Structured data:" and before the final "Overview:", whitespace-stripped.

python benchmark/ragtruth/build_inputs.py [OUT_DIR]   (default var/ragtruth_full)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tasks  # noqa: E402

TASK_ORDER = ("QA", "Summary", "Data2txt")


def data2txt_block(prompt: str) -> str:
    start = prompt.index("Structured data:") + len("Structured data:")
    end = prompt.rindex("Overview:")
    return prompt[start:end].strip()


def passages_for(task: str, source: dict) -> list[str]:
    if task == "QA":
        return tasks.split_passages(source["source_info"]["passages"])
    if task == "Summary":
        return tasks.split_article(source["source_info"])
    return [data2txt_block(source["prompt"])]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parents[1] / "var" / "ragtruth_full")
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    with open(tasks.DATA_DIR / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sources[row["source_id"]] = row
    inputs, gold = [], []
    with open(tasks.DATA_DIR / "response.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source = sources.get(row["source_id"])
            if row["split"] != "test" or source is None or source["task_type"] not in TASK_ORDER:
                continue
            task = source["task_type"]
            inputs.append({"id": row["id"], "task": task, "context_passages": passages_for(task, source),
                           "generated_answer": row["response"]})
            labels = [{"start": x["start"], "end": x["end"], "label_type": x["label_type"]} for x in row["labels"]]
            gold.append({"id": row["id"], "task": task, "model": row["model"], "labels": labels})
    for name, rows in (("inputs.jsonl", inputs), ("gold.jsonl", gold)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {t: sum(r["task"] == t for r in inputs) for t in TASK_ORDER}
    print(f"wrote {len(inputs)} responses to {out}: {counts}")


if __name__ == "__main__":
    main()
