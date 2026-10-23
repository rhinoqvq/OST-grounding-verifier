"""Run one method over the RAGTruth test inputs and write one prediction record per response.

Reads inputs.jsonl from build_inputs.py (never gold). Each record is {"id", "task", "claims",
"units", "status", "wall_ms"}: the API response's claims, plus the optimized method's answer-side
units with their spans and probabilities. Resumable: ids already in --out are skipped. Every model
call is recorded in --db, so a rerun with the same db replays unchanged calls for free.

Method and switches come from the environment like the service (GV_METHOD, GV_SOURCE_TREE, ...),
with --method taking precedence.

python benchmark/ragtruth/run.py [--method optimized] [--tasks QA Summary Data2txt]
    [--inputs var/ragtruth_full/inputs.jsonl] [--out var/ragtruth_full/preds_<method>.jsonl] [--db ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from verifier.harness.config import Settings  # noqa: E402
from verifier.harness.errors import VerifierError  # noqa: E402
from verifier.harness.pipeline import run_request  # noqa: E402
from verifier.output.schemas import VerifyRequest  # noqa: E402
from verifier.service import build_dependencies  # noqa: E402

DATA = ROOT / "var" / "ragtruth_full"


def units_from_trace(trace) -> list[dict]:
    """Answer-side units with their spans and judgments (optimized method only)."""
    tree = getattr(trace, "_tree", None)
    judgments = getattr(trace, "_judgments", {})
    if tree is None:
        return []
    out = []
    for node_id, judgment in judgments.items():
        node = tree.nodes.get(node_id)
        if node is None or node.origin != "answer":
            continue
        out.append(
            {
                "node_id": node_id,
                "kind": node.kind.value,
                "verdict": judgment.verdict.value,
                "flags": judgment.flags,
                "probabilities": judgment.probabilities,
                "evidence": judgment.evidence_sentence_ids,
                "anchor_spans": [[s.start, s.end] for s in node.anchor_spans],
            }
        )
    return out


async def run_one(item, deps, sem):
    async with sem:
        rec = {"id": item["id"], "task": item["task"]}
        started = time.perf_counter()
        try:
            req = VerifyRequest(context_passages=item["context_passages"], generated_answer=item["generated_answer"])
            response, trace = await run_request(req, deps)
            rec["claims"] = [c.model_dump() for c in response.claims]
            rec["units"] = units_from_trace(trace)
            rec["status"] = "ok"
        except VerifierError as exc:
            rec["status"] = f"error:{type(exc).__name__}"
            rec["error"] = str(exc)[:500]
        except Exception as exc:  # noqa: BLE001 - a bulk run must not die mid-way
            rec["status"] = f"unexpected:{type(exc).__name__}"
            rec["error"] = str(exc)[:500]
        rec["wall_ms"] = (time.perf_counter() - started) * 1000
        return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["QA", "Summary", "Data2txt"])
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--method", default=None, help="overrides GV_METHOD")
    ap.add_argument("--inputs", default=str(DATA / "inputs.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    overrides = {"proposer": "jev", "cache_mode": "record"}
    if args.method:
        overrides["method"] = args.method
    settings = Settings(**overrides)
    out = Path(args.out or DATA / f"preds_{settings.method}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = settings.model_copy(update={"db_path": args.db or str(out.with_suffix(".sqlite3"))})
    items = [json.loads(l) for l in open(args.inputs, encoding="utf-8") if l.strip()]
    items = [i for i in items if i["task"] in args.tasks]
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()} if out.exists() else set()
    todo = [i for i in items if i["id"] not in done]
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print(f"method={settings.method} rev={rev} tasks={args.tasks} items {len(items)} done {len(done)} todo {len(todo)}",
          flush=True)
    deps = build_dependencies(settings)
    sem = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    ok = err = 0
    try:
        with open(out, "a", encoding="utf-8") as f:
            tasks = [asyncio.create_task(run_one(i, deps, sem)) for i in todo]
            for n, t in enumerate(asyncio.as_completed(tasks), 1):
                rec = await t
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                ok += rec["status"] == "ok"
                err += rec["status"] != "ok"
                if n % 200 == 0 or n == len(tasks):
                    print(f"  {n}/{len(tasks)} ok={ok} err={err} {(time.perf_counter() - started) / 60:.1f}min", flush=True)
    finally:
        if hasattr(deps.decider, "aclose"):
            await deps.decider.aclose()
        if deps.conn is not None:
            deps.conn.close()
    print(f"finished ok={ok} err={err} -> {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
