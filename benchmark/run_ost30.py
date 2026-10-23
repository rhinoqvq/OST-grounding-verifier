"""Run one of the three methods over the frozen OST30 set (no pipeline changes).

Each item is sent exactly as a /verify request (context_passages + generated_answer); the
gold fields are never passed to the system. The data file must match the hash in
ost30.sha256. Raw model calls are recorded in a per-method
cache DB, so a re-run replays instead of paying again.

Usage:
    python benchmark/run_ost30.py --method optimized|minimal_llm|baseline|minimal_jev [--proposer jev] [--tag _sol]
    (--tag keeps a second configuration of the same method apart, e.g. minimal_llm on another model:
     results_minimal_llm_sol.jsonl; provider and model come from GV_LLM_PROVIDER / GV_OPENAI_MODEL)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verifier.harness.config import Settings  # noqa: E402
from verifier.harness.errors import VerifierError  # noqa: E402
from verifier.harness.pipeline import Dependencies, run_request  # noqa: E402
from verifier.output.schemas import VerifyRequest  # noqa: E402
from verifier.service import build_dependencies  # noqa: E402

FROZEN = Path(__file__).resolve().parent / "ost30.jsonl"
FROZEN_SHA256 = "dade47dba2088c465f02858a911cc5436a7e2c298cba5a804f2d8bb4d23d1bba"



def units_from_trace(trace) -> list[dict]:
    """Answer-side units with their spans and judgments (for span-level stats)."""
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


async def run_one(item: dict, deps: Dependencies, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        record = {"id": item["id"], "method": deps.settings.method}
        started = time.perf_counter()
        try:
            request = VerifyRequest(context_passages=item["context_passages"],
                                    generated_answer=item["generated_answer"])
            response, trace = await run_request(request, deps)
            record["claims"] = [c.model_dump() for c in response.claims]
            record["units"] = units_from_trace(trace)
            record["status"] = "ok"
        except VerifierError as exc:
            record["status"] = getattr(exc, "status", f"error:{type(exc).__name__}")
            record["error"] = str(exc)[:500]
        except Exception as exc:  # noqa: BLE001 — one failing item must not stop the run
            record["status"] = f"unexpected:{type(exc).__name__}"
            record["error"] = str(exc)[:500]
        record["wall_ms"] = (time.perf_counter() - started) * 1000
        return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["optimized", "minimal_llm", "baseline", "minimal_jev", "minimal_jev_seg"])
    parser.add_argument("--proposer", default=None, choices=["llm", "jev"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--data", default=str(FROZEN))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    data = Path(args.data)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        sys.exit(f"{data} does not match the frozen OST30 hash ({digest[:12]} != {FROZEN_SHA256[:12]})")
    items = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]

    out_dir = ROOT / "var/ost30"
    name = f"{args.method}{args.tag}"
    out_path = out_dir / f"results_{name}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [i for i in items if i["id"] not in done]

    overrides = {"method": args.method}
    if args.proposer:
        overrides["proposer"] = args.proposer
    settings = Settings(cache_mode="record", db_path=str(out_dir / f"runs_{name}.sqlite3"), **overrides)
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT, timeout=10).stdout.strip()
    meta = {"method": settings.method, "proposer": settings.proposer, "jev_model": settings.jev_model,
            "llm_provider": settings.llm_provider, "qwen_model": settings.qwen_model,
            "openai_model": settings.openai_model, "git_rev": git_rev,
            "data_sha256": digest, "concurrency": args.concurrency}
    (out_dir / f"meta_{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{meta} | items {len(items)}, done {len(done)}, running {len(todo)}")

    deps = build_dependencies(settings)
    semaphore = asyncio.Semaphore(args.concurrency)
    ok = err = 0
    try:
        with open(out_path, "a", encoding="utf-8") as out:
            tasks = [asyncio.create_task(run_one(item, deps, semaphore)) for item in todo]
            for task in asyncio.as_completed(tasks):
                record = await task
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                ok += record["status"] == "ok"
                err += record["status"] != "ok"
                print(f"  {record['id']} {record['status']} {record['wall_ms'] / 1000:.1f}s", flush=True)
    finally:
        if hasattr(deps.decider, "aclose"):
            await deps.decider.aclose()
        if deps.conn is not None:
            deps.conn.close()
    print(f"finished: ok={ok} err={err} -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
