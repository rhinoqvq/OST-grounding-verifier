"""Per-item latency and list-price cost for the OST30 runs (reads the run DBs; no model calls).

Each model call row carries the run_id of the request that made it; runs.request_json gives the
item. Latency is the item's wall time from results_<config>.jsonl (timer starts once the item
holds one of the 4 concurrency slots, so it excludes queueing but not contention).

Usage: python benchmark/cost_latency_ost30.py [configs...]   (default: minimal_llm_sol optimized)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var/ost30"
GOLD = Path(__file__).resolve().parent / "ost30.jsonl"
PRICE = {"jev": (0.042e-6, 0.0), "qwen": (0.15e-6, 0.50e-6), "openai": (2e-6, 10e-6)}  # per token in / out


def key(passages: list[str], answer: str) -> str:
    return hashlib.sha256(json.dumps([passages, answer], ensure_ascii=False).encode()).hexdigest()


def per_item(config: str, items: dict) -> dict:
    by_key = {key(it["context_passages"], it["generated_answer"]): iid for iid, it in items.items()}
    conn = sqlite3.connect(f"file:{OUT / f'runs_{config}.sqlite3'}?mode=ro", uri=True)
    run_item = {}
    for run_id, req in conn.execute("select run_id, request_json from runs"):
        r = json.loads(req)
        run_item[run_id] = by_key.get(key(r["context_passages"], r["generated_answer"]))
    stats = {iid: {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "call_ms": 0.0} for iid in items}
    orphan = 0
    for run_id, prov, usage, lat in conn.execute("select run_id, provider, usage_json, latency_ms from model_calls"):
        iid = run_item.get(run_id)
        if iid is None:
            orphan += 1
            continue
        u = json.loads(usage) if usage else {}
        i = int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
        o = int(u.get("output_tokens") or u.get("completion_tokens") or 0)
        pin, pout = PRICE[prov]
        s = stats[iid]
        s["calls"] += 1
        s["in"] += i
        s["out"] += o
        s["usd"] += i * pin + o * pout
        s["call_ms"] += lat or 0.0
    conn.close()
    for line in (OUT / f"results_{config}.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        stats[rec["id"]]["wall_s"] = rec["wall_ms"] / 1000
    return {"items": stats, "orphan_calls": orphan}


def main() -> None:
    configs = sys.argv[1:] or ["minimal_llm_sol", "optimized"]
    items = {json.loads(l)["id"]: json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines()}
    out = {c: per_item(c, items) for c in configs}
    for c, r in out.items():
        tot = r["items"].values()
        print(f"{c}: calls {sum(s['calls'] for s in tot)}, in {sum(s['in'] for s in tot)}, "
              f"usd {sum(s['usd'] for s in tot):.4f}, orphan calls {r['orphan_calls']}")
    rows = []
    for iid, it in items.items():
        row = {"id": iid, "type": it["type"], "context_chars": sum(len(p) for p in it["context_passages"]),
               "answer_chars": len(it["generated_answer"]), "claims": len(it["gold_claims"])}
        for c in configs:
            row[c] = out[c]["items"][iid]
        rows.append(row)
    (OUT / "cost_latency.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
