"""Paired comparison of two RAGTruth runs: answer-level and character-level F1, with intervals.

Per response the response-level tp/fp/fn and character tp/fp/fn (a failed request counts as
the wrong answer: a miss on a hallucinated response, an alarm on a clean one); paired bootstrap over responses resampled within each task, 10,000 resamples, seed
20261022, percentile 95% CI of (A - B). "A better" only when the interval excludes 0.

Each side is one or more preds files in the full run's record schema; together they must cover
the tasks being compared.

python benchmark/ragtruth/paired.py --a A.jsonl[.gz] --b B.jsonl[.gz] [--tasks QA Summary Data2txt]
    [--names A B] [--data var/ragtruth_full]   (inputs.jsonl and gold.jsonl from build_inputs.py)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spans import char_union, flagged_claim_spans  # noqa: E402


def _data_dir() -> Path:
    for i, a in enumerate(sys.argv):
        if a == "--data" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return Path(os.environ.get("GV_RAGTRUTH_FULL") or HERE.parents[1] / "var" / "ragtruth_full")


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


FULL = _data_dir()
inputs = {json.loads(l)["id"]: json.loads(l) for l in _open(FULL / "inputs.jsonl") if l.strip()}
gold = {json.loads(l)["id"]: json.loads(l) for l in _open(FULL / "gold.jsonl") if l.strip()}


def load(paths):
    preds = {}
    for p in paths:
        for line in _open(p):
            if line.strip():
                r = json.loads(line)
                preds[r.get("id") or r["response_id"]] = r
    return preds


def per_response(preds, ids):
    out = {}
    for i in ids:
        g, p = gold[i], preds[i]
        is_bad = bool(g["labels"])
        gchars = char_union([(x["start"], x["end"]) for x in g["labels"]])
        if p.get("status", "ok") != "ok":
            flag, pchars = (not is_bad), set()
        else:
            flag = any(c["verdict"] != "SUPPORTED" for c in p["claims"])
            groups, _ = flagged_claim_spans(p, inputs[i]["generated_answer"], strict=True)
            pchars = char_union([s for grp in groups for s in grp])
        out[i] = (flag and is_bad, flag and not is_bad, (not flag) and is_bad,
                  len(pchars & gchars), len(pchars - gchars), len(gchars - pchars))
    return out


def f1(sums, offset):
    tp, fp, fn = sums[..., offset], sums[..., offset + 1], sums[..., offset + 2]
    return np.where(tp > 0, 200 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["QA", "Summary", "Data2txt"])
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--data", default=None, help="folder with inputs.jsonl and gold.jsonl")
    args = ap.parse_args()
    ids_by = {t: [i for i in gold if gold[i]["task"] == t] for t in args.tasks}
    all_ids = [i for t in args.tasks for i in ids_by[t]]
    a, b = per_response(load(args.a), all_ids), per_response(load(args.b), all_ids)
    groups = {t: [t] for t in args.tasks}
    if {"Summary", "Data2txt"} <= set(args.tasks):
        groups["Summary+Data2txt"] = ["Summary", "Data2txt"]
    if len(args.tasks) > 1:
        groups["Overall"] = list(args.tasks)
    rng = np.random.default_rng(20261022)
    full, boot = {}, {}
    for t, ids in ids_by.items():
        x = np.array([a[i] for i in ids], dtype=float)
        y = np.array([b[i] for i in ids], dtype=float)
        idx = rng.integers(0, len(ids), size=(10_000, len(ids)))
        full[t] = (x.sum(axis=0), y.sum(axis=0))
        boot[t] = tuple(np.concatenate([z[idx[k:k + 500]].sum(axis=1) for k in range(0, 10_000, 500)]) for z in (x, y))
    na, nb = args.names
    print(f"{'subset':17s} {'metric':8s} {nb:>8s} {na:>8s} {'diff':>7s}  95% CI            reading")
    for name, ts in groups.items():
        for metric, off in (("resp F1", 0), ("char F1", 3)):
            fa = float(f1(sum(full[t][0] for t in ts), off))
            fb = float(f1(sum(full[t][1] for t in ts), off))
            d = f1(sum(boot[t][0] for t in ts), off) - f1(sum(boot[t][1] for t in ts), off)
            lo, hi = np.percentile(d, [2.5, 97.5])
            reading = f"{na} better" if lo > 0 else f"{na} worse" if hi < 0 else "no evidence"
            print(f"{name:17s} {metric:8s} {fb:8.1f} {fa:8.1f} {fa - fb:+7.2f}  [{lo:+.2f}, {hi:+.2f}]  {reading}")


if __name__ == "__main__":
    main()
