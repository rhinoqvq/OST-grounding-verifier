"""Score OST30 runs: the end-to-end main board (claims taken from the system's own decomposition,
never from the gold) plus the engineering side board (latency, tokens, list-price cost).

Matching and aggregation rules:
- Alignment: a system claim is placed in the answer by exact substring (then without trailing
  punctuation). If it covers at least half of two or more gold sentences it is a merge and its
  verdict counts for each of them; otherwise it goes to the gold claim it overlaps most. A claim that
  cannot be placed (a rewritten claim) goes to the gold claim with the largest content-word overlap,
  taken in both directions on light stems (>= 0.5); its answer-side span is that gold sentence.
  Claims no rule can place are placed by hand in ost30_align_adjudication.json. Claims landing on a
  non-claim sentence are "non-claim"; the rest are "extra".
- A gold sentence split into several system claims takes the strongest verdict among them
  (C > U > S). A gold claim no system claim reaches is MISSED: wrong for accuracy, and a miss
  for alarms; missed real errors stay in every denominator.
- Citation hit: for gold S / C claims judged correctly, a system citation in the same passage
  overlapping a gold citation. OST30-15 claim 4 is excluded (its frozen gold citation is known
  not to support the claim).
- Evidence-checked accuracy: a correct S / C verdict counts only if its citations support the
  claim: a citation hit, or a ruling in ost30_cite_adjudication.json (missing = not
  supported). A right label resting on wrong evidence (OST30-20: the injected message) is wrong.

Results are read from var/ost30/results_<config>.jsonl (written by run_ost30.py).

Usage: python benchmark/score_ost30.py [configs...]
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
GOLD = HERE / "ost30.jsonl"
OUT = ROOT / "var/ost30"
METHODS = sys.argv[1:] or ["baseline", "minimal_llm", "optimized"]
JEV_IN = 0.042e-6                      # official price; output tokens free
QWEN_IN, QWEN_OUT = 0.15e-6, 0.50e-6   # qwen3.8-flash, third-party price, upper bound
OPENAI_IN, OPENAI_OUT = 2e-6, 10e-6    # gpt-6.1-sol, third-party price (reasoning tokens bill as output)
CITE_EXCLUDE = {("OST30-15", 3)}
ADJ = json.loads((HERE / "ost30_cite_adjudication.json").read_text(encoding="utf-8"))
# claims no text rule can place (a system's own inference, heavy rewording), placed by hand: method -> id -> claim_text -> gold #
ALIGN_ADJ = json.loads((HERE / "ost30_align_adjudication.json").read_text(encoding="utf-8"))
S, C, U, MISSED = "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "MISSED"
FLAG = {C, U}

WORD = re.compile(r"[A-Za-z0-9']+")
STOP = set("""a an the of to in on at for and or but is are was were be been being it its this that these those
with as by from has have had not no he she they them his her their there which who whom what when where
will would can could may might should do does did so than then also into about over after before""".split())


def stem(w: str) -> str:
    for suf in ("ing", "ed", "er", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w


def content(s: str) -> set[str]:
    return {stem(w.lower()) for w in WORD.findall(s) if w.lower() not in STOP and len(w) > 2}


def share(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a))


def similar(a: set, b: set) -> float:
    # both directions: a rewritten claim adds words (He -> Rabbi Joel Teitelbaum), a split one drops them
    return max(share(a, b), share(b, a))


def locate(text: str, answer: str):
    t = text.strip()
    for cand in (t, t.rstrip(".!?;, ")):
        if cand and cand in answer:
            i = answer.index(cand)
            return i, i + len(cand)
    return None


def overlap(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def align(item: dict, preds: list[dict], manual: dict | None = None):
    gold, answer = item["gold_claims"], item["generated_answer"]
    manual = manual or {}
    assigned = {k: [] for k in range(len(gold))}
    extra, nonclaim, merged = [], [], []
    for p in preds:
        span = locate(p["claim_text"], answer)
        best = None
        if p["claim_text"] in manual:
            k = manual[p["claim_text"]] - 1
            gs = gold[k]["answer_spans"][0]
            assigned[k].append({**p, "_span": span or (gs["start"], gs["end"]), "_span_from_gold": span is None})
            continue
        if span:
            hits = []
            for k, g in enumerate(gold):
                gs = (g["answer_spans"][0]["start"], g["answer_spans"][0]["end"])
                o = overlap(span, gs)
                if o:
                    hits.append((o, k, o >= 0.5 * (gs[1] - gs[0])))
            covered = [k for _, k, half in hits if half]
            if len(covered) > 1:          # one system claim merging several gold sentences
                merged.append(len(covered))
                for k in covered:
                    assigned[k].append({**p, "_span": span})
                continue
            if hits:
                best = max(hits)[1]
            elif any(overlap(span, (n["start"], n["end"])) for n in item.get("nonclaim_spans", [])):
                nonclaim.append({**p, "_span": span})
                continue
        if best is None:
            pc = content(p["claim_text"])
            scores = [(similar(pc, content(g["claim_text"])), k) for k, g in enumerate(gold)]
            if scores and max(scores)[0] >= 0.5:
                best = max(scores)[1]
                gs = gold[best]["answer_spans"][0]
                # placed by meaning, not by text: its answer-side span is the gold sentence it was matched to
                assigned[best].append({**p, "_span": span or (gs["start"], gs["end"]), "_span_from_gold": span is None})
                continue
            if any(similar(pc, content(t)) >= 0.5 for t in item.get("gold_nonclaims", [])):
                nonclaim.append({**p, "_span": span})
                continue
        if best is None:
            extra.append({**p, "_span": span})
        else:
            assigned[best].append({**p, "_span": span})
    return assigned, extra, nonclaim, merged


def strongest(preds: list[dict]) -> str:
    verdicts = {p["verdict"] for p in preds}
    return C if C in verdicts else U if U in verdicts else S if verdicts else MISSED


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def gold_cite_spans(item, claim):
    out = []
    for q in claim["citations"]:
        i = item["context_passages"][q["index"]].find(q["quote"])
        out.append((q["index"], i, i + len(q["quote"])))
    return out


def pred_cite_spans(item, preds):
    out = []
    for p in preds:
        for q in p.get("citations") or []:
            idx, quote = q.get("index"), q.get("quote") or ""
            if not isinstance(idx, int) or not 0 <= idx < len(item["context_passages"]):
                continue
            start = q.get("start_char")
            if start is None:
                start = item["context_passages"][idx].find(quote)
            if start is not None and start >= 0:
                out.append((idx, start, start + len(quote)))
    return out


def score_items(items: dict, results: dict, method: str) -> dict:
    rows, claims = [], []
    manual_align = ALIGN_ADJ.get(method, {})
    adj, unruled = ADJ.get(method, {}), []
    cite_valid = cite_total = 0
    cite_hit = cite_eligible = 0
    gold_chars = pred_chars = both_chars = 0
    unlocated_flags = spans_from_gold = 0
    for iid, item in items.items():
        res = results.get(iid)
        preds = res["claims"] if res and res["status"] == "ok" else []
        assigned, extra, nonclaim, merged = align(item, preds, manual_align.get(iid))
        verdicts = []
        for k, g in enumerate(item["gold_claims"]):
            pv = strongest(assigned[k])
            verdicts.append(pv)
            claims.append({"id": iid, "k": k, "family": item["family_id"], "category": item["category"],
                           "bucket": "ABCDEF"[(int(iid.split("-")[1]) - 1) // 5],
                           "gold": g["verdict"], "pred": pv, "ev_ok": pv == g["verdict"]})
            if g["verdict"] in (S, C) and pv == g["verdict"]:
                gs, ps = gold_cite_spans(item, g), pred_cite_spans(item, assigned[k])
                hit = any(gi == pi and overlap((ga, gb), (pa, pb)) for gi, ga, gb in gs for pi, pa, pb in ps)
                if (iid, k) not in CITE_EXCLUDE:
                    cite_eligible += 1
                    cite_hit += hit
                if not hit or (iid, k) in CITE_EXCLUDE:
                    ruling = adj.get(f"{iid}#{k + 1}")
                    if ruling is None:
                        unruled.append(f"{iid}#{k + 1}")
                    claims[-1]["ev_ok"] = bool(ruling and ruling["ok"])
        for p in preds:
            for q in p.get("citations") or []:
                cite_total += 1
                idx = q.get("index")
                cite_valid += (isinstance(idx, int) and 0 <= idx < len(item["context_passages"])
                               and bool(q.get("quote")) and q["quote"] in item["context_passages"][idx])
        # character-level localisation on the answer
        n = len(item["generated_answer"])
        gmask, pmask = bytearray(n), bytearray(n)
        for e in item["gold_error_spans"]:
            gmask[e["start"]:e["end"]] = b"\x01" * (e["end"] - e["start"])
        for p in [q for v in assigned.values() for q in v] + extra + nonclaim:
            spans_from_gold += bool(p.get("_span_from_gold"))
            if p["verdict"] in FLAG:
                if p["_span"] is None:
                    unlocated_flags += 1
                    continue
                pmask[p["_span"][0]:p["_span"][1]] = b"\x01" * (p["_span"][1] - p["_span"][0])
        gold_chars += sum(gmask)
        pred_chars += sum(pmask)
        both_chars += sum(1 for a, b in zip(gmask, pmask) if a and b)
        flagged_outside = sum(p["verdict"] in FLAG for p in extra + nonclaim)
        gold_err = any(g["verdict"] in FLAG for g in item["gold_claims"])
        pred_err = any(v in FLAG for v in verdicts) or flagged_outside > 0
        passed = all(v == g["verdict"] for v, g in zip(verdicts, item["gold_claims"])) and flagged_outside == 0
        passed_ev = passed and all(c["ev_ok"] for c in claims if c["id"] == iid)
        rows.append({"id": iid, "type": item["type"], "category": item["category"],
                     "status": res["status"] if res else "missing", "gold": [g["verdict"][0] for g in item["gold_claims"]],
                     "pred": [v[0] if v != MISSED else "-" for v in verdicts], "n_pred": len(preds),
                     "extra": len(extra), "flagged_extra_or_nonclaim": flagged_outside,
                     "merged_claims": len(merged), "split_gold": sum(len(v) > 1 for v in assigned.values()),
                     "gold_err": gold_err, "pred_err": pred_err, "pass": passed, "pass_ev": passed_ev,
                     "wall_ms": res.get("wall_ms") if res else None})

    def metrics(cl):
        acc = sum(c["gold"] == c["pred"] for c in cl) / max(1, len(cl))
        per = {lab: prf(sum(c["gold"] == lab and c["pred"] == lab for c in cl),
                        sum(c["gold"] != lab and c["pred"] == lab for c in cl),
                        sum(c["gold"] == lab and c["pred"] != lab for c in cl)) for lab in (S, C, U)}
        alarm = prf(sum(c["gold"] in FLAG and c["pred"] in FLAG for c in cl),
                    sum(c["gold"] not in FLAG and c["pred"] in FLAG for c in cl),
                    sum(c["gold"] in FLAG and c["pred"] not in FLAG for c in cl))
        return acc, sum(per[l]["F1"] for l in per) / 3, per, alarm

    acc, macro, per, alarm = metrics(claims)
    errs = [c for c in claims if c["gold"] in FLAG]
    flagged = [c for c in errs if c["pred"] in FLAG]
    resp = prf(sum(r["gold_err"] and r["pred_err"] for r in rows), sum(not r["gold_err"] and r["pred_err"] for r in rows),
               sum(r["gold_err"] and not r["pred_err"] for r in rows))
    # family bootstrap (29 and 30 share a family), 2,000 resamples
    fams = sorted({c["family"] for c in claims})
    rng = random.Random(20261022)
    boot_acc, boot_alarm, boot_ev = [], [], []
    for _ in range(2000):
        pick = [rng.choice(fams) for _ in fams]
        cl = [c for f in pick for c in claims if c["family"] == f]
        a, _, _, al = metrics(cl)
        boot_acc.append(a)
        boot_alarm.append(al["F1"])
        boot_ev.append(sum(c["ev_ok"] for c in cl) / len(cl))
    ci = lambda xs: (sorted(xs)[int(0.025 * len(xs))], sorted(xs)[int(0.975 * len(xs)) - 1])  # noqa: E731
    walls = [r["wall_ms"] for r in rows if r["wall_ms"] is not None and r["status"] == "ok"]
    cats, buckets = {}, {}
    for c in claims:
        cats.setdefault(c["category"], []).append(c["gold"] == c["pred"])
        buckets.setdefault(c["bucket"], []).append(c["gold"] == c["pred"])
    return {
        "items": len(rows), "items_ok": sum(r["status"] == "ok" for r in rows),
        "claims": len(claims), "accuracy": acc, "accuracy_ci95": ci(boot_acc), "macro_f1": macro, "per_label": per,
        "alarm": alarm, "alarm_f1_ci95": ci(boot_alarm),
        "false_support": sum(c["pred"] == S for c in errs), "missed_errors": sum(c["pred"] == MISSED for c in errs),
        "missed_claims": sum(c["pred"] == MISSED for c in claims), "errors_total": len(errs),
        "c_u_exact_among_flagged": sum(c["pred"] == c["gold"] for c in flagged) / max(1, len(flagged)),
        "response_level": resp, "items_passed": sum(r["pass"] for r in rows),
        "accuracy_evidence": sum(c["ev_ok"] for c in claims) / len(claims), "accuracy_evidence_ci95": ci(boot_ev),
        "items_passed_evidence": sum(r["pass_ev"] for r in rows), "unadjudicated": unruled,
        "extra_claims": sum(r["extra"] for r in rows), "flagged_outside_gold": sum(r["flagged_extra_or_nonclaim"] for r in rows),
        "char_level": {**prf(both_chars, pred_chars - both_chars, gold_chars - both_chars),
                       "pred_over_gold_chars": pred_chars / max(1, gold_chars), "unlocated_flagged_claims": unlocated_flags,
                       "claims_placed_by_meaning": spans_from_gold},
        "citations": {"valid": cite_valid, "total": cite_total, "hit": cite_hit, "eligible": cite_eligible},
        "by_category_accuracy": {k: sum(v) / len(v) for k, v in cats.items()},
        "by_bucket_accuracy": {k: sum(v) / len(v) for k, v in sorted(buckets.items())},
        "merged_claims": sum(r["merged_claims"] for r in rows), "split_gold": sum(r["split_gold"] for r in rows),
        "latency_s": {"p50": statistics.median(walls) / 1000 if walls else None,
                      "p95": sorted(walls)[max(0, int(0.95 * len(walls)) - 1)] / 1000 if walls else None},
        "rows": rows,
    }


def cost(method: str) -> dict:
    db = OUT / f"runs_{method}.sqlite3"
    if not db.exists():
        return {}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tok = {}
    for prov, uj in conn.execute("select provider, usage_json from model_calls"):
        u = json.loads(uj) if uj else {}
        t = tok.setdefault(prov, {"calls": 0, "in": 0, "out": 0})
        t["calls"] += 1
        t["in"] += int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
        t["out"] += int(u.get("output_tokens") or u.get("completion_tokens") or 0)
    conn.close()
    usd = sum(v["in"] * JEV_IN for p, v in tok.items() if p == "jev")
    usd += sum(v["in"] * QWEN_IN + v["out"] * QWEN_OUT for p, v in tok.items() if p == "qwen")
    usd += sum(v["in"] * OPENAI_IN + v["out"] * OPENAI_OUT for p, v in tok.items() if p == "openai")
    return {"providers": tok, "usd_estimate": usd}


def main() -> None:
    items = {json.loads(l)["id"]: json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines()}
    report = {}
    for m in METHODS:
        path = OUT / f"results_{m}.jsonl"
        if not path.exists():
            print(f"{m}: no results")
            continue
        results = {json.loads(l)["id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
        report[m] = {**score_items(items, results, m), "cost": cost(m)}
        r = report[m]
        print(f"\n== {m}: ok {r['items_ok']}/{r['items']} | claim acc {r['accuracy']:.3f} "
              f"[{r['accuracy_ci95'][0]:.3f}, {r['accuracy_ci95'][1]:.3f}] | macro-F1 {r['macro_f1']:.3f} | "
              f"alarm P/R/F1 {r['alarm']['P']:.3f}/{r['alarm']['R']:.3f}/{r['alarm']['F1']:.3f} | "
              f"false support {r['false_support']}/{r['errors_total']} | missed claims {r['missed_claims']} | "
              f"items passed {r['items_passed']}/30 | evidence-checked acc {r['accuracy_evidence']:.3f} "
              f"[{r['accuracy_evidence_ci95'][0]:.3f}, {r['accuracy_evidence_ci95'][1]:.3f}] passed {r['items_passed_evidence']}/30 "
              f"(unruled {len(r['unadjudicated'])}) | response F1 {r['response_level']['F1']:.3f} | "
              f"char F1 {r['char_level']['F1']:.3f} (pred/gold chars {r['char_level']['pred_over_gold_chars']:.2f}) | "
              f"cites valid {r['citations']['valid']}/{r['citations']['total']} hit {r['citations']['hit']}/{r['citations']['eligible']} | "
              f"p50/p95 {r['latency_s']['p50'] or 0:.1f}/{r['latency_s']['p95'] or 0:.1f} s | merges {r['merged_claims']} splits {r['split_gold']} | cost ${r['cost'].get('usd_estimate', 0):.3f}")
    (OUT / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
