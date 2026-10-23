"""Check ost30.jsonl against the dataset format (docs/appendix/ost30-extraction.zh-CN.md section 9).

Request fields must match the assignment's /verify request; gold claims, once filled,
must match its response invariants: verdict enum, citations non-empty for
SUPPORTED/CONTRADICTED and empty for UNSUPPORTED, index in bounds, quote a verbatim
substring of context_passages[index].
"""
import json
import sys
from pathlib import Path

CATEGORIES = {"fully_supported", "subtle_distortion", "extrinsic_hallucination", "multi_passage", "no_claims"}
VERDICTS = {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
EXPECTED_TYPES = {f"{i:02d}" for i in range(1, 31)}


def check(rec: dict) -> list[str]:
    errs = []
    for key, typ in (("id", str), ("category", str), ("type_id", str), ("context_passages", list),
                     ("generated_answer", str), ("gold_claims", list), ("gold_status", str), ("source", dict)):
        if not isinstance(rec.get(key), typ):
            errs.append(f"{key}: missing or not {typ.__name__}")
    if errs:
        return errs
    if rec["category"] not in CATEGORIES:
        errs.append(f"category {rec['category']!r} not in {sorted(CATEGORIES)}")
    passages = rec["context_passages"]
    if not passages or not all(isinstance(p, str) and p for p in passages):
        errs.append("context_passages must be a non-empty list of non-empty strings")
    if rec["gold_status"] not in ("pending", "labeled", "reviewed"):
        errs.append(f"gold_status {rec['gold_status']!r}")
    if rec["gold_status"] != "pending" and not rec["gold_claims"] and rec["category"] != "no_claims":
        errs.append("gold_status says labeled but gold_claims is empty")
    for i, c in enumerate(rec["gold_claims"]):
        where = f"gold_claims[{i}]"
        if not isinstance(c.get("claim_text"), str) or not c["claim_text"]:
            errs.append(f"{where}.claim_text missing")
        if c.get("verdict") not in VERDICTS:
            errs.append(f"{where}.verdict {c.get('verdict')!r}")
        cites = c.get("citations")
        if not isinstance(cites, list):
            errs.append(f"{where}.citations must be a list")
            continue
        if c.get("verdict") == "UNSUPPORTED" and cites:
            errs.append(f"{where}: UNSUPPORTED must have no citations")
        if c.get("verdict") in ("SUPPORTED", "CONTRADICTED") and not cites:
            errs.append(f"{where}: {c['verdict']} needs at least one citation")
        for j, q in enumerate(cites):
            idx, quote = q.get("index"), q.get("quote")
            if not isinstance(idx, int) or not 0 <= idx < len(passages):
                errs.append(f"{where}.citations[{j}].index {idx!r} out of range")
            elif not isinstance(quote, str) or not quote or quote not in passages[idx]:
                errs.append(f"{where}.citations[{j}].quote is not a verbatim substring of passage {idx}")
        for sp in c.get("answer_spans", []):
            if not 0 <= sp["start"] < sp["end"] <= len(rec["generated_answer"]):
                errs.append(f"{where}.answer_spans {sp} out of range")
    # optional span fields added by enrich_ost30.py
    n = len(rec["generated_answer"])
    for key in ("gold_error_spans", "nonclaim_spans"):
        for sp in rec.get(key, []):
            if not 0 <= sp["start"] < sp["end"] <= n:
                errs.append(f"{key} {sp} out of range")
    for sp in rec.get("gold_error_spans", []):
        claim = rec["gold_claims"][sp["claim_index"]]
        if claim["verdict"] == "SUPPORTED":
            errs.append(f"gold_error_spans points at a SUPPORTED claim ({sp['claim_index']})")
        elif not any(a["start"] <= sp["start"] and sp["end"] <= a["end"] for a in claim.get("answer_spans", [])):
            errs.append(f"gold_error_spans {sp} is outside its claim")
    return errs


def main(path: str) -> int:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    bad, seen, pending = 0, set(), 0
    for n, line in enumerate(lines, 1):
        rec = json.loads(line)
        errs = check(rec)
        if rec.get("id") in seen:
            errs.append("duplicate id")
        seen.add(rec.get("id"))
        pending += rec.get("gold_status") == "pending"
        for e in errs:
            print(f"line {n} {rec.get('id')}: {e}")
        bad += bool(errs)
    types = {json.loads(l)["type_id"] for l in lines}
    print(f"{len(lines)} records, {bad} with errors, {pending} gold pending; "
          f"missing types: {', '.join(sorted(EXPECTED_TYPES - types)) or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
