"""Add the spec's gold fields to ost30.jsonl (docs/appendix/ost30-extraction.zh-CN.md section 9), from labels already there.

Per claim:   answer_spans   where the claim sits in generated_answer
             evidence_sets  [citations] for S / C (one AND-set; no alternatives are annotated), [] for U
Per record:  gold_error_spans  the wrong part of each C / U claim: ANAH's correction '"X" to "Y"' gives X;
                               for answers we wrote, the part we changed; otherwise the whole claim
             nonclaim_spans    where gold_nonclaims sit in the answer
             source_hash       sha256 of context_passages
             family_id         dataset + topic / item id (29 and 30 share one)
Run last, after gen_ost30_synth.py.

Usage: enrich_ost30.py OST30.jsonl [PICKS.json]   (default var/anah/picks.json)
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
PICKS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "var/anah/picks.json"
picks = {o["type_id"]: o["pick"] for o in json.loads(PICKS.read_text(encoding="utf-8"))
         if o["pick"]}

# claims we wrote that are not verbatim pieces of the answer: (record id, claim index) -> answer substring
CLAIM_IN_ANSWER = {("OST30-14", 0): "The creator of A Burial at Ornans is Gustave Courbet",
                   ("OST30-14", 1): "who was born on 10 June 1819"}
# the changed part of answers we wrote: (record id, claim index) -> answer substring
ERROR_PART = {("OST30-10", 1): "always include", ("OST30-25", 1): "In an intelligent tutoring system",
              ("OST30-28", 1): "31,626", ("OST30-22", 0): "All"}


def find(text: str, piece: str, start: int = 0) -> dict:
    i = text.find(piece, start)
    assert i >= 0, (piece[:60], text[:80])
    return {"start": i, "end": i + len(piece)}


def family(r) -> str:
    s = r["source"]
    key = next(s[k] for k in ("topic", "example_id", "id", "qid", "page", "response_id", "table_id") if s.get(k))
    return f"{s['dataset'].split(' ')[0]}:{key}"


recs = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
for r in recs:
    ans, rid = r["generated_answer"], r["id"]
    anah_units = {}
    if r["source"]["dataset"] == "ANAH":
        anah_units = {u["text"]: u for u in picks[r["type_id"]]["units"]}
    cursor, errors = 0, []
    for k, c in enumerate(r["gold_claims"]):
        span = find(ans, CLAIM_IN_ANSWER.get((rid, k), c["claim_text"]), cursor)
        cursor = span["end"]
        c["answer_spans"] = [span]
        c["evidence_sets"] = [c["citations"]] if c["citations"] else []
        if c["verdict"] == "SUPPORTED":
            continue
        part = ERROR_PART.get((rid, k))
        u = anah_units.get(c["claim_text"])
        if part is None and u and u.get("corr") and u["corr"][0].strip() and u["corr"][0].strip() in c["claim_text"]:
            part = u["corr"][0].strip()
        err = find(ans, part, span["start"]) if part else dict(span)
        assert span["start"] <= err["start"] and err["end"] <= span["end"], (rid, k)
        errors.append({**err, "claim_index": k,
                       "basis": "ANAH correction" if (u and part and (rid, k) not in ERROR_PART) else
                                ("changed part" if (rid, k) in ERROR_PART else "whole claim")})
    r["gold_error_spans"] = errors
    nc_cursor = 0
    r["nonclaim_spans"] = []
    for t in r.get("gold_nonclaims", []):
        sp = find(ans, t, nc_cursor)
        nc_cursor = sp["end"]
        r["nonclaim_spans"].append(sp)
    r["source_hash"] = hashlib.sha256(json.dumps(r["context_passages"], ensure_ascii=False).encode("utf-8")).hexdigest()
    r["family_id"] = family(r)

OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
by_basis = {}
for r in recs:
    for e in r["gold_error_spans"]:
        by_basis[e["basis"]] = by_basis.get(e["basis"], 0) + 1
print(f"enriched {len(recs)} records; error spans by basis: {by_basis}; "
      f"families: {len({r['family_id'] for r in recs})}")
