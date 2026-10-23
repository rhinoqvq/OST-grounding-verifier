"""Pick one ANAH (English) answer per OST30 question type (docs/appendix/ost30-extraction.zh-CN.md).

ANAH annotates every answer sentence as None / Contradictory / Unverifiable / No Fact, with
a reference fragment and, for hallucinations, a correction '"X" to "Y"'. Per type: a
label-aware filter builds a candidate pool, candidates are ordered by a seeded RNG
("20261022-<type>"), and the first candidate that is not skipped, whose topic is unused and
whose S/C references all repair to verbatim substrings of the documents, is picked.

Usage: pick_ost30_anah.py OUT.json [SKIP.json]
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "var" / "anah" / "anah_all.json"
SEED = "20261022"
VERDICT = {"None": "SUPPORTED", "Contradictory": "CONTRADICTED", "Unverifiable": "UNSUPPORTED"}

WORD = re.compile(r"[A-Za-z0-9']+")
STOP = set("""a an the of to in on at for and or but is are was were be been being it its this that these those
with as by from has have had not no he she they them his her their there which who whom what when where
will would can could may might should do does did so than then also into about over after before""".split())


def words(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s)]


def content(s: str) -> set[str]:
    return {w for w in words(s) if w not in STOP and len(w) > 2}


def split_sentences(text: str) -> list[str]:
    """ANAH's sentence units: 92% of English answers align with this split."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------- citation repair ----------

def _find(piece: str, docs: list[str]) -> tuple[int, str] | None:
    piece = piece.strip().strip('"“”\'').strip()
    if len(piece) < 15:
        return None
    for variant in (piece, piece.rstrip(". ")):
        for i, d in enumerate(docs):
            if variant and variant in d:
                return i, variant
    spaced = re.sub(r"([.!?;:,)])(?=[A-Z(])", r"\1 ", piece)
    toks = spaced.rstrip(". ").split()
    if not toks:
        return None
    pat = re.compile(r"\s*".join(re.escape(t) for t in toks))
    for i, d in enumerate(docs):
        m = pat.search(d)
        if m:
            return i, d[m.start():m.end()]
    return None


def repair(fragment: str, docs: list[str]) -> list[tuple[int, str]] | None:
    """Verbatim (index, quote) pieces for one reference fragment, or None if any piece is missing."""
    hit = _find(fragment, docs)
    if hit:
        return [hit]
    parts = [p for p in re.split(r"\s*(?:\.\.\.|…)\s*|(?<=[.!?])\s*(?=[A-Z])", fragment) if p.strip()]
    found = [_find(p, docs) for p in parts if len(p.strip()) >= 15]
    if found and all(found):
        return found
    return None


def fragments(ref: str) -> list[str]:
    ref = ref.strip()
    if ref in ("", "None"):
        return []
    return [f.strip() for f in re.split(r"<SEP>|参考\d+[:：]", ref) if f.strip()]


# ---------- load answers ----------

def parse(ann: str) -> dict:
    lab = "NF" if ann.startswith("<No Fact>") else (re.match(r"<Hallucination>\s*(\w+)", ann).group(1))
    ref = re.search(r"<Reference>\s*(.*?)(?:\n<Correction>|$)", ann, re.S)
    cor = re.search(r'<Correction>\s*"(.*)"\s*to\s*"(.*)"', ann, re.S)
    return {"label": lab, "ref": ref.group(1) if ref else "", "corr": (cor.group(1), cor.group(2)) if cor else None}


answers = []
for t, row in enumerate(json.loads(DATA.read_text(encoding="utf-8"))):
    if row["language"] != "en":
        continue
    docs = row["documents"]
    for source, ak, nk in (("gpt35_with_docs", "GPT3.5_answers_D", "human_GPT3.5_answers_D_ann"),
                           ("internlm_closed_book", "InternLM_answers", "human_InternLM_answers_ann")):
        for qi, (ans, ann) in enumerate(zip(row[ak], row[nk])):
            sents = split_sentences(ans)
            if len(sents) != len(ann) or not ann:
                continue
            units = []
            ok = True
            for s, a in zip(sents, ann):
                u = {"text": s, **parse(a)}
                if u["label"] in ("None", "Contradictory"):
                    cites = []
                    for f in fragments(u["ref"]):
                        got = repair(f, docs)
                        if got is None:
                            ok = False
                            break
                        cites += got
                    if not cites:
                        ok = False
                    u["citations"] = cites
                units.append(u)
            answers.append({
                "key": f"{row['name']}#{source}#{qi}", "topic": row["name"], "topic_row": t, "source": source,
                "question": row["selected_questions"][qi], "answer": ans, "documents": docs,
                "doc_chars": sum(len(d) for d in docs), "units": units, "citations_ok": ok,
                "labels": {u["label"] for u in units},
            })


def is_wiki(doc: str) -> bool:
    """ANAH documents are Wikipedia articles ("# Title" first line) or Encyclopaedia Britannica entries; only the
    former are openly licensed (CC BY-SA)."""
    return doc.lstrip().startswith("# ")


def open_version(a):
    """The answer with Britannica documents removed and citations re-indexed, or None if an S/C citation would be
    lost. Dropping a document cannot add support, so U labels stay valid."""
    keep = [i for i, d in enumerate(a["documents"]) if is_wiki(d)]
    if not keep or any(i not in keep for u in a["units"] for i, _ in u.get("citations", [])):
        return None
    remap = {old: new for new, old in enumerate(keep)}
    units = [{**u, "citations": [(remap[i], q) for i, q in u.get("citations", [])]} for u in a["units"]]
    docs = [a["documents"][i] for i in keep]
    return {**a, "documents": docs, "doc_chars": sum(len(d) for d in docs), "units": units,
            "dropped_docs": [i for i in range(len(a["documents"])) if i not in keep]}


def still_fits(tid, a) -> bool:
    """Type conditions that depend on the documents, rechecked after Britannica documents are dropped."""
    if tid == "19":
        return len({i for u in a["units"] for i, _ in u.get("citations", [])}) >= 2
    if tid == "26":
        return 20000 <= a["doc_chars"] <= 40000
    if tid == "27":
        return sum(len(split_sentences(d)) for d in a["documents"]) > 60
    if tid == "28":
        return a["doc_chars"] >= 100000
    return True


def has(a, lab):
    return lab in a["labels"]


def units(a, lab):
    return [u for u in a["units"] if u["label"] == lab]


clean = [a for a in answers if a["labels"] <= {"None"} and len(a["units"]) >= 2]
_ov: dict[str, float] = {}


def overlap(a) -> float:
    if a["key"] not in _ov:
        w = content(a["answer"])
        _ov[a["key"]] = len(w & content(" ".join(a["documents"]))) / max(1, len(w))
    return _ov[a["key"]]


def lowest(pool, share):
    if not pool:
        return []
    cut = sorted(map(overlap, pool))[int(len(pool) * share)]
    return [a for a in pool if overlap(a) <= cut]


NUM = re.compile(r"\d+(?:\.\d+)?")
NEG = re.compile(r"\b(not|no|never|none|without|neither|nor|cannot|failed to|refused)\b|n't\b", re.I)
TIME = re.compile(r"\b(1\d\d\d|20\d\d)\b|\b(January|February|March|April|May|June|July|August|September|October|"
                  r"November|December)\b")
CAUSE = re.compile(r"\b(because|due to|led to|leading to|caused|causing|resulted in|as a result|thanks to|"
                   r"therefore|which led)\b", re.I)
STRONG = re.compile(r"\b(must|always|never|required|requires|guarantee[sd]?|mandatory|only)\b", re.I)
WEAK = re.compile(r"\b(may|can|could|might|sometimes|often|optional)\b", re.I)
QUANT = re.compile(r"\b(at least|at most|up to|more than|less than|fewer than|over|nearly|approximately|about|"
                   r"around|some)\s+[\d,]+", re.I)
PRONOUN = re.compile(r"^(It|They|This|These|He|She|His|Her|Their|Its|Such)\b")
ATTRIB = re.compile(r"\b(according to|reported|stated|said|wrote|described|claimed|historians|described as)\b", re.I)
ALIAS = re.compile(r"\b(?:also known as|known as|nicknamed|called|a\.k\.a\.?)\s+[\"'“]?([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,3})")
QUALIFIER = re.compile(r"\b(only|all|every|first|last|largest|most|sole|entire|never|always|exactly)\b", re.I)
CAPS = re.compile(r"\b[A-Z][a-z]{2,}\b")


def corr(u):
    return u["corr"] or ("", "")


def off_by_one(u) -> bool:
    x, y = corr(u)
    xs, ys = {float(n) for n in NUM.findall(x.replace(",", ""))}, {float(n) for n in NUM.findall(y.replace(",", ""))}
    return bool(xs) and not (xs & ys) and any(abs(a - b) == 1 for a in xs for b in ys)


def neg_flip(u) -> bool:
    x, y = corr(u)
    return bool(x and y) and bool(NEG.search(x)) != bool(NEG.search(y))


def time_swap(u) -> bool:
    x, y = corr(u)
    return bool(TIME.search(x) and TIME.search(y)) and set(TIME.findall(x)) != set(TIME.findall(y))


def name_swap(a, u) -> bool:
    x, y = corr(u)
    nx, ny = set(CAPS.findall(x)), set(CAPS.findall(y))
    rest = content(x) - {w.lower() for w in nx}, content(y) - {w.lower() for w in ny}
    docs = " ".join(a["documents"])
    return bool(nx and ny and nx != ny and (nx - ny) and all(n in docs for n in nx - ny)
                and len(rest[0] & rest[1]) >= max(1, len(rest[0] | rest[1]) // 3))


def qualifier_only(u) -> bool:
    x, y = corr(u)
    wx, wy = words(x), words(y)
    diff = set(wx) ^ set(wy)
    return 0 < len(diff) <= 3 and any(QUALIFIER.fullmatch(w) for w in diff)


def ref_docs(a) -> set[int]:
    return {i for u in a["units"] for i, _ in u.get("citations", [])}


def prev_condition(a, u) -> bool:
    texts = [v["text"] for v in a["units"]]
    i = texts.index(u["text"])
    return i > 0 and bool(PRONOUN.match(u["text"])) and bool(
        re.search(r"\b(if|when|unless|only|provided|while|during)\b", texts[i - 1], re.I)) and len(
        content(texts[i - 1]) & content(u["text"])) >= 2


TYPES = [
    ("01", "直接支持与同义改写", "全句无幻觉（3 句以上），答案实词在原文出现比例最低的四分之一",
     lambda: lowest([a for a in clean if len(a["units"]) >= 3], 0.25)),
    ("02", "明确反证：数字差一", "有「矛盾」句，订正前后的数字正好差 1",
     lambda: [a for a in answers if any(off_by_one(u) for u in units(a, "Contradictory"))]),
    ("03", "无证据新增细节", "只有「无法核实」、没有「矛盾」，且来自不看文档作答的 InternLM（凭外部知识写的内容）",
     lambda: [a for a in answers if has(a, "Unverifiable") and not has(a, "Contradictory")
              and a["source"] == "internlm_closed_book"]),
    ("04", "多事实混合判定", "同一回答里同时有无幻觉、矛盾、无法核实三种句子",
     lambda: [a for a in answers if {"None", "Contradictory", "Unverifiable"} <= a["labels"]]),
    ("05", "纯非事实回答", "回答里每一句都是「非事实」",
     lambda: [a for a in answers if a["labels"] == {"NF"}]),
    ("07", "否定反转", "有「矛盾」句，订正前后一边有否定词、一边没有",
     lambda: [a for a in answers if any(neg_flip(u) for u in units(a, "Contradictory"))]),
    ("08", "量词（上下界）", "全句无幻觉，含 at least / more than / about 等量词且后接数字",
     lambda: [a for a in clean if QUANT.search(a["answer"])]),
    ("09", "版本与时间条件", "有「矛盾」句，订正前后的年份或月份不同",
     lambda: [a for a in answers if any(time_swap(u) for u in units(a, "Contradictory"))]),
    ("10", "模态加强", "「无法核实」句里有 must / always / only 等强模态词，且原文有一句和它实词重合 3 个以上、用的是 may / can",
     lambda: [a for a in answers if any(
         STRONG.search(u["text"]) and any(WEAK.search(s) and len(content(s) & content(u["text"])) >= 3
                                          for d in a["documents"] for s in split_sentences(d))
         for u in units(a, "Unverifiable"))]),
    ("11", "相似实体的属性错绑", "有「矛盾」句，订正只换了人名 / 地名，且被换掉的名字在原文出现过",
     lambda: [a for a in answers if any(name_swap(a, u) for u in units(a, "Contradictory"))]),
    ("12", "跨句代词与指代", "全句无幻觉，第二句起有以 It / They / He / She / This 开头的句子",
     lambda: [a for a in clean if any(PRONOUN.match(u["text"]) for u in a["units"][1:])]),
    ("13", "别名映射后支持", "全句无幻觉，原文有 also known as / known as / nicknamed 等别名，回答里用了这个别名",
     lambda: [a for a in clean if any(m.group(1) in a["answer"] for d in a["documents"] for m in ALIAS.finditer(d))]),
    ("15", "先后顺序不蕴含因果", "「无法核实」句里有 because / led to / due to 等因果词",
     lambda: [a for a in answers if any(CAUSE.search(u["text"]) for u in units(a, "Unverifiable"))]),
    ("17", "主题相关但证据不足", "恰好一句「无法核实」、没有「矛盾」，该句和原文的实词重合在一半以上",
     lambda: [a for a in answers if len(units(a, "Unverifiable")) == 1 and not has(a, "Contradictory") and (
         lambda u: len(content(u["text"]) & content(" ".join(a["documents"]))) >= 0.5 * max(1, len(content(u["text"]))))(
         units(a, "Unverifiable")[0])]),
    ("19", "跨段综合（干净）", "全句无幻觉，引文分别来自两篇以上不同的文档",
     lambda: [a for a in clean if len(ref_docs(a)) >= 2]),
    ("21", "嵌套复合句", "有一句 35 词以上、带从句连接词的长句被标为矛盾或无法核实",
     lambda: [a for a in answers if any(
         len(words(u["text"])) >= 35 and re.search(r"\b(if|when|while|although|which|who|whereas)\b", u["text"], re.I)
         for u in a["units"] if u["label"] in ("Contradictory", "Unverifiable"))]),
    ("22", "主句正确、限定词错误", "有「矛盾」句，订正只改了 1–3 个词，其中有 only / all / first / largest 等限定词",
     lambda: [a for a in answers if any(qualifier_only(u) for u in units(a, "Contradictory"))]),
    ("23", "元话语与归属", "全句无幻觉，含 according to / reported / said / described 等归属说法",
     lambda: [a for a in clean if ATTRIB.search(a["answer"])]),
    ("25", "跨句条件继承", "「无法核实」句以 This / It / They 开头，前一句带 if / when / unless / only 等条件，两句重合两个以上实词",
     lambda: [a for a in answers if any(prev_condition(a, u) for u in units(a, "Unverifiable"))]),
    ("26", "长文档（干净）", "全句无幻觉，原文 2–4 万字符",
     lambda: [a for a in clean if 20000 <= a["doc_chars"] <= 40000]),
    ("27", "超过 60 句：远距离改写", "全句无幻觉，原文超过 60 句，答案实词重合率在最低的一半",
     lambda: lowest([a for a in clean if sum(len(split_sentences(d)) for d in a["documents"]) > 60], 0.5)),
    ("28", "超长文档（10–20 万字符）", "原文 10 万字符以上，回答里同时有矛盾和无法核实",
     lambda: [a for a in answers if a["doc_chars"] >= 100000 and {"Contradictory", "Unverifiable"} <= a["labels"]]),
]

SKIP: dict[str, dict[str, str]] = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")) if len(sys.argv) > 2 else {}
used_topics: set[str] = set()
out = []
for tid, name, rule, fn in TYPES:
    pool = sorted(fn(), key=lambda a: a["key"])
    rng = random.Random(f"{SEED}-{tid}")
    rng.shuffle(pool)
    skip = SKIP.get(tid, {})
    auto, lic = [], []
    pick = None
    for a in ([] if "*" in skip else pool):
        if a["key"] in skip:
            continue
        if a["topic"] in used_topics:
            continue
        if not a["citations_ok"]:
            auto.append(a["key"])
            continue
        a_open = open_version(a)
        if a_open is None or not still_fits(tid, a_open):  # needs Britannica text, which cannot be published
            lic.append(a["key"])
            continue
        pick = a_open
        break
    if pick:
        used_topics.add(pick["topic"])
    out.append({"type_id": tid, "type": name, "rule": rule, "n_candidates": len(pool),
                "n_candidates_citations_ok": sum(a["citations_ok"] for a in pool),
                "skipped": skip, "skipped_citations": auto, "skipped_license": lic, "pick": pick,
                "seeded_head": [a["key"] for a in pool[:12]]})

# 29 / 30: same topic and question, GPT-3.5 (with documents) clean + InternLM (closed book) with C and U
by_q = defaultdict(dict)
for a in answers:
    by_q[(a["topic"], a["question"])][a["source"]] = a
pairs = sorted(
    (k for k, v in by_q.items() if k[0] not in used_topics and "gpt35_with_docs" in v and "internlm_closed_book" in v
     and open_version(v["gpt35_with_docs"]) and open_version(v["internlm_closed_book"])
     and v["gpt35_with_docs"]["labels"] <= {"None"} and v["gpt35_with_docs"]["citations_ok"]
     and {"Contradictory", "Unverifiable"} <= v["internlm_closed_book"]["labels"]
     and v["internlm_closed_book"]["citations_ok"]))
rng = random.Random(f"{SEED}-29")
pair = rng.choice(pairs) if pairs else None
for tid, name, src in (("29", "共享原文第一次（干净）", "gpt35_with_docs"), ("30", "共享原文第二次（含错）", "internlm_closed_book")):
    out.append({"type_id": tid, "type": name,
                "rule": "同一话题同一问题：GPT-3.5（看文档作答）全句无幻觉，InternLM（不看文档）同时有矛盾和无法核实；随机挑一个问题",
                "n_candidates": len(pairs), "n_candidates_citations_ok": len(pairs), "skipped": {}, "skipped_citations": [], "skipped_license": [],
                "pick": open_version(by_q[pair][src]) if pair else None, "seeded_head": []})

Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=sorted), encoding="utf-8")
print(f"answers aligned: {len(answers)}, citations repairable: {sum(a['citations_ok'] for a in answers)}")
for o in out:
    p = o["pick"]
    print(o["type_id"], o["type"], f"| pool {o['n_candidates']} (citations ok {o['n_candidates_citations_ok']})",
          "|", "—" if not p else f"{p['key']} | {p['doc_chars']}c | {[u['label'][0] for u in p['units']]}")
