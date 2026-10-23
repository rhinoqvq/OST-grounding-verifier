"""Write ost30.jsonl from the ANAH picks (pick_ost30_anah.py output), gold included.

Gold claims are ANAH's human-checked sentence labels (None -> SUPPORTED, Contradictory ->
CONTRADICTED, Unverifiable -> UNSUPPORTED); "No Fact" sentences go to gold_nonclaims.
Citations are ANAH's reference fragments repaired to verbatim substrings of the documents.
Rewrites ost30.jsonl; run add_external_ost30.py afterwards for 06/14/16/18.

Usage: build_ost30_anah.py PICKS.json OUT_DIR
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_ost30 import check  # noqa: E402

PICKS = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
OUT = Path(sys.argv[2])

CATEGORY = {
    "01": "fully_supported", "02": "subtle_distortion", "03": "extrinsic_hallucination",
    "04": "subtle_distortion", "05": "no_claims", "07": "subtle_distortion", "08": "fully_supported",
    "09": "subtle_distortion", "10": "extrinsic_hallucination", "11": "subtle_distortion",
    "12": "fully_supported", "13": "fully_supported", "15": "extrinsic_hallucination",
    "17": "extrinsic_hallucination", "19": "multi_passage", "20": "extrinsic_hallucination",
    "21": "extrinsic_hallucination", "22": "subtle_distortion", "23": "fully_supported",
    "24": "subtle_distortion", "25": "extrinsic_hallucination", "26": "fully_supported",
    "27": "fully_supported", "28": "subtle_distortion", "29": "fully_supported", "30": "subtle_distortion",
}
FIT = {
    "01": ("符合", "全句无幻觉（5 句，讲索布拉昂战役的后续），答案实词在原文里出现的比例在最低的四分之一"),
    "02": ("符合", "回答说 1864 年 10 月 27 日，原文是 1865 年 10 月 22 日，年份差一"),
    "03": ("符合", "InternLM 不看文档作答，加了「萨特马尔大拉比」「萨特马尔哈西德派创始人」「反锡安主义」等现实中为真、原文没有的内容；最后一句 ANAH 标成「非事实」，标签存疑"),
    "04": ("符合", "一句有依据、一句矛盾（防线起止点）、两句无法核实"),
    "07": ("符合", "回答说卡夫卡的作品「生前没有发表」，原文是「生前发表的很少」（有几部发表过），否定说过了头"),
    "08": ("符合", "「多达 500 部」「约三分之一」「约 300 万词」都是原文给的量词"),
    "09": ("符合", "发掘开始的年份 1968 写成了 1928"),
    "11": ("符合", "把罗曼·哈贝尔夫妇的事挂到了他父亲约翰·哈贝尔身上"),
    "12": ("符合", "第二句起用 He 指代希罗多德"),
    "13": ("部分符合", "原文有 known as 一类别名，回答用了其中的称呼，但不需要跨段对应，匹配较浅"),
    "15": ("符合", "原文有 1936 年精神崩溃、1937 年退休，回答说「部分因为健康原因退休」，把先后说成了因果"),
    "17": ("符合", "只有一句，列了柏拉图著作涉及的学科，主题相关但原文没有这样列"),
    "19": ("符合", "两句的引文分别来自两篇文档"),
    "21": ("部分符合", "长句带 who 从句（查士丁尼二世逃到特尔维尔的宫廷，后者答应帮他复位），但 ANAH 是整句判无法核实，不是只错一部分"),
    "23": ("符合", "回答以 According to the Book of Deuteronomy 开头，把说法归属到《申命记》"),
    "26": ("符合", "原文约 3.2 万字符（基辅史），全句无幻觉"),
    "27": ("符合", "原文超过 60 句、约 1.7 万字符，全句无幻觉"),
    "28": ("符合", "原文约 11.3 万字符，一句矛盾、两句无法核实"),
    "29": ("符合", "和 30 题同一个问题，GPT-3.5 看着文档作答，全句无幻觉"),
    "30": ("符合", "和 29 题同一个问题，InternLM 不看文档作答，一句矛盾、一句无法核实"),
}
VERDICT = {"None": "SUPPORTED", "Contradictory": "CONTRADICTED", "Unverifiable": "UNSUPPORTED"}
MODEL = {"gpt35_with_docs": "GPT-3.5（看着文档作答）", "internlm_closed_book": "InternLM（不看文档作答）"}


def reasoning(u) -> str:
    if u["label"] == "None":
        return "ANAH: no hallucination; supported by the cited passage."
    if u["label"] == "Unverifiable":
        return "ANAH: unverifiable from the documents."
    x, y = u["corr"] or ("", "")
    return f'ANAH: contradicts the cited passage (correction: "{x.strip()}" -> "{y.strip()}").'


# Human review (10-22) under the labelling rules in docs/appendix/ost30-extraction.zh-CN.md section 8: the gold follows the assignment's
# definitions, so where ANAH's label is looser it is changed here and the change is recorded in notes.label_changes.
REVIEW = {
    "01": {"note": "10-22 人工复核通过；第 3 条的 As a result 按口径 4 视为连接词，保持 S"},
    "03": {"note": "10-22 人工复核通过",
           "promote_nonclaim": {
               "text": "He was a very influential figure in the Orthodox Jewish community in Hungary before the Holocaust.",
               "verdict": "SUPPORTED",
               "citations": [{"index": 0, "quote": "Teitelbaum, a major rabbinic leader in pre-Holocaust Hungary"}],
               "reasoning": "The passage calls Teitelbaum a major rabbinic leader in pre-Holocaust Hungary.",
               "change": "ANAH 标「非事实」→ S：这是事实陈述（口径 2），原文称他是大屠杀前匈牙利的重要拉比领袖"}},
    "13": {"note": "10-22 人工复核通过",
           "add_citations": {1: [{"index": 0, "quote": 'The Emancipation Proclamation having stated that freedmen would be '
                                                       '"received into the armed service of the United States," enlisting '
                                                       'these freedmen became official policy.'}]},
           "changes": ["第 2 条判定不变（S），补一条引文支撑后半句「授权军队招募非裔美国人当兵」（口径 3）；ANAH 原引文只支撑解放奴隶"]},
    "07": {"note": "10-22 人工复核通过（原先的 Wilberforce 题 ANAH 标错，换成种子顺序下一条）",
           "add_citations": {1: [{"index": 0, "quote": "Brod ignored these instructions, and had much of his work published"}]},
           "changes": ["第 2 条保持 ANAH 的 S：按宽松读法，his works 指他的大部分作品，原文说布罗德在他死后出版了大部分作品、"
                       "他死后才成名；补一条布罗德出版的引文（口径 3）"]},
    "15": {"note": "10-22 复核通过，按 ANAH 原标保持。已知两处：第 3 条判 C 可宽可严（原文没有直接反驳「觉得该换人领导」，"
                   "但回答把后人评价错当成退休原因）；第 4 条判 S 对，但 ANAH 的两条引文不支持该句，原文 6% 处有原话"
                   "「Baldwin retired in 1937 and was succeeded by Neville Chamberlain.」，评分引文时要注意这条。"
                   "第一次跑分后按口径 3 统一筛查全部 S 金标，改了第 2 条",
           "set_verdict": {1: ("UNSUPPORTED", "The passage supports the 1936 nervous breakdown, but it mentions his arthritis "
                                             "only after his wife's death in 1945, years after he retired, and never says it "
                                             "made his duties difficult.")},
           "changes": ["第 2 条 ANAH 标「无幻觉」→ U：原文只撑住 1936 年精神崩溃；关节炎在原文里是 1945 年以后的事（他 1937 年已退休），"
                       "「难以履职」原文没有（口径 3）"]},
    "04": {"note": "10-22 复核通过；第一次跑分后按口径 3 统一筛查全部 S 金标，改了第 1 条；第 4 条差别不大，按 ANAH 不改"
                   "（引文只撑住「属于防线网络」，「拖住德军、争取时间」在原文另一句，「遍布全国」原文没有）",
           "set_verdict": {0: ("UNSUPPORTED", "The passage says the line was built to meet the threat of a Nazi invasion "
                                             "during World War II and formed the principal defensive front for North East "
                                             "England, but it never says it was an anti-tank line.")},
           "changes": ["第 1 条 ANAH 标「无幻觉」→ U：原文没说是反坦克防线（tank 出现 0 次），引文撑不起整句（口径 3）"]},
    "21": {"note": "10-22 人工复核通过",
           "replace_citations": {0: [{"index": 0, "quote": "The Byzantines were skilled at using diplomacy as a weapon of war."}]},
           "set_verdict": {7: ("UNSUPPORTED", "The passage only says Byzantium preferred diplomacy because war was costly; "
                                             "it never says these subsidies let it avoid a war with the Bulgarians, and the "
                                             "Tervel tribute it refers to is not in the passage.")},
           "changes": ["第 1 条判定不变（S），引文换成更直接的「The Byzantines were skilled at using diplomacy as a weapon of war.」",
                       "第 8 条 ANAH 标「无幻觉」→ U：「这样的补贴」指原文没有的特尔维尔进贡，「避免与保加利亚人开战」原文没写；"
                       "ANAH 引的两句只是泛泛说拜占庭偏好外交，撑不起整句（口径 1）"]},
}

for _tid, _why in {'02': '无问题', '09': '无问题', '11': '无问题', '26': '无问题', '27': '无问题', '29': '无问题', '30': '无问题', '08': '差别不大，按 ANAH 不改：第 2 条回答说「约三分之一存世」，原文是「不到三分之一」', '12': '差别不大，按 ANAH 不改：第 2 条「可能在节庆上朗诵」原文另一句有，引文没带上', '17': '差别不大，按 ANAH 不改：列出的学科原文大多讲过，「心理学」很弱；整句是并列断言，保持 U', '23': '差别不大，按 ANAH 不改：第 1 条「据《申命记》」原文没明说出处，但讲的就是这段经文叙事'}.items():
    REVIEW.setdefault(_tid, {"note": f"10-22 复核通过（{_why}）"})


def apply_review(rec) -> None:
    r = REVIEW.get(rec["type_id"])
    if not r:
        return
    changes = list(r.get("changes", []))
    for k, cites in r.get("add_citations", {}).items():
        rec["gold_claims"][k]["citations"] += cites
    for k, cites in r.get("replace_citations", {}).items():
        rec["gold_claims"][k]["citations"] = cites
    for k, (verdict, why, *cites) in r.get("set_verdict", {}).items():
        c = rec["gold_claims"][k]
        c.update(verdict=verdict, reasoning=why,
                 citations=[] if verdict == "UNSUPPORTED" else (cites[0] if cites else c["citations"]))
    p = r.get("promote_nonclaim")
    if p:
        rec["gold_nonclaims"].remove(p["text"])
        rec["gold_claims"].append({"claim_text": p["text"], "verdict": p["verdict"], "reasoning": p["reasoning"],
                                   "citations": p["citations"]})
        rec["gold_claims"].sort(key=lambda c: rec["generated_answer"].index(c["claim_text"]))
        changes.append(p["change"])
    rec["gold_status"] = "reviewed"
    rec["notes"]["review"] = r["note"]
    if changes:
        rec["notes"]["label_changes"] = changes


records = []
for o in PICKS:
    a = o["pick"]
    if not a:
        continue
    tid = o["type_id"]
    gold, nonclaims = [], []
    for u in a["units"]:
        if u["label"] == "NF":
            nonclaims.append(u["text"])
            continue
        cites = []
        if u["label"] in ("None", "Contradictory"):
            seen = set()
            for i, q in u["citations"]:
                if (i, q) not in seen:
                    seen.add((i, q))
                    cites.append({"index": i, "quote": q})
        gold.append({"claim_text": u["text"], "verdict": VERDICT[u["label"]], "reasoning": reasoning(u),
                     "citations": cites})
    rec = {
        "id": f"OST30-{tid}", "category": CATEGORY[tid], "type_id": tid, "type": o["type"],
        "context_passages": a["documents"], "generated_answer": a["answer"],
        "gold_claims": gold, "gold_nonclaims": nonclaims, "gold_status": "labeled",
        "source": {"dataset": "ANAH", "hf": "opencompass/anah", "split": "train", "license": "Apache-2.0",
                   "topic": a["topic"], "question": a["question"], "answer_model": MODEL[a["source"]], "key": a["key"],
                   **({"dropped_documents": a["dropped_docs"],
                       "transform": "去掉了 ANAH 原文里的 Encyclopaedia Britannica 文档（版权不开放），只留维基百科文档；"
                                    "所有 S / C 引文都在留下的文档里"} if a.get("dropped_docs") else {})},
        "notes": {"fit": FIT[tid][0], "fit_note": FIT[tid][1],
                  "gold_origin": "ANAH 人工核过的句子级标签；引文是 ANAH 的参考片段修成的原文逐字子串"},
    }
    apply_review(rec)
    errs = check(rec)
    assert not errs, (rec["id"], errs)
    records.append(rec)

(OUT / "ost30.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
print(f"wrote {len(records)} ANAH records")
