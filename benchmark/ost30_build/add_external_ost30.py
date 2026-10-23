"""Fill OST30 gap types from other public datasets (docs/appendix/ost30-extraction.zh-CN.md section 11).

Reads the selected source rows from ost30_external_rows.json next to this script (fetched through the HF
dataset viewer) and writes/replaces records 06, 14, 16, 18 in ost30.jsonl. Gold verdicts
come from each dataset's own label; citations are the premises / supporting paragraph
sentences that label rests on. Run after build_ost30_anah.py, which rewrites ost30.jsonl.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_ost30 import check  # noqa: E402

ROWS = json.loads((Path(__file__).parent / "ost30_external_rows.json").read_text(encoding="utf-8"))
OUT = Path(sys.argv[1])

folio = {r["example_id"]: r for r in ROWS["folio"]}
musique = {r["id"]: r for r in ROWS["musique"]}
faitheval = {r["qid"]: r for r in ROWS["faitheval"]}


def premises(r):
    return [p for p in r["premises"].split("\n") if p.strip()]


records = []

# 05: a RAGTruth refusal (train split, quality "good", no hallucination labels) -> no claims
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ragtruth"))
from tasks import DATA_DIR, split_passages  # noqa: E402

rt_sources = {json.loads(l)["source_id"]: json.loads(l)
              for l in (DATA_DIR / "source_info.jsonl").read_text(encoding="utf-8").splitlines()}
rt = next(json.loads(l) for l in (DATA_DIR / "response.jsonl").read_text(encoding="utf-8").splitlines()
          if json.loads(l)["id"] == "16298")
rt_src = rt_sources[rt["source_id"]]
records.append({
    "id": "OST30-05", "category": "no_claims", "type_id": "05", "type": "纯非事实回答",
    "context_passages": split_passages(rt_src["source_info"]["passages"]), "generated_answer": rt["response"],
    "gold_claims": [], "gold_nonclaims": [rt["response"]], "gold_status": "labeled",
    "source": {"dataset": "RAGTruth", "split": "train", "task": "QA", "response_id": rt["id"],
               "source_id": rt["source_id"], "generator": rt["model"], "question": rt_src["source_info"]["question"],
               "quality": rt["quality"], "license": "MIT"},
    "notes": {"fit": "符合", "fit_note": "回答只有一句「无法根据原文回答」，没有可核对的事实",
              "gold_origin": "RAGTruth 标为没有幻觉、质量 good；回答是拒答，按无事实处理，金标为空"},
})

r = folio[548]
records.append({
    "id": "OST30-06", "category": "extrinsic_hallucination", "type_id": "06", "type": "If / only if 方向",
    "context_passages": premises(r), "generated_answer": r["conclusion"],
    "gold_claims": [{
        "claim_text": "Mike has an F1 visa.", "verdict": "UNSUPPORTED",
        "reasoning": "An F1 holder needs CPT or OPT to work, but Mike needing CPT does not imply he holds an F1 "
                     "visa; he could hold a J1. FOLIO labels the conclusion Uncertain.",
        "citations": []}],
    "gold_status": "reviewed",
    "source": {"dataset": "FOLIO", "hf": "tasksource/folio", "split": "train", "example_id": 548,
               "story_id": r["story_id"], "label": r["label"], "license": "CC BY-SA 4.0",
               "transform": "每条前提作为一段原文，结论原样作为回答"},
    "notes": {"fit": "符合", "fit_note": "前提只说 F1 签证要申请 CPT 才能工作，回答由「Mike 要申请 CPT」反推他有 F1，是把必要条件当充分条件",
              "gold_origin": "判定来自 FOLIO 标签 Uncertain → UNSUPPORTED"},
})

r = folio[165]
P = premises(r)
records.append({
    "id": "OST30-18", "category": "multi_passage", "type_id": "18", "type": "跨段排他性反证",
    "context_passages": P, "generated_answer": r["conclusion"],
    "gold_claims": [{
        "claim_text": "Elizabeth is a king.", "verdict": "CONTRADICTED",
        "reasoning": "Elizabeth is a leader of a country for life, such leaders are either a king or a queen, and "
                     "Elizabeth is a queen, so she is not a king. FOLIO labels the conclusion False.",
        "citations": [{"index": 1, "quote": "Leaders of a country for life are either a king or a queen."},
                      {"index": 4, "quote": "Elizabeth is a queen."},
                      {"index": 5, "quote": "Elizabeth is a leader of a country for life."}]}],
    "gold_status": "reviewed",
    "source": {"dataset": "FOLIO", "hf": "tasksource/folio", "split": "train", "example_id": 165,
               "story_id": r["story_id"], "label": r["label"], "license": "CC BY-SA 4.0",
               "transform": "每条前提作为一段原文，结论原样作为回答"},
    "notes": {"fit": "符合", "fit_note": "第 1 段给排他规则（终身领导人要么是国王要么是女王），第 4、5 段给成员身份，合起来否定回答",
              "gold_origin": "判定来自 FOLIO 标签 False → CONTRADICTED；引文是这个结论依赖的三条前提"},
})

r = musique["2hop__107905_110222"]
paras = [p["paragraph_text"] for p in r["paragraphs"]]
records.append({
    "id": "OST30-14", "category": "multi_passage", "type_id": "14", "type": "多跳但缺关键前提",
    "context_passages": paras,
    "generated_answer": "The creator of A Burial at Ornans is Gustave Courbet, who was born on 10 June 1819.",
    "gold_claims": [
        {"claim_text": "The creator of A Burial at Ornans is Gustave Courbet.", "verdict": "SUPPORTED",
         "reasoning": "The supporting paragraph says the painting is by Gustave Courbet.",
         "citations": [{"index": 10, "quote": "A Burial At Ornans (, also known as A Funeral At Ornans) is a painting "
                                              "of 1849–50 by Gustave Courbet"}]},
        {"claim_text": "Gustave Courbet was born on 10 June 1819.", "verdict": "UNSUPPORTED",
         "reasoning": "No paragraph gives Courbet's birth date; the other paragraphs give birth dates of unrelated "
                      "people. MuSiQue marks the question unanswerable.",
         "citations": []}],
    "gold_status": "reviewed",
    "source": {"dataset": "MuSiQue", "hf": "bdsaglam/musique (musique_full_v1.0_dev)", "split": "validation",
               "id": r["id"], "answerable": r["answerable"], "license": "CC BY 4.0",
               "transform": "20 个段落原样作为原文；回答用题目和数据集答案写成一句陈述"},
    "notes": {"fit": "符合", "fit_note": "第一跳（画作的作者是 Courbet）有段落支持，第二跳（Courbet 的生日）的段落被拿掉；其他段落里有别人的生日当干扰",
              "gold_origin": "第二跳 U 来自 MuSiQue 的 unanswerable 标记；第一跳 S 的段落来自 is_supporting 标记，引文句由我们选"},
})

# 19: MuSiQue answerable 2-hop; both hops are stated in the paragraph text (first two candidates in seeded order
# were skipped: "producer of Mistress" and "Bettany's spouse" are not stated in their paragraphs)
r = musique["2hop__781079_9237"]
paras = [p["paragraph_text"] for p in r["paragraphs"]]
hop1 = ("is located in the western part of Rairok on the south side of Majuro Atoll, the capital of the Republic of "
        "the Marshall Islands")
hop2 = ("There is also a small community of Ahmadiyya Muslims based in Majuro, with the first mosque opening in the "
        "capital in September 2012.")
records.append({
    "id": "OST30-19", "category": "multi_passage", "type_id": "19", "type": "跨段综合（干净）",
    "context_passages": paras,
    "generated_answer": "Marshall Islands International Airport is on Majuro Atoll, the capital of the Marshall Islands. "
                        "The first mosque in Majuro opened in September 2012.",
    "gold_claims": [
        {"claim_text": "Marshall Islands International Airport is on Majuro Atoll, the capital of the Marshall Islands.",
         "verdict": "SUPPORTED", "reasoning": "Stated in the airport paragraph.",
         "citations": [{"index": 8, "quote": hop1}]},
        {"claim_text": "The first mosque in Majuro opened in September 2012.", "verdict": "SUPPORTED",
         "reasoning": "Stated in the Marshall Islands paragraph.", "citations": [{"index": 10, "quote": hop2}]}],
    "gold_status": "labeled",
    "source": {"dataset": "MuSiQue", "hf": "bdsaglam/musique (musique_full_v1.0_dev)", "split": "validation",
               "id": r["id"], "answerable": r["answerable"], "license": "CC BY 4.0",
               "transform": "20 个段落原样作为原文；回答按两跳的子问题和答案写成两句陈述"},
    "notes": {"fit": "符合", "fit_note": "第一句靠机场那段（在马朱罗环礁），第二句靠马绍尔群岛那段（马朱罗第一座清真寺 2012 年 9 月开放），两段各支持一句；其余 18 段是干扰",
              "gold_origin": "MuSiQue 标的两个支持段落；两句都在段落里写明，引文由我们选",
              "skipped": {"2hop__475808_71302": "第一跳「Mistress 的制片人」在段落里只写了主演；第二跳段落讲的是《教父 2》",
                          "2hop__72487_510545": "第二跳「Bettany 的配偶」段落里只写了两人同片演出"}},
})

# 16: FaithEval inconsistent subset, Wikipedia-based subsets only (SQuAD, HotpotQA, NQ, DROP), seeded order.
# The first pick (cab69864, NewsQA) was dropped because its context is CNN news text. Skipped on the way:
# 2f404b56 (HTML table markup), 753fd169 (modified document still keeps the old answer, no conflict),
# 1b0eab0d ([PAR]/[TLE]/[SEP] markup), 8cb37ca2 (conflict only visible by adding up the score).
qid = next(k for k in faitheval if k.startswith("10853963"))
r = faitheval[qid]
docs = [d.strip() for d in r["context"].split("Document:") if d.strip()]
claim16 = "The German architect Gottfried Semper was asked to produce a design for the museum."
records.append({
    "id": "OST30-16", "category": "multi_passage", "type_id": "16", "type": "来源之间冲突",
    "context_passages": docs, "generated_answer": claim16,
    "gold_claims": [{
        "claim_text": claim16, "verdict": "UNSUPPORTED",
        "reasoning": "Passage 0 says Gottfried Semper produced the 1855 design at Cole's request; passage 1 says Ludwig "
                     "Mies van der Rohe did. The sources conflict and neither takes priority, so the claim cannot be "
                     "verified from the context.",
        "citations": []}],
    "gold_status": "labeled",
    "source": {"dataset": "FaithEval (inconsistent)", "hf": "Salesforce/FaithEval-inconsistent-v1.0", "split": "test",
               "qid": r["qid"], "subset": r["subset"], "license": "Apache-2.0（原文出自 SQuAD，即维基百科，CC BY-SA）",
               "transform": "按 Document: 切成两段；回答用第一种答案写成一句陈述"},
    "notes": {"fit": "符合", "fit_note": "两段对同一件事（1855 年应 Cole 之请为博物馆做设计的德国建筑师）给出不同的人名，回答只采信其中一段",
              "gold_origin": "FaithEval 的标准答案是「原文有冲突」；按本数据集的约定，只采信冲突一方的说法判 UNSUPPORTED",
              "replaced": "原来的 cab69864 原文是 CNN 新闻，公开仓库有版权顾虑，10-22 换掉"},
})

# 22: VitaminC test (real revision), REFUTES; the claim adds "All" to a sentence the evidence otherwise backs
ROOT = Path(__file__).resolve().parents[2]


def detok(s: str) -> str:
    s = s.replace("-LRB-", "(").replace("-RRB-", ")")
    s = re.sub(r"\s+([.,;:!?)'])", r"\1", s)
    return re.sub(r"\(\s+", "(", s).strip()


vitaminc = {json.loads(l)["unique_id"]: json.loads(l)
            for l in (ROOT / "var/vitaminc/test.jsonl").read_text(encoding="utf-8").splitlines()}
r = vitaminc["5ebe6f30c9e77c0008408a82_3"]
records.append({
    "id": "OST30-22", "category": "subtle_distortion", "type_id": "22", "type": "主句正确、限定词错误",
    "context_passages": [r["evidence"]], "generated_answer": detok(r["claim"]),
    "gold_claims": [{
        "claim_text": detok(r["claim"]), "verdict": "CONTRADICTED",
        "reasoning": "The film did get reviews from critics, but they were mixed to positive, so not all of them "
                     "were favorable. VitaminC labels the claim REFUTES.",
        "citations": [{"index": 0, "quote": "Dolphin Tale 2 received mixed to positive reviews from critics"}]}],
    "gold_status": "labeled",
    "source": {"dataset": "VitaminC", "hf": "tals/vitaminc", "split": "test", "unique_id": r["unique_id"],
               "page": r["page"], "revision_type": r["revision_type"], "label": r["label"], "license": "CC BY-SA 3.0",
               "big_bench_canary": r["big_bench_canary"],
               "transform": "证据句原样作为原文；claim 只去掉标点前多余的空格作为回答"},
    "notes": {"fit": "符合", "fit_note": "「影评人给了评价」是对的，错在 All：原文是褒贬不一到正面",
              "gold_origin": "判定来自 VitaminC 标签 REFUTES → CONTRADICTED；引文是证据句里对应的那一段"},
})

# 24: SciTab table (first in seeded order with all three labels); one supports, one refutes, one not-enough-info claim
scitab = json.loads((ROOT / "var/scitab/sci_tab.json").read_text(encoding="utf-8"))
TABLE = "1551d88a-ea00-4a48-a576-e3535dd3b391"
rows_t = [x for x in scitab if x["table_id"] == TABLE]
t0 = rows_t[0]
clean_cell = lambda c: re.sub(r"\[(BOLD|ITALIC|EMPTY)\]\s*", "", c).strip()
header = " | ".join(clean_cell(c) for c in t0["table_column_names"])
body = [" | ".join(clean_cell(c) for c in row) for row in t0["table_content_values"]]
table_text = "\n".join([t0["table_caption"], header, *body])
by_label = {lab: sorted([x for x in rows_t if x["label"] == lab], key=lambda x: x["id"])
            for lab in ("supports", "refutes", "not enough info")}
avg = next(line for line in body if line.startswith("Average"))
picked = {}
for lab in ("refutes", "supports", "not enough info"):
    pool = by_label[lab]
    if lab == "supports":  # not the rewording of the refuted claim, or the answer contradicts itself
        ref = picked["refutes"]["claim"]
        pool = [x for x in pool if len(set(x["claim"].lower().split()) & set(ref.lower().split()))
                < 0.5 * len(set(ref.lower().split()))]
    picked[lab] = random.Random(f"20261022-24-{lab}").choice(pool)
s, c, u = picked["supports"], picked["refutes"], picked["not enough info"]
records.append({
    "id": "OST30-24", "category": "subtle_distortion", "type_id": "24", "type": "结构化字段（表格）",
    "context_passages": [table_text], "generated_answer": " ".join(x["claim"] for x in (s, c, u)),
    "gold_claims": [
        {"claim_text": s["claim"], "verdict": "SUPPORTED",
         "reasoning": "Average row: Proposed 0.579 is above every model except Word2Vec 0.6141. SciTab label: supports.",
         "citations": [{"index": 0, "quote": header}, {"index": 0, "quote": avg}]},
        {"claim_text": c["claim"], "verdict": "CONTRADICTED",
         "reasoning": "SPINE is far below every other model on almost every test (average 0.207). SciTab label: refutes.",
         "citations": [{"index": 0, "quote": header}, {"index": 0, "quote": avg}]},
        {"claim_text": u["claim"], "verdict": "UNSUPPORTED",
         "reasoning": "The table says nothing about this. SciTab label: not enough info.", "citations": []}],
    "gold_status": "labeled",
    "source": {"dataset": "SciTab", "github": "XinyuanLu00/SciTab", "table_id": TABLE, "paper_id": t0["paper_id"],
               "claim_ids": [s["id"], c["id"], u["id"]], "license": "MIT",
               "transform": "表格按「标题 / 表头 / 每行一条，单元格用 | 分隔」写成一段原文，去掉 [BOLD] 等标记；同一张表上支持、反驳、信息不足的 claim 各取一条，按这个顺序拼成回答"},
    "notes": {"fit": "符合", "fit_note": "多行数值表，回答三句分别对应有依据、与表格矛盾、表格没有的信息",
              "gold_origin": "判定来自 SciTab 标签；引文是表头和 Average 行，由我们选"},
})

REVIEWED = {  # human review of the parts we added (answers, citations, conventions), 10-22
    "OST30-16": "10-22 人工复核通过：两段只差人名，回答只采信一方，按约定判 U",
    "OST30-19": "10-22 人工复核通过：两句分别由段 8、段 10 支持；段 3 只说机场在马朱罗、没说首都，不算替代证据",
    "OST30-22": "10-22 人工复核通过：「褒贬不一」否定「全部正面」",
    "OST30-24": "10-22 人工复核通过：第 3 条按 SciTab 标签判 U（表里没区分哪些模型是无监督的）",
    "OST30-05": "10-22 复核通过：拒答，没有可核对的事实，金标为空",
}
for rec in records:
    if rec["id"] in REVIEWED:
        rec["gold_status"] = "reviewed"
        rec["notes"]["review"] = REVIEWED[rec["id"]]

for rec in records:
    errs = check(rec)
    assert not errs, (rec["id"], errs)

existing = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
new_ids = {r["id"] for r in records}
merged = sorted([r for r in existing if r["id"] not in new_ids] + records, key=lambda r: r["type_id"])
OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8")
print(f"wrote {len(records)} external records; {len(merged)} records total")
