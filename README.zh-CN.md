# grounding-verifier

[English](README.md) | 中文

一个 HTTP 服务：把大模型生成的回答拆成原子事实 claim，逐条对照给定的原文段落核查，并附上逐字引文（`SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`）。

**状态：开发中。** 三种方法、评测集和跑分脚本都已完成，结果见下面的「评测结果」。完整的评测报告还在写。

## 一段话讲设计

原文段落和回答都以不可变的方式存下来，后面所有环节都指向字符偏移，所以每条引文都由代码从原文里切出，天然逐字。

拆分时，代码先在回答的每个句子里找出候选切分点，再由 Jev（一个经过校准的判定模型）逐个决定切不切。这样每个 claim 单元都是回答里的一段原文，系统不生成任何文字，也就不会悄悄"修正"回答。原文按同样的方式拆分，判定和证据的问题也由 Jev 回答。

确定性的部分全部由普通代码保证：
- 偏移；
- 覆盖；
- 数字和否定的一致性检查；
- 在语义树上汇总判定；
- 最后的响应约束：引文段号不越界、引文逐字、引文和判定一致。

系统没把握的 claim 报为 `UNSUPPORTED` 并写明原因，从不猜。

## 附带测评集上的评测结果（OST30）

评测集是 [`benchmark/ost30.jsonl`](benchmark/ost30.jsonl)：
- 30 题，71 条金标 claim（S 34、U 26、C 11）。
- 每行就是一个 `/verify` 请求，金标和接口响应的格式相同。
- 冻结哈希在 [`benchmark/ost30.sha256`](benchmark/ost30.sha256)，跑分脚本发现文件和哈希不符就拒跑。
- 题目改编自公开数据集：ANAH、RAGTruth、FOLIO、MuSiQue、FaithEval、VitaminC、SciTab。其中 4 题是在维基百科原文上自己写的回答。许可证和逐题出处见 [`benchmark/DATA_LICENSE.md`](benchmark/DATA_LICENSE.md)。

题目按类别分布如下：

| 类别 | 题数 |
|---|---:|
| 完全支持的回答 | 8 |
| 细微扭曲（数字差一、条件互换、否定反转等） | 9 |
| 外部幻觉（原文没有，包括现实中为真的） | 8 |
| 多段综合 | 4 |
| 边界情况：没有可核对的事实 | 1 |

四种方法：
- baseline：不调模型。
- 单调用 Jev：每题向 Jev 发一次请求，放进全部原文和整段回答，问回答是 SUPPORTED、CONTRADICTED 还是 UNSUPPORTED、哪一段是证据；整段回答作为一条 claim 输出。
- minimal_llm：每题调一次 gpt-6.1-sol，提示词里的三条判定定义照抄题目原文。
- OST（optimized，OP29）：只调 Jev。

四种方法的输入都只有 `context_passages` 和 `generated_answer`，金标不发给系统。

| 指标 | baseline | 单调用 Jev | minimal_llm（gpt-6.1-sol） | OST（optimized） |
|---|---|---|---|---|
| claim 准确率（71 条） | 0.662 | 0.662 | **0.845** | 0.803 |
| 95% 区间（按题重采样） | 0.545–0.761 | 0.523–0.784 | 0.750–0.922 | 0.672–0.898 |
| 三分类 macro-F1 | 0.591 | 0.630 | **0.819** | 0.760 |
| C/U 报警 F1 | 0.780 | 0.815 | **0.909** | 0.877 |
| 真错判成 S（共 37 条错） | 5 | 4 | **2** | 5 |
| 整题全对 | 10 / 30 | 13 / 30 | **19 / 30** | 17 / 30 |
| 每题耗时 p50 / p95 | 0.0 / 0.1 s | **0.2 / 0.3 s** | 7.1 / 18.1 s | 1.0 / 1.8 s |
| 30 题成本（按标价推算） | $0 | **$0.007** | $0.45 | $0.09 |

配对差（claim 准确率，按题重采样 10,000 次，29 / 30 两题共用原文，算作一组）：
- OST − minimal_llm：−4.2 个百分点，95% 区间 [−13.0, +3.8]；
- OST − 单调用 Jev：+14.1，95% 区间 [+3.4, +25.3]；
- OST − baseline：+14.1，95% 区间 [+2.0, +26.1]。

按原文长度分档：30 题里，短（< 5 千字符）、中（5 千–2 万）、长（2 万–12.4 万）三档，OST 的成本分别是 sol 的 0.07、0.10、0.27 倍，合计 0.20 倍；每题耗时 1.2 / 1.0 / 1.5 秒，sol 是 9.2 / 8.5 / 9.2 秒；单调用 Jev 在各档都是 0.2 秒、每题不到 $0.001。下图把每条回答的成本沿原文长度画到设计上限（原文 50 万字符），数据是 300 + 500 条 ANAH 回答（只看汇总）；Jev 按标价只算输入，输出免费。

![每条回答的成本随原文长度](docs/figures/cost_by_source_length.svg)

**结论**：在自建的 30 题评测集上：
- 和 gpt-6.1-sol 单次调用相比，OST 的正确率分不出高下（0.803 对 0.845，差值的 95% 区间 [−13.0, +3.8] 包含 0），每题都更快（p50 1.0 秒对 7.1 秒），按标价推算成本是它的 0.20 倍；
- 和 Jev 单次调用相比，OST 的 claim 准确率高 14.1 个百分点 [+3.4, +25.3]：每题一次请求最快、最便宜，但整段回答只有一个判定，claim 准确率和不调模型的 baseline 一样（0.662），也说不出错在哪一句。

几点说明：
- **成本**：是 token 数乘标价算出来的，不是实际账单。gpt-6.1-sol 按第三方标价，输入 $2、输出 $10（每百万 token）；Jev 按官方标价，输入 $0.042（每百万 token），输出不收费。
- **跑的轮数**：每一列都是一轮。OST 这一列是现行默认配置（OP29，提交 `c78c508`；`main` 之后的提交不改变它的判定）。Jev 每次跑的输出会有出入：和上一个默认配置（整句护栏版）相比，71 条判定里变了 2 条（04#4 S→U，判定模型给 S 的概率从 0.52 变成 0.46；18#1 C→U，换了一句证据），这两条在更早的几轮之间也在跳。sol 这一列的提示词用题目原文的定义；之前那版转述定义的提示词准确率相同，只有 1 条判定不同。
- **题集**：30 题对两个配置之间的差别只能给出很宽的区间，所以上面不说谁胜谁负。另外，这套题是我们按自己的架构设计的。
- **只比了 sol**：minimal_llm 换成 qwen3.8-flash，30 题只要约 $0.03，约是 OST 的三分之一。

## 运行

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Windows；其他系统用 .venv/bin/pip
cp .env.example .env                        # 填入 OPENAI_API_KEY、TYPESAFE_API_KEY
.venv/Scripts/uvicorn verifier.service:app --port 8000
```

```bash
curl -X POST localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"context_passages": ["..."], "generated_answer": "..."}'
```

## 测试

```bash
.venv/Scripts/python -m pytest
```

整套测试完全离线：模型行为用假对象按脚本给出，作业里的 SQLite/WAL 例子作为固定用例端到端跑一遍。

## 跑分

三种方法各跑一遍，再打分、统计每题的耗时和成本。每种方法需要的 key：
- baseline：不需要；
- minimal_llm：`OPENAI_API_KEY`，默认模型就是 gpt-6.1-sol；
- OST（optimized）：`TYPESAFE_API_KEY`。

```bash
.venv/Scripts/python benchmark/run_ost30.py --method baseline
.venv/Scripts/python benchmark/run_ost30.py --method minimal_llm
.venv/Scripts/python benchmark/run_ost30.py --method optimized
.venv/Scripts/python benchmark/score_ost30.py baseline minimal_llm optimized
.venv/Scripts/python benchmark/cost_latency_ost30.py minimal_llm optimized
```

结果写在 `var/ost30/`：每种方法的输出是 `results_<方法>.jsonl`，每次模型调用都记在 `runs_<方法>.sqlite3` 里。
- 中断后再跑，已经跑完的题会跳过。
- 已经记录过的调用直接从库里取，不会再花钱。
- 要用另一个模型再跑同一种方法，加 `--tag`（如 `--tag _qwen`），结果会单独存一份。

评分规则写在 [`benchmark/score_ost30.py`](benchmark/score_ost30.py) 开头：
- 系统拆出的 claim 怎么对到金标句上；
- 一句被拆成几条时怎么合成这一句的判定；
- 引文怎么算撑得住。

人工裁定记在 `benchmark/` 下的两份 json 里。
