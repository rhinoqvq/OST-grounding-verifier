# grounding-verifier

[English](README.md) | 中文

一个 HTTP 服务：把大模型生成的回答对照给定的原文段落做核查，拆成 claim，逐条判定 `SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`，并附上逐字引文。

**状态：开发中。** 三种方法、评测集和跑分脚本都已完成，完整的评测报告还在写。

三种方法在这些文件夹里：

| 方法 | 文件夹 | 做什么 |
|---|---|---|
| baseline | [`src/verifier/methods/baseline/`](src/verifier/methods/baseline/) | 不调模型。回答的每一句是一条 claim，BM25 选引文，词面覆盖、数字和否定检查给判定。 |
| minimal_llm | [`src/verifier/methods/minimal_llm/`](src/verifier/methods/minimal_llm/)（gpt-6.1-sol）<br>[`src/verifier/methods/minimal_jev/`](src/verifier/methods/minimal_jev/)（jev） | 每条回答调一次模型，放进全部原文和整段回答。gpt-6.1-sol 在这一次调用里自己拆分、判定、给引文；Jev 只回答选择题，整段回答就是一条 claim。 |
| optimized（OST） | [`src/verifier/methods/optimized/`](src/verifier/methods/optimized/) | 我们的 OP 版本，见下文。只调 Jev。 |

三种方法共用同一个服务（[`src/verifier/service.py`](src/verifier/service.py)）和同一套响应约束（[`src/verifier/output/`](src/verifier/output/)）。

**OP29：我们的优化版本。** OPn 是优化方法的迭代编号，从 OP0 到 OP29；OP29（提交 `c78c508`）是现行默认。它先用一个 Jev 问题把回答的每一句对照整份原文判一次，只有没被明确判为支持的句子，才拆成 claim 逐条判定。在附带的 30 题评测集上，它和 gpt-6.1-sol 单次调用的正确率分不出高下，成本是它的 0.20 倍，耗时约七分之一。在我们的开发集 RAGTruth 上，它标出错误片段的能力约是 Jev 单次调用的两倍（字符级 F1 31.8 对 15.0）。

详细文档在 [`docs/`](docs/)：
- [版本迭代](docs/version-history.zh-CN.md)：OP0 → OP29 分三个阶段，每个阶段改了什么、数字动了多少。
- [消融实验](docs/ablation.zh-CN.md)：OST 的每个机制各自贡献多少。
- [成本](docs/cost.zh-CN.md)：各方法、各版本的成本，以及钱花在哪一环。
- [失败分析](docs/failure-analysis.zh-CN.md)：附带测评集上各方法错在哪些题、为什么错。

## OP29 的实现和效果

怎么做的：

1. **只认偏移，不认文字。** 原文段落和回答都以不可变的方式存下来，后面所有环节都指向字符偏移。每条引文都由代码从原文里切出，天然逐字。
2. **先按句判定。** 回答的每一句向 Jev 问一次，对照整份原文（原文超过约 2.6 万 token 才收窄），判 S、C 还是 U。最高选项是 S 而且找到了证据段落，这句里的 claim 就都继承 S，再用一道选择题挑出引文句。大多数 claim 在这一步就定了。
3. **其余的逐条判定。** 其他句子拆成答案树：代码提出候选切分点，由 Jev 逐个决定切不切；元话语门去掉「根据所给段落」这类话。每条 claim 对照 Jev 排出的相关原文句判定；如果没有一句过相关性门槛，就保留排名第一的那句作证据。之后还有出处门、反驳过滤，以及给需要两段原文才能支持的 claim 用的桥接检查。原文超过约 2.6 万 token 时，还会把原文建成树，把 claim 映射上去。
4. **记账交给普通代码：** 在答案树上汇总判定，以及最后的响应约束（引文段号不越界、引文逐字、引文和判定一致）。

它不做的：这条路径上没有确定性的数字或否定检查；按句判出的 S 取的是 Jev 的最高选项，不设概率门槛。

效果，在 RAGTruth 上（2,700 条回答；这是我们的开发集，数字偏乐观）：

| | minimal_llm（jev） | OST（OP29） | 差值，95% 区间 |
|---|---:|---:|---|
| 回答级 F1：这段回答有没有错 | **77.7** | 69.6 | −8.0 [−9.8, −6.2] |
| 字符级 F1：错在哪些片段 | 15.0 | **31.8** | +16.9 [+15.6, +18.2] |
| 每条回答成本（按标价，只算输入） | **$0.00005** | $0.00095 | |

- 判断一段回答「有没有错」，每条一次请求更准；能说出「错在哪」的是 OST。每个机制各自贡献多少，见[消融实验](docs/ablation.zh-CN.md)。
- 从 OP1 到 OP29，在 900 条 QA 回答上：回答级 F1 38.9 → 48.0，字符级 F1 24.2 → 36.9，每条回答成本 $0.00097 → $0.00066。三个阶段的经过见[版本迭代](docs/version-history.zh-CN.md)。
- 从 OP27 到 OP29，RAGTruth 全量成本从 $6.91 降到 $2.57，主要因为判为支持的句子不再做逐条映射和判定，原文树也只在长原文上建；F1 同时上升 +0.71 [+0.25, +1.17] / +0.76 [+0.46, +1.04]。分环节的账见[成本](docs/cost.zh-CN.md)。

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

参与比较的方法：
- baseline：不调模型。
- minimal_llm（gpt-6.1-sol）：每题调一次 gpt-6.1-sol，提示词里的三条判定定义照抄题目原文。
- minimal_llm（jev）：每题向 Jev 发一次请求，放进全部原文和整段回答，问回答是 SUPPORTED、CONTRADICTED 还是 UNSUPPORTED、哪一段是证据；整段回答作为一条 claim 输出。
- OST（optimized，OP29）：只调 Jev。

所有方法的输入都只有 `context_passages` 和 `generated_answer`，金标不发给系统。

| 指标 | baseline | minimal_llm（gpt-6.1-sol） | minimal_llm（jev） | OST（optimized） |
|---|---|---|---|---|
| claim 准确率（71 条） | 0.662 | **0.845** | 0.662 | 0.803 |
| 95% 区间（按题重采样） | 0.545–0.761 | 0.750–0.922 | 0.523–0.784 | 0.672–0.898 |
| 三分类 macro-F1 | 0.591 | **0.819** | 0.630 | 0.760 |
| C/U 报警 F1 | 0.780 | **0.909** | 0.815 | 0.877 |
| 回答级 F1（这段回答有没有错） | 0.750 | **0.878** | 0.865 | 0.865 |
| 字符级 F1（错在哪些片段） | 0.715 | 0.729 | 0.693 | **0.753** |
| 真错判成 S（共 37 条错） | 5 | **2** | 4 | 5 |
| 整题全对 | 10 / 30 | **19 / 30** | 13 / 30 | 17 / 30 |
| 每题耗时 p50 / p95 | 0.0 / 0.1 s | 7.1 / 18.1 s | **0.2 / 0.3 s** | 1.0 / 1.8 s |
| 30 题成本（按标价推算） | $0 | $0.45 | **$0.007** | $0.09 |

配对差（claim 准确率，按题重采样 10,000 次，29 / 30 两题共用原文，算作一组）：
- OST − minimal_llm：−4.2 个百分点，95% 区间 [−13.0, +3.8]；
- OST − minimal_llm（jev）：+14.1，95% 区间 [+3.4, +25.3]；
- OST − baseline：+14.1，95% 区间 [+2.0, +26.1]。

按原文长度分档：30 题里，短（< 5 千字符）、中（5 千–2 万）、长（2 万–12.4 万）三档，OST 的成本分别是 sol 的 0.07、0.10、0.27 倍，合计 0.20 倍；每题耗时 1.2 / 1.0 / 1.5 秒，sol 是 9.2 / 8.5 / 9.2 秒；minimal_llm（jev）在各档都是 0.2 秒、每题不到 $0.001。下图把每条回答的成本沿原文长度画到设计上限（原文 50 万字符），数据是 300 + 500 条 ANAH 回答（只看汇总）；Jev 按标价只算输入，输出免费。各方法和各版本的成本见[成本](docs/cost.zh-CN.md)。

![每条回答的成本随原文长度](docs/figures/cost_by_source_length.svg)

**结论**：在自建的 30 题评测集上：
- 和 minimal_llm（gpt-6.1-sol）相比，OST 的正确率分不出高下（0.803 对 0.845，差值的 95% 区间 [−13.0, +3.8] 包含 0），每题都更快（p50 1.0 秒对 7.1 秒），按标价推算成本是它的 0.20 倍；
- 和 minimal_llm（jev）相比，判断一段回答有没有错，两者一样（回答级 F1 都是 0.865）。OST 多出来的是逐条判定和定位：claim 准确率高 14.1 个百分点 [+3.4, +25.3]，字符级 F1 0.753 对 0.693。minimal_llm（jev）整段只给一个判定，对齐到金标时每一句都继承这个判定，所以 claim 准确率这一项对它天然不利。它也是最快、最便宜的（每题 0.2 秒、$0.0002）。

几点说明：
- **成本**：是 token 数乘标价算出来的，不是实际账单。gpt-6.1-sol 按第三方标价，输入 $2、输出 $10（每百万 token）；Jev 按官方标价，输入 $0.042（每百万 token），输出不收费。
- **跑的轮数**：每一列都是一轮。OST 这一列是现行默认配置（OP29，提交 `c78c508`；`main` 之后的提交不改变它的判定）。Jev 每次跑的输出会有出入：和上一个默认配置（整句护栏版）相比，71 条判定里变了 2 条（04#4 S→U，判定模型给 S 的概率从 0.52 变成 0.46；18#1 C→U，换了一句证据），这两条在更早的几轮之间也在跳。sol 这一列的提示词用题目原文的定义；之前那版转述定义的提示词准确率相同，只有 1 条判定不同。
- **题集**：30 题对两个配置之间的差别只能给出很宽的区间，所以上面不说谁胜谁负。另外，这套题是我们按自己的架构设计的。
- **只比了 sol**：minimal_llm 换成 qwen3.8-flash，30 题只要约 $0.03，约是 OST 的三分之一。

各方法错在哪些题、为什么错，见[失败分析](docs/failure-analysis.zh-CN.md)。

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

每种方法各跑一遍，再打分、统计每题的耗时和成本。每种方法需要的 key：
- baseline：不需要；
- minimal_llm（gpt-6.1-sol），`--method minimal_llm`：`OPENAI_API_KEY`，默认模型就是 gpt-6.1-sol；
- minimal_llm（jev），`--method minimal_jev`：`TYPESAFE_API_KEY`；
- OST（optimized）：`TYPESAFE_API_KEY`。

```bash
.venv/Scripts/python benchmark/run_ost30.py --method baseline
.venv/Scripts/python benchmark/run_ost30.py --method minimal_llm
.venv/Scripts/python benchmark/run_ost30.py --method minimal_jev
.venv/Scripts/python benchmark/run_ost30.py --method optimized
.venv/Scripts/python benchmark/score_ost30.py baseline minimal_llm minimal_jev optimized
.venv/Scripts/python benchmark/cost_latency_ost30.py minimal_llm minimal_jev optimized
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
