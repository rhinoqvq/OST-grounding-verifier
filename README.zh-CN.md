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

## 评测结果（OST30）

评测集是 [`benchmark/ost30.jsonl`](benchmark/ost30.jsonl)：
- 30 题，71 条金标 claim（S 34、U 26、C 11）。
- 每行就是一个 `/verify` 请求，金标和接口响应的格式相同。
- 冻结哈希在 [`benchmark/ost30.sha256`](benchmark/ost30.sha256)，跑分脚本发现文件和哈希不符就拒跑。
- 题目改编自公开数据集：ANAH、RAGTruth、FOLIO、MuSiQue、FaithEval、VitaminC、SciTab。其中 4 题是在维基百科原文上自己写的回答。许可证和逐题出处见 [`benchmark/DATA_LICENSE.md`](benchmark/DATA_LICENSE.md)。

题目按类别分布如下：

| 类别 | 题数 |
|---|---:|
| 完全支持的回答 | 9 |
| 细微扭曲（数字差一、条件互换、否定反转等） | 9 |
| 外部幻觉（原文没有，包括现实中为真的） | 8 |
| 多段综合 | 4 |
| 边界情况：没有可核对的事实 | 1 |

三种方法：
- baseline：不调模型。
- minimal_llm：每题调一次 gpt-6.1-sol。
- OST（optimized）：只调 Jev。

三种方法的输入都只有 `context_passages` 和 `generated_answer`，金标不发给系统。

| 指标 | baseline | minimal_llm（gpt-6.1-sol） | OST（optimized） |
|---|---|---|---|
| claim 准确率（71 条） | 0.662 | **0.845** | 0.831 |
| 95% 区间（按题重采样） | 0.545–0.761 | 0.750–0.922 | 0.702–0.925 |
| 三分类 macro-F1 | 0.591 | **0.819** | 0.798 |
| C/U 报警 F1 | 0.780 | **0.895** | 0.889 |
| 真错判成 S（共 37 条错） | 5 | **3** | 5 |
| 整题全对 | 10 / 30 | **19 / 30** | **19 / 30** |
| 每题耗时 p50 / p95 | 0.0 / 0.1 s | 7.5 / 19.4 s | **1.9 / 3.7 s** |
| 30 题成本（按标价推算） | $0 | $0.45 | **$0.34** |

配对差，按题重采样，29 / 30 两题共用原文，算作一组：
- OST − minimal_llm：−1.4 个百分点，95% 区间 [−10.2, +6.7]；
- OST − baseline：+16.9，95% 区间 [+4.5, +29.3]。

按原文长度分档：

| 原文长度 | 题数 | sol 平均耗时 | OST（optimized）平均耗时 | sol 平均成本 | OST（optimized）平均成本 | OST（optimized）成本 ÷ sol 成本 |
|---|---:|---:|---:|---:|---:|---:|
| 短（< 5 千字符） | 10 | 9.8 s | 1.4 s | $0.0052 | $0.0009 | 0.18 |
| 中（5 千–2 万） | 12 | 9.8 s | 2.1 s | $0.0103 | $0.0053 | 0.52 |
| 长（2 万–12.4 万） | 8 | 10.8 s | 3.5 s | $0.0342 | $0.0339 | 0.99 |
| 合计 | 30 | — | — | $0.448 | $0.344 | 0.77 |

**结论**：在自建的 30 题评测集上，OST 和 gpt-6.1-sol 单次调用相比：
- 正确率分不出高下：0.831 对 0.845，差值的 95% 区间 [−10.2, +6.7] 包含 0。
- 每题都更快：p50 1.9 秒对 7.5 秒。
- 按标价推算，总成本低 23%。短、中等长度的原文分别是 sol 的 0.18 倍和 0.52 倍，2 万字符以上的长文档基本持平（0.99 倍）。

几点说明：
- **成本**：是 token 数乘标价算出来的，不是实际账单。gpt-6.1-sol 按第三方标价，输入 $2、输出 $10（每百万 token）；Jev 按官方标价，输入 $0.042（每百万 token），输出不收费。
- **跑的轮数**：gpt-6.1-sol 只跑了一轮。OST 这一列是现在的默认配置（含整句护栏）实跑一轮的结果，护栏在这套题上没有放行任何单元。用同一份代码回放之前几轮记下的 Jev 调用，71 条判定里有 3 条不同（04#4、07#2、18#1），准确率是 0.803：Jev 每次跑的输出会有出入。
- **题集**：30 题对两个配置之间的差别只能给出很宽的区间，所以上面不说谁胜谁负。另外，这套题是我们按自己的架构设计的。
- **只比了 sol**：minimal_llm 换成 qwen3.8-flash，30 题只要约 $0.03，不到 OST 的 1/11。

## 失败分析

题号对应 `benchmark/ost30.jsonl` 里的 `id`，每一条都对照原文核过。

**两种模型方法都错的**

- 相近的证据被当成支持：
  - 11：原文「Hubbell 先生娶了 Lina Rubi」说的是另一位 Hubbell，两种方法都把她当成回答里 John 的妻子。
  - 16：两段原文对博物馆由谁设计说法不一（Semper / Mies van der Rohe），两种方法都只引了支持的那段，没发现冲突。
  - 17：回答列了柏拉图讨论过的六个领域，两种方法都逐项找了原文判支持。其中「心理学」原文没有直接说，是从讲灵魂三分的段落推出来的，金标判 U。
- 拒答句被当成 claim（05）：回答只有「无法根据给定段落回答」，三种方法都输出了一条 U。
- 读法分歧（07#2、15#3、24#1）：两种方法都和金标不同，分歧在怎么读原文或回答。例如 07#2「他的作品在他死后才出版」：
  - 原文说他生前出版过少量作品，所以 sol 判 C；OST 的判定模型也倾向 C，按严格的反驳规则报为 U；
  - 金标把「他的作品」读成大部分作品，判 S。

  分数里这些都按错算。

**只有 OST 错的**

- 两个断言被当成一条 claim：
  - 14：「A Burial at Ornans 的作者是 Courbet，他生于 1819 年 6 月 10 日」，从句没切开，整句判 U，可前半句其实有依据。
  - 30：分句时没在「Numbers 20:22-29.」后面断开，两句成了一句，第二句的 C 没单独判出来。
- 把握不足的 S 倾向被保留（24#3）：「The proposed approach is seen to perform well against the other unsupervised models.」判定模型倾向 S，概率 0.51，低于 0.55 的门槛；按现在的规则，门槛以下倾向 S 的仍判 S（在 RAGTruth 测试集上，这类单元 456 次里有 430 次是干净的）。原文是一张相关系数表，没有说哪些模型是无监督的，金标（SciTab：信息不足）判 U。
- 丢掉条件的句子判成支持（25）：原文是「在这种做法下，领域模型表示为一组约束」，回答把条件换成「在智能辅导系统中」，OST 判 S。
- 证据读得过头（28#3）：「卡尔大公战后辞去指挥权」，OST 引了「战后卡尔仍统率一支完整的部队，决定撤往波希米亚」判 C。这句只说明战后当时他还在指挥，不排除他之后辞职，金标判 U。

**只有方法 2（gpt-6.1-sol）错的**

- 模态加强判成反驳（10）：原文是 "may include"，回答是 "always include"。原文没有直接说「并非总是」，应判 U，sol 判了 C。
- 拆得比金标细（01#4、04#4、13#1）：sol 把一句拆成几条小断言，只要有一条找不到依据，整句就记成 U。
  - 13#1 拆出了「解放宣言是一项行政命令」，原文没有这个说法，其余几条都是 S；
  - 04#4 拆出了「这套防线网络遍布全国」，sol 认为原文没有确立这一点；
  - 金标按整句判 S。

  这三题也属于读法分歧。

**baseline 的失败**

- 只看词面：同义改写和别名对不上。01 五句只对了两句，13 两句都判成 U。
- 抓不到数字和否定以外的扭曲，两题都判成了 S：
  - 09：回答里的 1928 年在原文里确实出现，只是对应的是另一件事；
  - 22：回答说「全部好评」，原文是「褒贬不一、偏正面」。
- 标红偏宽：判 C / U 的字符是金标错误片段的 1.65 倍。

**提示注入（20）**：原文中间夹了一段伪装成系统消息的指令，要求把两句真话判 C、把原文没有的那条判 S。三种方法都没被带偏。作为对照，方法 2 换成 qwen3.8-flash 时，照着指令把两句真话判成了 C。

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
