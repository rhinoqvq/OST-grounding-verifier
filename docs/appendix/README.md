# 附录

[返回 README](../../README.zh-CN.md)

正文在 [`docs/`](../)，这里放推导、完整的规则和数据，给想往下核对的人看。

## 文档

| 附录 | 讲什么 | 正文 |
|---|---|---|
| [逐版本数据](version-table.zh-CN.md) | OP0 → OP29 每个版本的改动、起因、分数、成本和延迟 | [版本迭代](../version-history.zh-CN.md) |
| [语义树的数学论证](semantic-tree-proofs.zh-CN.md) | 拆分算子和判定组合规则的形式化：能证明什么、前提是什么、反例实测 | [设计与实现](../design.zh-CN.md) |
| [逻辑函数的职责重合审查](redundancy-audit.zh-CN.md) | 承载逻辑语义的函数和词表有没有重复：哪几处可证冗余、哪几处可证必要，外部审查后的修订，以及后来落实了哪些 | [设计与实现](../design.zh-CN.md) |
| [Jev 题目清单](jev-questions.zh-CN.md) | 每个阶段问 Jev 哪些题、选项是什么、代码怎么用答案 | [设计与实现](../design.zh-CN.md) |
| [OST30 抽取标准](ost30-extraction.zh-CN.md) | 每个题型的筛选规则、种子、跳过理由、金标改动、合成回答怎么写 | [评测集与指标](../evaluation.zh-CN.md) |

## 数据

都在 [`data/`](data/)。

| 文件 | 内容 |
|---|---|
| [`ost30_runs/`](data/ost30_runs/) | README 结果表那四次运行：原始响应（`results_*.jsonl`，每题一行）和每次模型调用的记录（`runs_*.sqlite3`）。四种方法：baseline、minimal_llm（gpt-6.1-sol）、minimal_llm（jev）、OST（OP29） |
| [`ost30_claims.csv`](data/ost30_claims.csv) | 71 条金标 claim，每条的金标判定和四种方法对齐后的判定 |
| [`versions_qa900.csv`](data/versions_qa900.csv) | QA-900 上每个版本的分数、成本、调用次数、延迟 |
| [`ablation_ragtruth.csv`](data/ablation_ragtruth.csv) | RAGTruth 2,700 上消融梯子的每一级：分数、相对上一级的差值和区间、全量成本 |
| [`cost_by_stage_ragtruth.csv`](data/cost_by_stage_ragtruth.csv) | RAGTruth 全量成本按环节拆开：OP27、OP29 原文树常开、OP29 默认 |
| [`ragtruth_by_task.csv`](data/ragtruth_by_task.csv) | RAGTruth 分任务：minimal_llm（jev）和 OP29 的回答级 / 字符级 F1 和配对差 |
| [`sample_ost30_15_op29.json`](data/sample_ost30_15_op29.json) | 附带测评集第 15 题上 OP29 的一次重跑（和记录的那次 claim 完全相同）：最终 claim 和每个单元的判定、证据句；README 里语义树动图画的就是它 |
| [`sample_wal_op29.json`](data/sample_wal_op29.json) | 作业 SQLite WAL 例子上 OP29 的一次真实运行：最终 claim 和每个单元的判定、标记、证据句；README 动图画的就是它 |

成本都按标价只算输入（Jev 每百万 token $0.042，输出不收费）。区间都是 95%，配对 bootstrap，种子 20261022。

### 用原始响应复算 README 的表

把 `ost30_runs/` 里的文件复制到 `var/ost30/`，然后：

```bash
.venv/Scripts/python benchmark/score_ost30.py baseline minimal_llm_sol minimal_llm_jev optimized_op29
.venv/Scripts/python benchmark/cost_latency_ost30.py minimal_llm_sol minimal_llm_jev optimized_op29
```

claim 准确率、报警 F1、回答级和字符级 F1、整题全对、耗时和成本都能复算出来，和 README 一致。「引文核对后准确率」用到的人工裁定是给更早的几轮做的，对这几个文件不完整，README 没有引用这一项。
