# grounding-verifier

English | [中文](README.zh-CN.md)

An HTTP service that checks an LLM-generated answer against the provided source passages: it
splits the answer into claims, judges each one `SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`, and
attaches verbatim citations.

**Status: work in progress.** The three methods, the evaluation set and the benchmark runner are
in place; the full evaluation report is still being written.

**Contents**

- [The three methods](#the-three-methods)
- [OP29, our optimized version](#op29-our-optimized-version)
- [Documents](#documents)
- [Evaluation report](#evaluation-report)
  - [The evaluation set](#the-evaluation-set)
  - [Results](#results)
  - [Key failure cases](#key-failure-cases)
  - [How we iterated](#how-we-iterated)
  - [Cost](#cost)
- [Run](#run)
- [Tests](#tests)
- [Benchmark](#benchmark)

## The three methods

Where the three methods are:

| method | folder | what it does |
|---|---|---|
| baseline | [`src/verifier/methods/baseline/`](src/verifier/methods/baseline/) | No model calls. Each answer sentence is a claim; BM25 picks the citation; lexical coverage, number and negation checks give the verdict. |
| minimal_llm | [`src/verifier/methods/minimal_llm/`](src/verifier/methods/minimal_llm/) (gpt-6.1-sol)<br>[`src/verifier/methods/minimal_jev/`](src/verifier/methods/minimal_jev/) (jev) | One model call per answer with all passages and the whole answer. gpt-6.1-sol splits, judges and cites in that call; Jev answers choice questions, so the whole answer is one claim. |
| optimized (OST) | [`src/verifier/methods/optimized/`](src/verifier/methods/optimized/) | Our OP versions, described below. Jev only. |

All three run behind the same service ([`src/verifier/service.py`](src/verifier/service.py)) and
the same response contract ([`src/verifier/output/`](src/verifier/output/)): citation indexes in
range, quotes verbatim in the source, and no S or C verdict without a citation.

## OP29, our optimized version

OPn numbers the iterations of the optimized method, OP0 to OP29;
OP29 (commit `c78c508`) is the default. It splits the answer into a semantic tree whose claims all
point at character offsets in the answer, and code cuts every citation out of the source by
offset. Judging has two steps: each sentence first gets one Jev question against the whole
source, and the claims of a sentence judged supported are settled as S; the remaining claims are
then judged one by one. Jev only answers choice questions and never writes text, so whether to
split, which verdict and which citation are all decided by code from its probabilities. On the
accompanying 30-item set it is not distinguishable in accuracy from a single gpt-6.1-sol call, at
0.20× the cost and a seventh of the latency. See [design](docs/design.zh-CN.md).

The animation walks through OP29's four steps on the assignment's SQLite WAL example; it is a
real run (5 Jev requests, about $0.0003).

![How OST checks an answer](docs/figures/how_it_works.svg)

## Documents

The README is the summary; the full write-ups are in [`docs/`](docs/) (in Chinese for now):

| document | what it covers |
|---|---|
| [Design](docs/design.zh-CN.md) | design rationale, the semantic tree and its operators, how Jev is used, one verification run step by step, deterministic validation, error boundaries |
| [Evaluation set and metrics](docs/evaluation.zh-CN.md) | how the 30 items were chosen, how the gold was set, scoring rules, metric definitions |
| [Full results](docs/results.zh-CN.md) | every metric on the accompanying set with notes, and the comparison on the RAGTruth development set |
| [Failure analysis](docs/failure-analysis.zh-CN.md) | which items each method gets wrong, and why |
| [Version history](docs/version-history.zh-CN.md) | OP0 → OP29 in three phases: what changed and what moved |
| [Ablation](docs/ablation.zh-CN.md) | what each OST mechanism contributes |
| [Cost](docs/cost.zh-CN.md) | cost per method and per version, and where the money goes |
| [Appendix](docs/appendix/) | per-version data, the semantic-tree proofs, the Jev question catalogue, the full OST30 extraction rules, plus raw responses and data tables |

## Evaluation report

### The evaluation set

[`benchmark/ost30.jsonl`](benchmark/ost30.jsonl): 30 items and 71 gold claims (S 34, U 26, C 11).
Each line is a `/verify` request, and the gold uses the same format as the API response; the
frozen hash is in [`benchmark/ost30.sha256`](benchmark/ost30.sha256). Items are adapted from
public datasets (ANAH, RAGTruth, FOLIO, MuSiQue, FaithEval, VitaminC, SciTab); four of them pair
Wikipedia text with answers written for this set. We designed the set around OST characteristics
found during development, so each item tests one specific mechanism.

| category | items |
|---|---:|
| fully supported answers | 8 |
| subtle distortions (off-by-one numbers, swapped conditions, inverted negations, …) | 9 |
| extrinsic hallucinations (not in the passages, including facts true in the real world) | 8 |
| multi-passage synthesis | 4 |
| edge case: no checkable facts | 1 |

Item selection, gold and scoring rules are in the
[evaluation set and metrics](docs/evaluation.zh-CN.md). The other three edge cases the assignment
names (empty context, malformed model output, upstream API errors) are covered by offline tests.

### Results

Every method receives only `context_passages` and `generated_answer`.

| metric | baseline | minimal_llm (gpt-6.1-sol) | minimal_llm (jev) | OST (OP29) |
|---|---|---|---|---|
| claim accuracy (71 claims, 95% interval) | 0.662 [0.545, 0.761] | **0.845** [0.750, 0.922] | 0.662 [0.523, 0.784] | 0.803 [0.672, 0.898] |
| C/U alarm F1 | 0.780 | **0.909** | 0.815 | 0.877 |
| answer-level F1 (does the answer contain an error) | 0.750 | **0.878** | 0.865 | 0.865 |
| character-level F1 (which spans are wrong) | 0.715 | 0.729 | 0.693 | **0.753** |
| real errors judged S (of 37) | 5 | **2** | 4 | 5 |
| items fully correct | 10 / 30 | **19 / 30** | 13 / 30 | 17 / 30 |
| latency per item p50 / p95 | 0.0 / 0.1 s | 7.1 / 18.1 s | **0.2 / 0.3 s** | 1.0 / 1.8 s |
| cost for 30 items (list price) | $0 | $0.45 | **$0.007** | $0.09 |

- **Against minimal_llm (gpt-6.1-sol)**, OST is not distinguishable in accuracy: OST − sol is
  −4.2 points, 95% interval [−13.0, +3.8]. OST is faster on every item and costs 0.20× as much.
- **Against minimal_llm (jev)**, the two are equal at telling whether an answer contains an error.
  What OST adds is per-claim verdicts and localization: claim accuracy +14.1 points
  [+3.4, +25.3], character-level F1 0.753 vs 0.693.

Thirty items only give wide intervals, so no winner is claimed. All metrics, the breakdown by
source length and the notes are in the [full results](docs/results.zh-CN.md).

### Key failure cases

- Both model-based methods take nearby evidence as support. In item 11 the source is about a
  different Hubbell; in item 16 two passages disagree, but only the supporting one is cited.
- Two of OST's six own errors come from splitting: in items 14 and 30 two assertions stay in one
  claim and are judged once. The rest are judging errors; in item 18, for example, the
  contradiction needs several passages combined and OST reports U.
- minimal_llm (gpt-6.1-sol) splits finer than the gold, so one unsupported fragment marks the
  whole sentence U.
- minimal_llm (jev) gives one verdict per answer: one error flags every correct sentence with it.
- The baseline is lexical only: paraphrases do not match, and distortions other than numbers and
  negation are missed.

Item-by-item analysis is in the [failure analysis](docs/failure-analysis.zh-CN.md).

### How we iterated

The optimized method went from OP0 to OP29 in three phases: changing how answers are split,
developing and tuning the semantic tree and its operators, and pipeline optimization driven by
ablations of modules and their combinations. Every version was regression-tested on the 900 QA
answers of the RAGTruth development set. From OP1 to OP29, answer-level F1 went 38.9 → 48.0,
character-level F1 24.2 → 36.9, and cost per answer $0.00097 → $0.00066.

The ablations of the last phase showed that one request per answer is the better detector of
*whether* an answer is wrong (answer-level F1 77.7 vs 69.6 on RAGTruth), while judging sentence
by sentence and claim by claim is what buys localization (character-level F1 15.0 vs 31.8).
Splitting the answer helps localization, and mapping onto a source tree showed no measurable
contribution, so OP29 builds the source tree only for long sources and lets supported sentences
skip claim-level judging; the full RAGTruth cost fell from $6.91 to $2.57. See the
[version history](docs/version-history.zh-CN.md) and the [ablation](docs/ablation.zh-CN.md).

### Cost

Costs are token counts times list prices, not billing: Jev input $0.042 per million tokens with
free output; gpt-6.1-sol at third-party prices of $2 / $10 per million input / output tokens. The
longer the source, the smaller OST's cost advantage over sol; above about 26k tokens of source
OST builds a source tree and its cost steps up.

![Cost per answer by source length](docs/figures/cost_by_source_length.svg)

Cost per method, per version and per pipeline stage is in [cost](docs/cost.zh-CN.md).

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Windows; use .venv/bin/pip elsewhere
cp .env.example .env                        # fill in OPENAI_API_KEY, TYPESAFE_API_KEY
.venv/Scripts/uvicorn verifier.service:app --port 8000
```

```bash
curl -X POST localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"context_passages": ["..."], "generated_answer": "..."}'
```

## Tests

```bash
.venv/Scripts/python -m pytest
```

The suite is fully offline: model behavior is scripted through fakes, and the
assignment's SQLite/WAL example runs end-to-end as a fixture.

## Benchmark

Run each method once, then score and compute per-item latency and cost. Keys per method:

- baseline: none.
- minimal_llm (gpt-6.1-sol), `--method minimal_llm`: `OPENAI_API_KEY` (gpt-6.1-sol is the default model).
- minimal_llm (jev), `--method minimal_jev`: `TYPESAFE_API_KEY`.
- OST (optimized): `TYPESAFE_API_KEY`.

```bash
.venv/Scripts/python benchmark/run_ost30.py --method baseline
.venv/Scripts/python benchmark/run_ost30.py --method minimal_llm
.venv/Scripts/python benchmark/run_ost30.py --method minimal_jev
.venv/Scripts/python benchmark/run_ost30.py --method optimized
.venv/Scripts/python benchmark/score_ost30.py baseline minimal_llm minimal_jev optimized
.venv/Scripts/python benchmark/cost_latency_ost30.py minimal_llm minimal_jev optimized
```

Results go to `var/ost30/`: each method's output is `results_<method>.jsonl`, and every model
call is recorded in `runs_<method>.sqlite3`.

- A rerun after an interruption skips the items already done.
- Calls already recorded are served from the database, so they are not paid for again.
- To run the same method with another model, add `--tag` (e.g. `--tag _qwen`); its results are
  stored separately.

The scoring rules are at the top of [`benchmark/score_ost30.py`](benchmark/score_ost30.py):

- how system claims are matched to gold sentences;
- how a sentence split into several claims gets one verdict;
- when a citation counts as supporting.

Hand rulings are in the two JSON files under `benchmark/`.

Reproducing:

- [`benchmark/ost30_build/`](benchmark/ost30_build/) rebuilds OST30 from the public source datasets, byte for byte.
- [`benchmark/ragtruth/`](benchmark/ragtruth/) builds the RAGTruth development inputs, runs a method and scores it; the
  recorded predictions of every run in the docs are included, so the numbers rescore without a model call.
- The raw responses and call logs of the four runs in the results table are in
  [`docs/appendix/data/ost30_runs/`](docs/appendix/data/ost30_runs/).
