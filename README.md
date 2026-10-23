# grounding-verifier

English | [中文](README.zh-CN.md)

An HTTP service that checks an LLM-generated answer against the provided source passages: it
splits the answer into claims, judges each one `SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`, and
attaches verbatim citations.

**Status: work in progress.** The three methods, the evaluation set and the benchmark runner are
in place; the full evaluation report is still being written.

Where the three methods are:

| method | folder | what it does |
|---|---|---|
| baseline | [`src/verifier/methods/baseline/`](src/verifier/methods/baseline/) | No model calls. Each answer sentence is a claim; BM25 picks the citation; lexical coverage, number and negation checks give the verdict. |
| minimal_llm | [`src/verifier/methods/minimal_llm/`](src/verifier/methods/minimal_llm/) (gpt-6.1-sol)<br>[`src/verifier/methods/minimal_jev/`](src/verifier/methods/minimal_jev/) (jev) | One model call per answer with all passages and the whole answer. gpt-6.1-sol splits, judges and cites in that call; Jev answers choice questions, so the whole answer is one claim. |
| optimized (OST) | [`src/verifier/methods/optimized/`](src/verifier/methods/optimized/) | Our OP versions, described below. Jev only. |

All three run behind the same service ([`src/verifier/service.py`](src/verifier/service.py)) and
the same response contract ([`src/verifier/output/`](src/verifier/output/)).

**OP29, our optimized version.** OPn numbers the iterations of the optimized method, OP0 to OP29;
OP29 (commit `c78c508`) is the default. It judges each answer sentence against the whole source
with one Jev question first, and only sentences that are not clearly supported are split into
claims and judged claim by claim. On the accompanying 30-item set it is not distinguishable in
accuracy from a single gpt-6.1-sol call, at 0.20× the cost and a seventh of the latency. On our
RAGTruth development set it marks the wrong spans about twice as well as a single Jev call
(character-level F1 31.8 vs 15.0).

Detailed write-ups are in [`docs/`](docs/) (in Chinese for now):
- [Version history](docs/version-history.zh-CN.md): OP0 → OP29 in three phases, what changed and what moved.
- [Ablation](docs/ablation.zh-CN.md): what each OST mechanism contributes.
- [Cost](docs/cost.zh-CN.md): cost per method and per version, and where the money goes.
- [Failure analysis](docs/failure-analysis.zh-CN.md): which items each method gets wrong on the accompanying set, and why.

## OP29: how it works and what it gets

How it works:

1. **Offsets, not text.** The passages and the answer are stored immutably and everything
   downstream points at character offsets. Every citation is cut from the source by code, so
   quotes are verbatim by construction.
2. **Sentence-level judging first.** Each answer sentence gets one Jev question against the whole
   source (narrowed only above about 26k tokens): S, C or U. When the top answer is S and an
   evidence passage is found, the sentence's claims inherit S, and a second choice question picks
   the citation sentence. Most claims are settled here.
3. **Claim-level judging for the rest.** Other sentences are split into an answer tree: code
   proposes cut points and Jev decides each one, and a meta-discourse gate drops phrases such as
   "based on the passages". Each claim is judged against the source sentences Jev ranks as
   relevant; if none passes the relevance bar, the top-ranked one is kept as evidence. A
   provenance gate, a contradiction filter and a bridge check for claims that need two passages
   follow. Sources over about 26k tokens are also built into a tree that claims are mapped onto.
4. **Plain code does the bookkeeping:** verdict aggregation over the answer tree and the response
   contract (citation indexes in range, quotes verbatim, citations consistent with verdicts).

What it does not do: this path has no deterministic number or negation check, and a
sentence-level S is Jev's top option without a probability threshold.

What it gets, on RAGTruth (2,700 answers; our development set, so the numbers are optimistic):

| | minimal_llm (jev) | OST (OP29) | difference, 95% interval |
|---|---:|---:|---|
| answer-level F1: does the answer contain an error | **77.7** | 69.6 | −8.0 [−9.8, −6.2] |
| character-level F1: which spans are wrong | 15.0 | **31.8** | +16.9 [+15.6, +18.2] |
| cost per answer (list price, input only) | **$0.00005** | $0.00095 | |

- One request per answer is the better detector of *whether* an answer is wrong; OST is the one
  that can say *where*. See the [ablation](docs/ablation.zh-CN.md) for what each mechanism adds.
- From OP1 to OP29, on the 900 QA answers: answer-level F1 38.9 → 48.0, character-level F1
  24.2 → 36.9, cost per answer $0.00097 → $0.00066. See the [version history](docs/version-history.zh-CN.md).
- From OP27 to OP29 the RAGTruth cost fell from $6.91 to $2.57, mostly because supported sentences
  skip claim-level mapping and judging and the source tree is built only for long sources; F1
  rose +0.71 [+0.25, +1.17] / +0.76 [+0.46, +1.04] at the same time. The stage-by-stage account is in
  [cost](docs/cost.zh-CN.md).

## Evaluation results on the accompanying evaluation set (OST30)

The evaluation set is [`benchmark/ost30.jsonl`](benchmark/ost30.jsonl):

- 30 items and 71 gold claims (S 34, U 26, C 11).
- Each line is a `/verify` request; the gold uses the same format as the API response.
- The frozen hash is in [`benchmark/ost30.sha256`](benchmark/ost30.sha256), and the runner
  refuses a file that does not match it.
- Items are adapted from public datasets: ANAH, RAGTruth, FOLIO, MuSiQue, FaithEval, VitaminC
  and SciTab. Four of them pair Wikipedia text with answers written for this set.
- Licenses and per-item attribution are in
  [`benchmark/DATA_LICENSE.md`](benchmark/DATA_LICENSE.md).

Items by category:

| category | items |
|---|---:|
| fully supported answers | 8 |
| subtle distortions (off-by-one numbers, swapped conditions, inverted negations, …) | 9 |
| extrinsic hallucinations (not in the passages, including facts true in the real world) | 8 |
| multi-passage synthesis | 4 |
| edge case: no checkable facts | 1 |

The methods compared:

- **baseline**: no model calls.
- **minimal_llm (gpt-6.1-sol)**: one gpt-6.1-sol call per item, with the three verdict definitions
  from the assignment word for word.
- **minimal_llm (jev)**: one Jev request per item with all passages and the whole answer, asked
  whether the answer is SUPPORTED, CONTRADICTED or UNSUPPORTED and which passage is the evidence;
  the answer is reported as one claim.
- **OST (optimized, OP29)**: Jev only.

Every method receives only `context_passages` and `generated_answer`; the gold is never sent.

| metric | baseline | minimal_llm (gpt-6.1-sol) | minimal_llm (jev) | OST (optimized) |
|---|---|---|---|---|
| claim accuracy (71 claims) | 0.662 | **0.845** | 0.662 | 0.803 |
| 95% interval (item resampling) | 0.545–0.761 | 0.750–0.922 | 0.523–0.784 | 0.672–0.898 |
| 3-way macro-F1 | 0.591 | **0.819** | 0.630 | 0.760 |
| C/U alarm F1 | 0.780 | **0.909** | 0.815 | 0.877 |
| answer-level F1 (does the answer contain an error) | 0.750 | **0.878** | 0.865 | 0.865 |
| character-level F1 (which spans are wrong) | 0.715 | 0.729 | 0.693 | **0.753** |
| real errors judged S (of 37) | 5 | **2** | 4 | 5 |
| items fully correct | 10 / 30 | **19 / 30** | 13 / 30 | 17 / 30 |
| latency per item p50 / p95 | 0.0 / 0.1 s | 7.1 / 18.1 s | **0.2 / 0.3 s** | 1.0 / 1.8 s |
| cost for 30 items (list price) | $0 | $0.45 | **$0.007** | $0.09 |

Paired differences in claim accuracy (resampling by item, 10,000 times; items 29 and 30 share a
source and count as one group):

- OST − minimal_llm: −4.2 points, 95% interval [−13.0, +3.8].
- OST − minimal_llm (jev): +14.1, 95% interval [+3.4, +25.3].
- OST − baseline: +14.1, 95% interval [+2.0, +26.1].

By source length: on the 30 items OST costs 0.07× / 0.10× / 0.27× of sol on short (< 5k chars) /
medium (5k–20k) / long (20k–124k) sources, 0.20× overall, at 1.2 / 1.0 / 1.5 s per item against
sol's 9.2 / 8.5 / 9.2 s; minimal_llm (jev) stays at 0.2 s and under $0.001 per item at
every length. The figure below follows the cost per answer out to the design limit (500k
characters of source), on 300 + 500 ANAH answers (aggregates only); Jev input priced at list,
output is free. Costs per method and per version are in [cost](docs/cost.zh-CN.md).

![Cost per answer by source length](docs/figures/cost_by_source_length.svg)

**Summary.** On our own 30-item set:

- against minimal_llm (gpt-6.1-sol), OST is not distinguishable in accuracy (0.803 vs 0.845; the
  95% interval of the difference, [−13.0, +3.8], contains 0), is faster on every item (p50 1.0 s
  vs 7.1 s) and costs 0.20× as much at list prices;
- against minimal_llm (jev), the two are equal at telling whether an answer contains an error
  (answer-level F1 0.865 each). What OST adds is per-claim verdicts and localization: claim
  accuracy +14.1 points [+3.4, +25.3] and character-level F1 0.753 vs 0.693. minimal_llm (jev)
  gives one verdict for the whole answer, and every gold sentence inherits it when aligned, so
  claim accuracy is unfavourable to it by construction. It is also the fastest and cheapest
  option (0.2 s and $0.0002 per item).

Notes:

- **Cost** is token counts times list prices, not billing. gpt-6.1-sol uses third-party prices
  ($2 / $10 per million input / output tokens); Jev uses its official price ($0.042 per million
  input tokens, output free).
- **Runs**: each column is one run. The OST column is the current default configuration (OP29,
  commit `c78c508`; later commits on `main` do not change its verdicts). Jev's answers vary from
  run to run: against the previous default (the sentence-guard version), 2 of 71 verdicts changed
  (04#4 S→U with the judge's S probability at 0.46 instead of 0.52; 18#1 C→U through a different
  evidence sentence), both on claims that had already flipped between earlier runs. The sol column
  is the prompt with the assignment's definitions; the earlier paraphrased prompt gave the same
  accuracy with one verdict different.
- **The set**: 30 items only give wide intervals for differences between configurations, so no
  winner is claimed above. The set was also designed by us, around our own architecture.
- **Only sol was compared**: with qwen3.8-flash, minimal_llm costs about $0.03 for the 30 items,
  about a third of OST.

**Key failure cases** (the full analysis is in the [failure analysis](docs/failure-analysis.zh-CN.md), in Chinese for now):

- Both model-based methods take nearby evidence as support. In item 11 the source is about a
  different Hubbell; in item 16 two passages disagree, but only the supporting one is cited.
- Two of OST's six own errors come from splitting: in items 14 and 30 two assertions stay in one
  claim and are judged once. The rest are judging errors; in item 18, for example, the
  contradiction needs several passages combined and OST reports U.
- minimal_llm (gpt-6.1-sol) splits finer than the gold, so one unsupported fragment marks the
  whole sentence U (01#4, 13#1).
- minimal_llm (jev) gives one verdict per answer: one error flags every correct sentence with it.
- The baseline is lexical only: paraphrases do not match, and distortions other than numbers and
  negation are missed.

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
