# grounding-verifier

English | [中文](README.zh-CN.md)

An HTTP service that decomposes an LLM-generated answer into atomic factual
claims, verifies each claim against the provided source passages, and attaches
verbatim citations (`SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`).

**Status: work in progress.** All four methods, the evaluation set and the
benchmark runner are in place (see *Evaluation results on the accompanying evaluation set* below); the full
evaluation report is still being written.

## Design in one paragraph

Source passages and the answer are stored immutably; everything downstream
points at character offsets, so every quote is sliced from the original text by
code and is verbatim by construction. Code finds candidate cut points in each
answer sentence and Jev — a calibrated decision model — decides each one, so
every claim unit is a verbatim span of the answer (no text is generated, so
nothing can quietly "fix" the answer); the sources are decomposed the same way,
and Jev answers the verdict/evidence questions; plain
code enforces everything deterministic: offsets, coverage, number/negation
sanity, verdict aggregation over the semantic tree, and a final response
contract (citation indexes in bounds, quotes verbatim, citations consistent
with verdicts). A claim the system is not confident about is reported as
`UNSUPPORTED` with the reason, never guessed.

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

The four methods:

- **baseline**: no model calls.
- **single-call Jev**: one Jev request per item — all passages and the whole answer, asked whether
  the answer is SUPPORTED, CONTRADICTED or UNSUPPORTED and which passage is the evidence; the
  answer is reported as one claim.
- **minimal_llm**: one gpt-6.1-sol call per item, with the three verdict definitions from the
  assignment word for word.
- **OST (optimized, OP29)**: Jev only.

Every method receives only `context_passages` and `generated_answer`; the gold is never sent.

| metric | baseline | single-call Jev | minimal_llm (gpt-6.1-sol) | OST (optimized) |
|---|---|---|---|---|
| claim accuracy (71 claims) | 0.662 | 0.662 | **0.845** | 0.803 |
| 95% interval (item resampling) | 0.545–0.761 | 0.523–0.784 | 0.750–0.922 | 0.672–0.898 |
| 3-way macro-F1 | 0.591 | 0.630 | **0.819** | 0.760 |
| C/U alarm F1 | 0.780 | 0.815 | **0.909** | 0.877 |
| real errors judged S (of 37) | 5 | 4 | **2** | 5 |
| items fully correct | 10 / 30 | 13 / 30 | **19 / 30** | 17 / 30 |
| latency per item p50 / p95 | 0.0 / 0.1 s | **0.2 / 0.3 s** | 7.1 / 18.1 s | 1.0 / 1.8 s |
| cost for 30 items (list price) | $0 | **$0.007** | $0.45 | $0.09 |

Paired differences in claim accuracy (resampling by item, 10,000 times; items 29 and 30 share a
source and count as one group):

- OST − minimal_llm: −4.2 points, 95% interval [−13.0, +3.8].
- OST − single-call Jev: +14.1, 95% interval [+3.4, +25.3].
- OST − baseline: +14.1, 95% interval [+2.0, +26.1].

By source length: on the 30 items OST costs 0.07× / 0.10× / 0.27× of sol on short (< 5k chars) /
medium (5k–20k) / long (20k–124k) sources, 0.20× overall, at 1.2 / 1.0 / 1.5 s per item against
sol's 9.2 / 8.5 / 9.2 s; the single-call Jev baseline stays at 0.2 s and under $0.001 per item at
every length. The figure below follows the cost per answer out to the design limit (500k
characters of source), on 300 + 500 ANAH answers (aggregates only); Jev input priced at list,
output is free.

![Cost per answer by source length](docs/figures/cost_by_source_length.svg)

**Summary.** On our own 30-item set:

- against a single gpt-6.1-sol call, OST is not distinguishable in accuracy (0.803 vs 0.845; the
  95% interval of the difference, [−13.0, +3.8], contains 0), is faster on every item (p50 1.0 s
  vs 7.1 s) and costs 0.20× as much at list prices;
- against a single Jev call, OST is +14.1 points in claim accuracy [+3.4, +25.3]: one request per
  item is the fastest and cheapest option, but it gives one verdict for the whole answer, so on
  claim accuracy it is no better than the no-model baseline (0.662), and it cannot say which
  sentence is wrong.

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
- single-call Jev (`minimal_jev`): `TYPESAFE_API_KEY`.
- minimal_llm: `OPENAI_API_KEY` (gpt-6.1-sol is the default model).
- OST (optimized): `TYPESAFE_API_KEY`.

```bash
.venv/Scripts/python benchmark/run_ost30.py --method baseline
.venv/Scripts/python benchmark/run_ost30.py --method minimal_jev
.venv/Scripts/python benchmark/run_ost30.py --method minimal_llm
.venv/Scripts/python benchmark/run_ost30.py --method optimized
.venv/Scripts/python benchmark/score_ost30.py baseline minimal_jev minimal_llm optimized
.venv/Scripts/python benchmark/cost_latency_ost30.py minimal_jev minimal_llm optimized
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
