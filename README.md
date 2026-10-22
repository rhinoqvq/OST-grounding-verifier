# grounding-verifier

English | [中文](README.zh-CN.md)

An HTTP service that decomposes an LLM-generated answer into atomic factual
claims, verifies each claim against the provided source passages, and attaches
verbatim citations (`SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`).

**Status: work in progress.** All three methods, the evaluation set and the
benchmark runner are in place (see *Evaluation results* below); the full
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

## Evaluation results (OST30)

The evaluation set is [`benchmark/ost30.jsonl`](benchmark/ost30.jsonl):

- 30 items and 71 gold claims (S 34, U 26, C 11).
- Each line is a `/verify` request; the gold follows the assignment's response format.
- The frozen hash is in [`benchmark/ost30.sha256`](benchmark/ost30.sha256), and the runner
  refuses a file that does not match it.
- Items are adapted from public datasets: ANAH, RAGTruth, FOLIO, MuSiQue, FaithEval, VitaminC
  and SciTab. Four of them pair Wikipedia text with answers written for this set.
- Licenses and per-item attribution are in
  [`benchmark/DATA_LICENSE.md`](benchmark/DATA_LICENSE.md).

All four categories the assignment asks for are covered:

| category | items |
|---|---:|
| fully supported answers | 9 |
| subtle distortions (off-by-one numbers, swapped conditions, inverted negations, …) | 9 |
| extrinsic hallucinations (not in the passages, including facts true in the real world) | 8 |
| multi-passage synthesis | 4 |
| edge case: no checkable facts | 1 |

The three methods:

- **baseline**: no model calls.
- **minimal_llm**: one gpt-6.1-sol call per item.
- **OST (optimized)**: Jev only.

Every method receives only `context_passages` and `generated_answer`; the gold is never sent.

| metric | baseline | minimal_llm (gpt-6.1-sol) | OST (optimized) |
|---|---|---|---|
| claim accuracy (71 claims) | 0.662 | **0.845** | 0.817 |
| 95% interval (item resampling) | 0.545–0.761 | 0.750–0.922 | 0.708–0.903 |
| 3-way macro-F1 | 0.591 | **0.819** | 0.766 |
| C/U alarm F1 | 0.780 | **0.895** | 0.892 |
| real errors judged S (of 37) | 5 | **3** | 4 |
| items fully correct | 10 / 30 | **19 / 30** | 17 / 30 |
| latency per item p50 / p95 | 0.0 / 0.1 s | 7.5 / 19.4 s | **1.8 / 3.5 s** |
| cost for 30 items (list price) | $0 | $0.45 | **$0.33** |

Paired differences (resampling by item; items 29 and 30 share a source and count as one group):

- OST − minimal_llm: −2.8 points, 95% interval [−11.3, +5.0].
- OST − baseline: +15.5, 95% interval [+4.7, +26.7].

By source length:

| source length | items | sol mean latency | OST (optimized) mean latency | sol mean cost | OST (optimized) mean cost | OST (optimized) cost ÷ sol cost |
|---|---:|---:|---:|---:|---:|---:|
| short (< 5k chars) | 10 | 9.8 s | 1.1 s | $0.0052 | $0.0009 | 0.17 |
| medium (5k–20k) | 12 | 9.8 s | 1.9 s | $0.0103 | $0.0049 | 0.48 |
| long (20k–124k) | 8 | 10.8 s | 3.1 s | $0.0342 | $0.0330 | 0.97 |
| total | 30 | — | — | $0.448 | $0.332 | 0.74 |

**Summary.** On our own 30-item set, compared with a single gpt-6.1-sol call, OST:

- is not distinguishable in accuracy: 0.817 vs 0.845, and the 95% interval of the difference,
  [−11.3, +5.0], contains 0;
- is faster on every item: p50 1.8 s vs 7.5 s;
- costs about a quarter less at list prices: about 1/6 and 1/2 of sol on short and medium
  sources, and about the same on sources over 20k characters.

Notes:

- **Cost** is token counts times list prices, not billing. gpt-6.1-sol uses third-party prices
  ($2 / $10 per million input / output tokens); Jev uses its official price ($0.042 per million
  input tokens, output free).
- **Runs**: gpt-6.1-sol was run once. OST was run twice, and 1 of 71 verdicts differed between
  the two runs.
- **The set**: 30 items only give wide intervals for differences between configurations, so no
  winner is claimed above. The set was also designed by us, around our own architecture.
- **Only sol was compared**: with qwen3.8-flash, minimal_llm costs about $0.03 for the 30 items,
  1/11 of OST.

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
- minimal_llm: `OPENAI_API_KEY` (gpt-6.1-sol is the default model).
- OST (optimized): `TYPESAFE_API_KEY`.

```bash
.venv/Scripts/python benchmark/run_ost30.py --method baseline
.venv/Scripts/python benchmark/run_ost30.py --method minimal_llm
.venv/Scripts/python benchmark/run_ost30.py --method optimized
.venv/Scripts/python benchmark/score_ost30.py baseline minimal_llm optimized
.venv/Scripts/python benchmark/cost_latency_ost30.py minimal_llm optimized
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
