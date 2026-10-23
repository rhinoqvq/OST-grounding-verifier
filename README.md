# grounding-verifier

English | [中文](README.zh-CN.md)

An HTTP service that decomposes an LLM-generated answer into atomic factual
claims, verifies each claim against the provided source passages, and attaches
verbatim citations (`SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`).

**Status: work in progress.** All three methods, the evaluation set and the
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
| claim accuracy (71 claims) | 0.662 | **0.845** | 0.831 |
| 95% interval (item resampling) | 0.545–0.761 | 0.750–0.922 | 0.702–0.925 |
| 3-way macro-F1 | 0.591 | **0.819** | 0.798 |
| C/U alarm F1 | 0.780 | **0.895** | 0.889 |
| real errors judged S (of 37) | 5 | **3** | 5 |
| items fully correct | 10 / 30 | **19 / 30** | **19 / 30** |
| latency per item p50 / p95 | 0.0 / 0.1 s | 7.5 / 19.4 s | **1.9 / 3.7 s** |
| cost for 30 items (list price) | $0 | $0.45 | **$0.34** |

Paired differences (resampling by item; items 29 and 30 share a source and count as one group):

- OST − minimal_llm: −1.4 points, 95% interval [−10.2, +6.7].
- OST − baseline: +16.9, 95% interval [+4.5, +29.3].

By source length:

| source length | items | sol mean latency | OST (optimized) mean latency | sol mean cost | OST (optimized) mean cost | OST (optimized) cost ÷ sol cost |
|---|---:|---:|---:|---:|---:|---:|
| short (< 5k chars) | 10 | 9.8 s | 1.4 s | $0.0052 | $0.0009 | 0.18 |
| medium (5k–20k) | 12 | 9.8 s | 2.1 s | $0.0103 | $0.0053 | 0.52 |
| long (20k–124k) | 8 | 10.8 s | 3.5 s | $0.0342 | $0.0339 | 0.99 |
| total | 30 | — | — | $0.448 | $0.344 | 0.77 |

**Summary.** On our own 30-item set, compared with a single gpt-6.1-sol call, OST:

- is not distinguishable in accuracy: 0.831 vs 0.845, and the 95% interval of the difference,
  [−10.2, +6.7], contains 0;
- is faster on every item: p50 1.9 s vs 7.5 s;
- costs 23% less at list prices: 0.18× and 0.52× of sol on short and medium sources, and about
  the same (0.99×) on sources over 20k characters.

Notes:

- **Cost** is token counts times list prices, not billing. gpt-6.1-sol uses third-party prices
  ($2 / $10 per million input / output tokens); Jev uses its official price ($0.042 per million
  input tokens, output free).
- **Runs**: gpt-6.1-sol was run once. OST's column is one run of the current default
  configuration, sentence guard included (it released no unit on this set). Replaying the Jev
  calls recorded in earlier runs through the same code changes 3 of 71 verdicts (04#4, 07#2,
  18#1) and gives an accuracy of 0.803: Jev's answers vary from run to run.
- **The set**: 30 items only give wide intervals for differences between configurations, so no
  winner is claimed above. The set was also designed by us, around our own architecture.
- **Only sol was compared**: with qwen3.8-flash, minimal_llm costs about $0.03 for the 30 items,
  under 1/11 of OST.

## Failure analysis

Item numbers are the `id`s in `benchmark/ost30.jsonl`; every case was checked against the source
text.

**Both model-based methods**

- Nearby evidence taken as support:
  - 11: the source's "Mr. Hubbell married … Lina Rubi" is about a different Hubbell; both methods
    took her for John's wife in the answer.
  - 16: the two passages disagree on who designed the museum (Semper / Mies van der Rohe); both
    methods cited only the supporting passage and missed the conflict.
  - 17: the answer lists six fields Plato discussed; both methods found a passage for each and
    judged it supported. "Psychology" is not stated in the source but inferred from a passage on
    the three parts of the soul; the gold is U.
- A refusal treated as a claim (05): the answer is only "Unable to answer based on given
  passages", and all three methods output one U claim.
- Reading disagreements (07#2, 15#3, 24#1): both methods differ from the gold on how to read the
  source or the answer. Example, 07#2 "only after his death were his works published":
  - the source says a few works were published in his lifetime, so sol judges C, and OST's judge
    leans C too (its strict contradiction rule reports it as U);
  - the gold reads "his works" as most of them and judges S.

  These are scored as errors.

**OST only**

- Two assertions kept as one claim:
  - 14, "The creator of A Burial at Ornans is Gustave Courbet, who was born on 10 June 1819": the
    relative clause was not cut, so the whole sentence was judged U although its first half is
    supported.
  - 30: sentence splitting did not break after "Numbers 20:22-29.", so two sentences became one
    and the second one's C was not judged on its own.
- A weak lean kept as supported (24#3): "The proposed approach is seen to perform well against
  the other unsupervised models." The judge leaned S at p=0.51, below the 0.55 bar, and a lean to
  S under the bar stays S (on the RAGTruth test such leans were clean 430 times in 456). The
  source is a table of correlation scores and does not say which models are unsupervised; the
  gold (SciTab: not enough info) is U.
- A dropped condition judged supported (25): the source says "in this case [this paradigm], the
  domain model is presented as a set of constraints"; the answer replaces the condition with "in
  an intelligent tutoring system", and OST judged S.
- Evidence over-read (28#3): for "Archduke Charles resigned his command shortly after the battle",
  OST cited "After the battle, Charles remained in command of a cohesive force and decided to
  retreat to Bohemia" and judged C. That sentence only says he was still in command right after
  the battle, not that he never resigned later; the gold is U.

**Method 2 (gpt-6.1-sol) only**

- A strengthened modal judged a contradiction (10): the source says "may include", the answer
  "always include". The source does not state "not always", so the gold is U; sol judged C.
- Splitting finer than the gold (01#4, 04#4, 13#1): sol splits a sentence into several small
  assertions, and if one of them has no support the whole sentence counts as U.
  - 13#1 yields "The Emancipation Proclamation was an executive order", which the source does not
    say; the other parts are S.
  - 04#4 yields "The wider network of stop lines extended across the country", which sol finds
    the source does not establish.
  - The gold judges each sentence as a whole, S.

  These three are also reading disagreements.

**Baseline**

- Lexical only: paraphrases and aliases do not match. 01 gets two of five sentences right, and
  both sentences of 13 are judged U.
- Distortions other than numbers and negation are missed, and both of these are judged S:
  - 09: the answer's 1928 does occur in the source, but for a different event;
  - 22: the answer says "all the reviews were favorable", the source "mixed to positive".
- Over-wide flagging: the characters it marks C / U are 1.65× the gold error spans.

**Prompt injection (20).** The source has an instruction disguised as a system message, telling
the checker to mark two true sentences CONTRADICTED and the unsupported one SUPPORTED. None of
the three methods followed it. For comparison, method 2 on qwen3.8-flash followed it and marked
the two true sentences C.

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
