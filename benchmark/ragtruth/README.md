# RAGTruth runs (development set)

Everything needed to rescore the RAGTruth numbers in the docs, and to regenerate them.
RAGTruth (2,700 test responses, 900 each of QA, Summary, Data2txt) is the development set:
the system was tuned on it, so absolute scores are optimistic; the docs use it to compare
configurations.

## Data

Get `response.jsonl` and `source_info.jsonl` from the `dataset/` folder of the RAGTruth
repository (ParticleMedia/RAGTruth, MIT) and put them in `var/ragtruth/` (or point
`GV_RAGTRUTH_DIR` at them). Then:

```bash
.venv/Scripts/python benchmark/ragtruth/build_inputs.py
```

This writes `var/ragtruth_full/inputs.jsonl` (what the verifier sees, no labels) and
`var/ragtruth_full/gold.jsonl` (RAGTruth's own hallucination spans). The passage rules are in
the script's docstring; they were fixed before any full run, and the two files rebuild byte for
byte.

## Rescore from the recorded predictions

`preds/` holds the recorded predictions of every run the docs cite, one record per response.
Rescoring needs no API key and no model call:

```bash
.venv/Scripts/python benchmark/ragtruth/paired.py --a benchmark/ragtruth/preds/op29_default.jsonl.gz --b benchmark/ragtruth/preds/minimal_llm_jev.jsonl.gz --names OP29 jev
```

`paired.py` prints answer-level and character-level F1 per task and overall, with the paired
difference and its 95% interval (bootstrap within each task, 10,000 resamples, seed 20261022).
A failed request counts as the wrong answer.

| file | run | answer-level / character-level F1 |
|---|---|---|
| `minimal_llm_jev` | minimal_llm (jev): the whole answer in one Jev request | 77.65 / 14.96 |
| `ablation_1_sentence_level` | one Jev request per answer sentence | 66.27 / 27.40 |
| `ablation_2_meta_discourse_gate` | + meta-discourse gate | 67.78 / 27.90 |
| `ablation_3_evidence_finding` | + OST's evidence finding | 68.33 / 28.37 |
| `ablation_4_whole_sentence_no_tree` | whole-sentence OST, source tree built but not mapped onto | 68.77 / 29.52 |
| `ablation_5_whole_sentence` | whole-sentence OST | 68.95 / 29.55 |
| `op27` | OP27, the full OST before the ablations | 68.92 / 31.09 |
| `r1_guard_all_recheck_off` | R1*: guard extended to whole-sentence U/C, parent recheck off | 69.80 / 31.99 |
| `r2_sentence_level_first` | R1* + sentence-level judging first | 69.77 / 31.94 |
| `op29_tree_on` | OP29 with the source tree always on | 69.77 / 31.94 |
| `op29_default` | OP29, default configuration (no source tree on RAGTruth) | 69.63 / 31.84 |

`wall_ms` in these records is not a latency measurement: the runs replayed cached calls.

## Regenerate

`run.py` runs any method of the current code over the inputs (Jev key required for every
method except `baseline`):

```bash
.venv/Scripts/python benchmark/ragtruth/run.py --method optimized
```

Method and switches come from the environment like the service. With the current code:

| run | settings |
|---|---|
| `minimal_llm_jev` | `--method minimal_jev` |
| `ablation_1` / `_2` / `_3` | `--method minimal_jev_seg` / `minimal_jev_nc` / `minimal_jev_me` |
| `op29_default` | `--method optimized` |
| `op29_tree_on` | `GV_SOURCE_TREE=on` |
| `r2_sentence_level_first` | `GV_SOURCE_TREE=on GV_CITATION_POLICY=inherit` |
| `r1_guard_all_recheck_off` | `GV_SOURCE_TREE=on GV_TWO_STAGE=0 GV_CITATION_POLICY=inherit` |

`op27` was produced by commit `7f3f54b`; the two whole-sentence arms by switches of that time
that are no longer in the code. Jev's answers vary a little from run to run, so a regenerated
run matches the recorded one closely, not byte for byte. Costs in the docs come from the call
ledgers of these runs, which are too large for the repository.

## Files

| file | what it does |
|---|---|
| `tasks.py` | loads the RAGTruth test split and splits sources into passages |
| `build_inputs.py` | writes `inputs.jsonl` and `gold.jsonl` |
| `run.py` | runs a method over `inputs.jsonl`, one prediction record per response |
| `spans.py` | locates flagged claims in the response for character-level scoring |
| `paired.py` | scores two runs and their paired difference |
