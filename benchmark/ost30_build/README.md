# Building OST30

These scripts rebuild [`../ost30.jsonl`](../ost30.jsonl) from the public source datasets, byte for
byte: the result hashes to the frozen value in [`../ost30.sha256`](../ost30.sha256). The
selection rules, seeds, skip reasons and gold changes are written out in
[`docs/appendix/ost30-extraction.zh-CN.md`](../../docs/appendix/ost30-extraction.zh-CN.md)
(in Chinese for now). No model is called.

## Source data

Put these under `var/` in the repository root:

| file | source | used for |
|---|---|---|
| `var/anah/anah_all.json` | ANAH ([opencompass/anah](https://huggingface.co/datasets/opencompass/anah), Apache-2.0): every row of the dataset (783 topics) as one JSON list, with the columns `name`, `language`, `documents`, `selected_questions`, `GPT3.5_answers_D`, `InternLM_answers`, `human_GPT3.5_answers_D_ann`, `human_InternLM_answers_ann`. Fetched through the Hugging Face dataset viewer. | 18 ANAH items, and the Wikipedia text of items 10, 20, 25 and 28 |
| `var/ragtruth/response.jsonl`, `var/ragtruth/source_info.jsonl` | RAGTruth ([ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth), MIT), the `dataset/` folder | item 05 |
| `var/vitaminc/test.jsonl` | VitaminC ([tals/vitaminc](https://huggingface.co/datasets/tals/vitaminc), CC BY-SA 3.0), test split | item 22 |
| `var/scitab/sci_tab.json` | SciTab ([XinyuanLu00/SciTab](https://github.com/XinyuanLu00/SciTab), MIT) | item 24 |

The FOLIO, MuSiQue and FaithEval rows used for items 06, 14, 16, 18 and 19 are already in
[`ost30_external_rows.json`](ost30_external_rows.json).

ANAH's documents include Encyclopaedia Britannica text, which is copyrighted. The scripts drop
those documents; nothing from them reaches `ost30.jsonl`.

## Run

```bash
.venv/Scripts/python benchmark/ost30_build/pick_ost30_anah.py var/anah/picks.json benchmark/ost30_build/pick_ost30_anah_skip.json
.venv/Scripts/python benchmark/ost30_build/build_ost30_anah.py var/anah/picks.json var/ost30_build
.venv/Scripts/python benchmark/ost30_build/add_external_ost30.py var/ost30_build/ost30.jsonl
.venv/Scripts/python benchmark/ost30_build/gen_ost30_synth.py var/ost30_build/ost30.jsonl
.venv/Scripts/python benchmark/ost30_build/enrich_ost30.py var/ost30_build/ost30.jsonl var/anah/picks.json
.venv/Scripts/python benchmark/ost30_build/check_ost30.py var/ost30_build/ost30.jsonl
```

Then compare `var/ost30_build/ost30.jsonl` with `benchmark/ost30.jsonl` (same sha256).

| script | what it does |
|---|---|
| `pick_ost30_anah.py` | filters ANAH answers per question type, orders candidates by seed `20261022-<type>`, takes the first one that passes; skipped candidates and their reasons are in `pick_ost30_anah_skip.json` |
| `build_ost30_anah.py` | writes the 18 ANAH items with gold from ANAH's sentence labels, citations repaired to verbatim substrings |
| `add_external_ost30.py` | adds the 8 items from RAGTruth, FOLIO, MuSiQue, FaithEval, VitaminC and SciTab |
| `gen_ost30_synth.py` | adds items 10, 20, 25 and 28: real Wikipedia text, answers written by a fixed edit, gold known by construction |
| `enrich_ost30.py` | adds the derived fields: answer spans, error spans, source hash, family id |
| `check_ost30.py` | checks every record against the request format and the response invariants |

The gold edits made during the manual review are part of these scripts, so the rebuild matches
the reviewed, frozen file.
