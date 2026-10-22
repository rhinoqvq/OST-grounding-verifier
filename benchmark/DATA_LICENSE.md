# OST30 data: licenses and attribution

`ost30.jsonl` (30 items, sha256 in `ost30.sha256`) is released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). ShareAlike is required because
the passages include Wikipedia, FOLIO and VitaminC text. The scripts in this directory are covered by
the repository's MIT license.

## Sources

| Source | License / terms | Items |
|---|---|---|
| Wikipedia text (ANAH, MuSiQue and SQuAD passages) | CC BY-SA | ANAH items, 10, 14, 16, 19, 20, 25, 28 |
| [ANAH](https://huggingface.co/datasets/opencompass/anah) annotations and answers | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | 18 ANAH items |
| Encyclopaedia Britannica text (part of ANAH's documents) | copyrighted, **removed** | — |
| [FOLIO](https://github.com/Yale-LILY/FOLIO) | CC BY-SA 4.0 | 06, 18 |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | CC BY 4.0 | 14, 19 |
| [FaithEval](https://huggingface.co/datasets/Salesforce/FaithEval-inconsistent-v1.0) | Apache-2.0 ([text](licenses/Apache-2.0.txt)); passage from SQuAD (Wikipedia, CC BY-SA) | 16 |
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | MIT, Copyright (c) 2023 Particle Media ([text](licenses/MIT-RAGTruth.txt)) | 05 |
| [MS MARCO](https://microsoft.github.io/msmarco/) passages inside RAGTruth | non-commercial research use only | 05 |
| [VitaminC](https://huggingface.co/datasets/tals/vitaminc) | CC BY-SA 3.0; carries the BIG-bench canary string (asks that the data not be used for training), kept in the item | 22 |
| [SciTab](https://github.com/XinyuanLu00/SciTab) | MIT, Copyright (c) 2023 Lu Xinyuan ([text](licenses/MIT-SciTab.txt)); table values from the cited arXiv paper | 24 |

Items 10, 20, 25 and 28 pair verbatim Wikipedia text from ANAH's documents with answers written
for this set (`source.synthetic_answer: true`). Item 20 also has an injected instruction that was
written for this set. Each item's `source` field records its dataset, identifiers and the changes
made (`transform`).

Item 05's passages come from MS MARCO and may be used for non-commercial research only, whatever
license the rest of the set carries.

## Per-item attribution

| Item | Source | Identifier | License / terms |
|---|---|---|---|
| OST30-01 | ANAH | `Battle of Sobraon#gpt35_with_docs#0` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-02 | ANAH | `Ambepussa#internlm_closed_book#0` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-03 | ANAH | `21 Kislev#internlm_closed_book#0` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-04 | ANAH | `Tyne Stop Line#internlm_closed_book#0` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-05 | RAGTruth | response 16298, source 12208 | MIT; passages from MS MARCO, non-commercial research only |
| OST30-06 | FOLIO | example 548, story 192 | CC BY-SA 4.0 |
| OST30-07 | ANAH | `Franz Kafka#internlm_closed_book#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-08 | ANAH | `Galen#gpt35_with_docs#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-09 | ANAH | `Titelberg#internlm_closed_book#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-10 | ANAH document (Wikipedia) + synthetic answer | topic "Psychosis", document 1 | Wikipedia text CC BY-SA; answer written for this set |
| OST30-11 | ANAH | `Hubbell Trading Post National Historic Site#internlm_closed_book#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-12 | ANAH | `Herodotus#gpt35_with_docs#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-13 | ANAH | `Abraham Lincoln#internlm_closed_book#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-14 | MuSiQue | `2hop__107905_110222` | CC BY 4.0; paragraphs from Wikipedia, CC BY-SA |
| OST30-15 | ANAH | `Stanley Baldwin#gpt35_with_docs#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-16 | FaithEval (inconsistent) | qid `10853963981b49d7b8fd18bdcab80e5c` | Apache-2.0; passage from SQuAD (Wikipedia), CC BY-SA |
| OST30-17 | ANAH | `Plato#internlm_closed_book#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-18 | FOLIO | example 165, story 56 | CC BY-SA 4.0 |
| OST30-19 | MuSiQue | `2hop__781079_9237` | CC BY 4.0; paragraphs from Wikipedia, CC BY-SA |
| OST30-20 | ANAH document (Wikipedia) + synthetic answer | topic "Alessandro Volta", document 1 | Wikipedia text CC BY-SA; answer written for this set |
| OST30-21 | ANAH | `Byzantine diplomacy#internlm_closed_book#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-22 | VitaminC | `5ebe6f30c9e77c0008408a82_3` (page "Dolphin Tale 2") | CC BY-SA 3.0; BIG-bench canary kept |
| OST30-23 | ANAH | `Moses#gpt35_with_docs#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-24 | SciTab | table `1551d88a-ea00-4a48-a576-e3535dd3b391`, paper arXiv 1807.07279v3 | MIT |
| OST30-25 | ANAH document (Wikipedia) + synthetic answer | topic "Intelligent tutoring system", document 0 | Wikipedia text CC BY-SA; answer written for this set |
| OST30-26 | ANAH | `History of Kyiv#internlm_closed_book#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-27 | ANAH | `Saugus, Massachusetts#internlm_closed_book#1` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-28 | ANAH document (Wikipedia) + synthetic answer | topic "Battle of Wagram", document 1 | Wikipedia text CC BY-SA; answer written for this set |
| OST30-29 | ANAH | `Aaron#gpt35_with_docs#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
| OST30-30 | ANAH | `Aaron#internlm_closed_book#2` | Apache-2.0 (annotations, answer); Wikipedia text CC BY-SA |
