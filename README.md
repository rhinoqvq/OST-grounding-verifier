# grounding-verifier

An HTTP service that decomposes an LLM-generated answer into atomic factual
claims, verifies each claim against the provided source passages, and attaches
verbatim citations (`SUPPORTED` / `CONTRADICTED` / `UNSUPPORTED`).

**Status: work in progress.** The optimized verifier pipeline is implemented;
the evaluation dataset, benchmark runner and the two comparison approaches
(non-LLM baseline, minimal-prompt LLM) are coming next, and this README will
grow a full evaluation report.

## Design in one paragraph

Source passages and the answer are stored immutably; everything downstream
points at character offsets, so every quote is sliced from the original text by
code and is verbatim by construction. A generative LLM proposes how the answer
splits into minimal claim units (it never sees the sources, so it cannot
quietly "fix" the answer); Jev — a calibrated decision model — checks each
proposed unit for fidelity and answers the verdict/evidence questions; plain
code enforces everything deterministic: offsets, coverage, number/negation
sanity, verdict aggregation over the semantic tree, and a final response
contract (citation indexes in bounds, quotes verbatim, citations consistent
with verdicts). A claim the system is not confident about is reported as
`UNSUPPORTED` with the reason, never guessed.

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
