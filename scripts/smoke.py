"""Real-API smoke test: verifies both keys work, then runs the assignment's
WAL example through the full pipeline against the live models.

Usage:
    .venv/Scripts/python scripts/smoke.py           # needs .env with both keys

Steps:
  1. Jev: one tiny choice question           -> reports model version, latency
  2. OpenAI: one tiny structured output      -> reports model, token usage
  3. POST-/verify-equivalent run of the assignment example (cache_mode=record,
     so the raw responses land in var/verifier.sqlite3 for replay)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel

from verifier.harness.config import Settings
from verifier.methods.optimized.pipeline import run_verification
from verifier.output.schemas import VerifyRequest
from verifier.service import build_dependencies

PASSAGE_0 = (
    "Beginning with version 3.7.0, Write-Ahead Log (WAL) mode is available. "
    "Advantages include significantly faster performance in most scenarios and increased "
    "concurrency: readers do not block writers and writers do not block readers. However, "
    "WAL does not work over a network filesystem because it requires all processes to share "
    "a small amount of memory via shared memory files (-shm)."
)
PASSAGE_1 = (
    "When in active use, the state of a WAL mode database is described by three separate "
    "files: the main database file 'X', the write-ahead log file 'X-wal', and the wal-index "
    "shared-memory file 'X-shm'. Automatic checkpointing occurs by default when the WAL "
    "passes approximately 1000 pages."
)
ANSWER = (
    "SQLite WAL mode allows multiple processes across different host machines on an NFS "
    "network share to write concurrently. Alongside the main database file, it uses two "
    "auxiliary files (-wal and -shm) and automatically checkpoints when the log reaches "
    "roughly 1,000 pages. By default, WAL files are encrypted with AES-256 at rest."
)


async def smoke_jev(deps) -> None:
    from verifier.models.decider import ChoiceQuestion

    print("== 1. Jev smoke ==")
    started = time.perf_counter()
    answers = await deps.decider.ask(
        "The sky on a clear day.",
        [
            ChoiceQuestion(
                question_id="q1",
                instructions="What color is described?",
                options={"blue": "The state describes blue.", "green": "The state describes green."},
            )
        ],
    )
    ms = (time.perf_counter() - started) * 1000
    a = answers["q1"]
    print(f"   choice={a.choice} probabilities={a.probabilities} confidence={a.confidence}")
    print(f"   reported model: {deps.decider.reported_model}, latency {ms:.0f} ms")
    assert a.choice == "blue", "Jev answered the sanity question wrong?!"


async def smoke_llm(deps) -> None:
    class Tiny(BaseModel):
        color: str
        word_count: int

    print(f"== 2. {deps.llm.name} structured-output smoke ==")
    started = time.perf_counter()
    result = await deps.llm.generate_structured(
        [
            {"role": "system", "content": "Answer as JSON matching the schema."},
            {"role": "user", "content": "The sky is blue. Report the color and the number of words in that sentence."},
        ],
        Tiny,
    )
    ms = (time.perf_counter() - started) * 1000
    print(f"   parsed: {result.model_dump()}  latency {ms:.0f} ms")
    print(f"   provider: {deps.llm.name}, model: {deps.llm.model}, "
          f"usage: in={deps.llm.usage.input_tokens} out={deps.llm.usage.output_tokens}")


async def smoke_pipeline(deps) -> None:
    print("== 3. Full pipeline on the assignment WAL example ==")
    request = VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)
    started = time.perf_counter()
    response, trace = await run_verification(request, deps)
    ms = (time.perf_counter() - started) * 1000
    for i, claim in enumerate(response.claims):
        cites = [f"[{c.index}] {c.quote[:60]}..." for c in claim.citations]
        print(f"   claim[{i}] {claim.verdict.value}: {claim.claim_text[:80]}")
        for c in cites:
            print(f"      cite {c}")
        print(f"      reasoning: {claim.reasoning[:110]}")
    print(f"   total {ms:.0f} ms; rounds:")
    for r in trace.as_dict()["rounds"]:
        print(f"      {r['name']}: {r['duration_ms']:.0f} ms" + (f"  ERROR {r['error']}" if r["error"] else ""))
    print(f"   jev usage: calls={deps.decider.usage.calls} in={deps.decider.usage.input_tokens} out={deps.decider.usage.output_tokens}")
    print(f"   openai usage: calls={deps.llm.usage.calls} in={deps.llm.usage.input_tokens} out={deps.llm.usage.output_tokens}")


async def main() -> None:
    settings = Settings(cache_mode="record")
    llm_key = settings.openai_api_key if settings.llm_provider == "openai" else settings.dashscope_api_key
    llm_key_name = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "DASHSCOPE_API_KEY"
    missing = [
        name
        for name, value in ((llm_key_name, llm_key), ("TYPESAFE_API_KEY", settings.typesafe_api_key))
        if not value
    ]
    if missing:
        print(f"missing keys in .env: {missing} (llm_provider={settings.llm_provider})")
        raise SystemExit(1)
    deps = build_dependencies(settings)
    try:
        await smoke_jev(deps)
        await smoke_llm(deps)
        await smoke_pipeline(deps)
        print("SMOKE OK")
    finally:
        if hasattr(deps.decider, "aclose"):
            await deps.decider.aclose()
        if deps.conn is not None:
            deps.conn.close()


if __name__ == "__main__":
    asyncio.run(main())
