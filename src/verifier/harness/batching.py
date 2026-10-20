"""Batch independent questions into as few requests as possible.

Latency is governed by the number of *dependent rounds*, not the number of
questions: everything inside one round shares a state and is independent, so it
goes out as one request (or a few, capped by max_questions_per_request, sent
concurrently).
"""

from __future__ import annotations

import asyncio

from verifier.models.decider import Answer, Decider, Question


async def ask_batched(
    decider: Decider,
    state: str,
    questions: list[Question],
    *,
    max_per_request: int,
    max_parallel: int,
) -> dict[str, Answer]:
    if not questions:
        return {}
    chunks = [
        questions[i : i + max_per_request] for i in range(0, len(questions), max_per_request)
    ]
    semaphore = asyncio.Semaphore(max_parallel)

    async def run(chunk: list[Question]) -> dict[str, Answer]:
        async with semaphore:
            return await decider.ask(state, chunk)

    results = await asyncio.gather(*(run(c) for c in chunks))
    merged: dict[str, Answer] = {}
    for part in results:
        merged.update(part)
    return merged
