"""Batch independent questions into as few requests as possible.

Latency is governed by the number of *dependent rounds*, not the number of
questions: everything inside one round shares a state and is independent, so it
goes out as one request (or a few, capped by max_questions_per_request, sent
concurrently).
"""

from __future__ import annotations

import asyncio
import logging

from verifier.harness.context import estimate_tokens
from verifier.models.decider import Answer, ChoiceQuestion, Decider, Question

log = logging.getLogger(__name__)

# Jev refuses a request above about 32k tokens (state plus every question in
# it); the estimate is tiktoken, so keep headroom for the tokenizer gap.
DEFAULT_REQUEST_TOKEN_BUDGET = 28_000
# Jev also refuses a single question above a size of its own (max_tokens_exceeded on
# one ~20k-token question while 20 questions totalling ~17k went through; TRIVIA+ valid,
# 2026-10-23: a source "sentence" of thousands of characters, a table with no full stop,
# repeated in every option of a cut question). Such a question is not sent and gets no
# answer: every optimized caller reads answers with .get and falls back (a cut not taken).
MAX_QUESTION_TOKENS = 12_000


def question_tokens(question: Question) -> int:
    text = question.instructions
    if isinstance(question, ChoiceQuestion):
        text += " ".join(f"{k} {v}" for k, v in question.options.items())
    return estimate_tokens(text)


def plan_chunks(
    state: str, questions: list[Question], *, max_per_request: int, max_tokens: int
) -> list[list[Question]]:
    """Consecutive chunks of at most `max_per_request` questions whose estimated
    tokens, together with the state, stay within `max_tokens`. A question that
    alone exceeds the budget still gets a chunk of its own."""
    budget = max_tokens - estimate_tokens(state)
    chunks: list[list[Question]] = []
    current: list[Question] = []
    used = 0
    for q in questions:
        cost = question_tokens(q)
        if current and (len(current) >= max_per_request or used + cost > budget):
            chunks.append(current)
            current, used = [], 0
        current.append(q)
        used += cost
    if current:
        chunks.append(current)
    return chunks


async def ask_batched(
    decider: Decider,
    state: str,
    questions: list[Question],
    *,
    max_per_request: int,
    max_parallel: int,
    max_tokens: int = DEFAULT_REQUEST_TOKEN_BUDGET,
) -> dict[str, Answer]:
    oversized = [q for q in questions if question_tokens(q) > MAX_QUESTION_TOKENS]
    if oversized:
        log.warning("not asking %d oversized question(s): %s", len(oversized),
                    ", ".join(q.question_id for q in oversized[:5]))
        questions = [q for q in questions if question_tokens(q) <= MAX_QUESTION_TOKENS]
    if not questions:
        return {}
    chunks = plan_chunks(state, questions, max_per_request=max_per_request, max_tokens=max_tokens)
    semaphore = asyncio.Semaphore(max_parallel)

    async def run(chunk: list[Question]) -> dict[str, Answer]:
        async with semaphore:
            return await decider.ask(state, chunk)

    results = await asyncio.gather(*(run(c) for c in chunks))
    merged: dict[str, Answer] = {}
    for part in results:
        merged.update(part)
    return merged
