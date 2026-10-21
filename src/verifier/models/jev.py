"""Jev (TypeSafe) implementation of the Decider interface.

Choice questions map to Jev `Choice` (label -> description criteria); yes/no
questions map to `Noul`. All questions in one `system_one` call are judged
independently against the same state, which is exactly our batching model.
"""

from __future__ import annotations

from typing import Any

from verifier.models.cache import ModelCallCache
from verifier.models.decider import Answer, ChoiceQuestion, DeciderUsage, Question, YesNoQuestion
from verifier.models.retry import classify_upstream_error, retrying


def to_jev_question(q: Question):
    from typesafe_sdk import Choice, Noul

    if isinstance(q, ChoiceQuestion):
        return Choice(instructions=q.instructions, criteria=dict(q.options))
    if isinstance(q, YesNoQuestion):
        return Noul(instructions=q.instructions)
    raise TypeError(f"unsupported question type {type(q).__name__}")


def from_jev_answer(question_id: str, raw: dict[str, Any]) -> Answer:
    if raw.get("type") == "choice":
        return Answer(
            question_id=question_id,
            choice=raw.get("choice"),
            probabilities=dict(raw.get("probabilities") or {}),
            confidence=raw.get("confidence"),
            raw=raw,
        )
    return Answer(question_id=question_id, yes_prob=raw.get("noul"), raw=raw)


class JevDecider:
    name = "jev"

    def __init__(self, *, api_key: str | None, model: str, cache: ModelCallCache, attempts: int = 3, timeout: float = 30.0, expected_served: str | None = None):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.attempts = attempts
        self.timeout = timeout
        self.expected_served = expected_served
        self.served_drift: str | None = None
        self.usage = DeciderUsage()
        self.reported_model: str | None = None
        self._client = None

    def _check_served(self, served: str | None) -> None:
        """Drift sentinel: the upstream alias ('jev-latest') can start serving
        a different model without any change on our side; that must surface as
        a warning on the first drifted call, never pass silently."""
        if not served or not self.expected_served or served == self.expected_served:
            return
        if self.served_drift is None:
            import logging

            logging.getLogger("verifier.jev").warning(
                "served model drift: expected %s, got %s", self.expected_served, served
            )
        self.served_drift = served

    def _get_client(self):
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            self._client = AsyncTypeSafeClient(api_key=self.api_key, model=self.model, timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ask(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        request = {
            "state": state,
            "questions": {
                q.question_id: (
                    {"type": "choice", "instructions": q.instructions, "options": q.options}
                    if isinstance(q, ChoiceQuestion)
                    else {"type": "noul", "instructions": q.instructions}
                )
                for q in questions
            },
        }

        async def fetch() -> dict[str, Any]:
            from typesafe_sdk import TypeSafeError

            jev_questions = {q.question_id: to_jev_question(q) for q in questions}

            async def once():
                try:
                    return await self._get_client().system_one(state, jev_questions)
                except TypeSafeError as exc:
                    # Classify immediately so tenacity's predicate sees the
                    # retryable flag on our own error type.
                    raise classify_upstream_error(exc, "jev") from exc

            response = None
            async for attempt in retrying(self.attempts):
                with attempt:
                    response = await once()
            assert response is not None
            dumped = response.model_dump()
            return {
                "model": dumped.get("model"),
                "answers": dumped.get("answers", {}),
                "usage": dumped.get("usage"),
            }

        # Retryable SDK errors are wrapped *inside* fetch via retrying(); a raw
        # non-TypeSafe exception (network layer) is classified here as well.
        try:
            result = await self.cache.call(provider="jev", model=self.model, request=request, fetch=fetch)
        except Exception as exc:
            from verifier.harness.errors import ReplayMiss, UpstreamError

            if isinstance(exc, (UpstreamError, ReplayMiss)):
                raise
            raise classify_upstream_error(exc, "jev") from exc

        self.reported_model = result.get("model") or self.reported_model
        self._check_served(result.get("model"))
        usage = result.get("usage") or {}
        self.usage.add(
            DeciderUsage(
                calls=1,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            )
        )
        answers = result.get("answers", {})
        missing = [q.question_id for q in questions if q.question_id not in answers]
        if missing:
            from verifier.harness.errors import MalformedModelOutput

            raise MalformedModelOutput("jev", f"missing answers for {missing[:5]}")
        return {qid: from_jev_answer(qid, raw) for qid, raw in answers.items() if qid in {q.question_id for q in questions}}
