"""Generative LLM access (OpenAI) with schema-enforced structured output.

Used for structure proposal (decomposition). The model is asked for JSON that
matches a Pydantic schema; if parsing/validation fails, the error is sent back
once for a repair attempt, after which we raise MalformedModelOutput. The
repair loop is deliberately explicit — "what happens when the model output is
broken" is part of the system, not something hidden inside a library.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from verifier.harness.errors import MalformedModelOutput, ReplayMiss, UpstreamError
from verifier.models.cache import ModelCallCache
from verifier.models.retry import classify_upstream_error, retrying

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMUsage:
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, usage: dict[str, Any] | None) -> None:
        self.calls += 1
        if usage:
            self.input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)


class OpenAILLM:
    name = "openai"

    def __init__(self, *, api_key: str | None, model: str, cache: ModelCallCache, attempts: int = 3, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.attempts = attempts
        self.timeout = timeout
        self.usage = LLMUsage()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        return self._client

    async def _complete_json(self, messages: list[dict[str, str]], schema: type[SchemaT]) -> dict[str, Any]:
        """One structured-output call; returns {"text": raw_json_str, "usage": {...}}."""
        request = {"messages": messages, "schema": schema.model_json_schema(), "model": self.model}

        async def fetch() -> dict[str, Any]:
            import openai

            async def once():
                try:
                    return await self._get_client().responses.parse(
                        model=self.model,
                        input=messages,  # type: ignore[arg-type]
                        text_format=schema,
                    )
                except openai.OpenAIError as exc:
                    raise classify_upstream_error(exc, "openai") from exc

            response = None
            async for attempt in retrying(self.attempts):
                with attempt:
                    response = await once()
            assert response is not None
            parsed = response.output_parsed
            usage = getattr(response, "usage", None)
            return {
                "text": parsed.model_dump_json() if parsed is not None else (response.output_text or ""),
                "usage": usage.model_dump() if usage is not None else None,
            }

        try:
            result = await self.cache.call(provider="openai", model=self.model, request=request, fetch=fetch)
        except Exception as exc:
            if isinstance(exc, (UpstreamError, ReplayMiss, MalformedModelOutput)):
                raise
            raise classify_upstream_error(exc, "openai") from exc
        self.usage.add(result.get("usage"))
        return result

    async def generate_structured(self, messages: list[dict[str, str]], schema: type[SchemaT]) -> SchemaT:
        """Structured output with one explicit repair round."""
        result = await self._complete_json(messages, schema)
        try:
            return schema.model_validate_json(result["text"])
        except (ValidationError, json.JSONDecodeError) as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": result["text"][:8000]},
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed validation with this error:\n"
                        f"{first_error}\n"
                        "Return a corrected JSON object that satisfies the schema. "
                        "Fix only the structure; do not change your analysis."
                    ),
                },
            ]
            retry_result = await self._complete_json(repair_messages, schema)
            try:
                return schema.model_validate_json(retry_result["text"])
            except (ValidationError, json.JSONDecodeError) as second_error:
                raise MalformedModelOutput(
                    "openai", f"schema validation failed twice: {second_error}"
                ) from second_error
