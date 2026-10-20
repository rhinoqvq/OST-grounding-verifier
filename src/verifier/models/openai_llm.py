"""Generative LLM access with schema-enforced structured output.

Two providers share this client, both through the OpenAI SDK:
- openai: the Responses API with `text_format` (full Structured Outputs);
- qwen:   DashScope's OpenAI-compatible endpoint, Chat Completions. Qwen
  Max/Plus/Flash support strict `json_schema` in non-thinking mode; older
  models only guarantee `json_object` (valid JSON, no schema) — our explicit
  repair round covers the difference.

If parsing/validation fails, the error is sent back once for a repair attempt,
after which we raise MalformedModelOutput. The repair loop is deliberately
explicit — "what happens when the model output is broken" is part of the
system, not something hidden inside a library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from verifier.harness.errors import MalformedModelOutput, ReplayMiss, UpstreamError
from verifier.models.cache import ModelCallCache
from verifier.models.retry import classify_upstream_error, retrying

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str                                   # "openai" | "qwen" (trace/cache label)
    model: str
    api_key: str | None
    base_url: str | None = None                 # None -> SDK default (api.openai.com)
    api: Literal["responses", "chat"] = "responses"
    schema_mode: Literal["json_schema", "json_object"] = "json_schema"
    extra_body: dict[str, Any] | None = None    # e.g. {"enable_thinking": False} for qwen
    temperature: float | None = 0.0             # decomposition should be as deterministic as possible


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


class StructuredLLM:
    def __init__(self, config: LLMProviderConfig, *, cache: ModelCallCache, attempts: int = 3, timeout: float = 60.0):
        self.config = config
        self.cache = cache
        self.attempts = attempts
        self.timeout = timeout
        self.usage = LLMUsage()
        self._client = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.api_key, base_url=self.config.base_url, timeout=self.timeout
            )
        return self._client

    async def _fetch_once(self, messages: list[dict[str, str]], schema: type[SchemaT]) -> dict[str, Any]:
        cfg = self.config
        client = self._get_client()
        temperature = {} if cfg.temperature is None else {"temperature": cfg.temperature}
        if cfg.api == "responses":
            response = await client.responses.parse(model=cfg.model, input=messages, text_format=schema, **temperature)  # type: ignore[arg-type]
            parsed = response.output_parsed
            text = parsed.model_dump_json() if parsed is not None else (response.output_text or "")
            usage = getattr(response, "usage", None)
            return {"text": text, "usage": usage.model_dump() if usage is not None else None}
        if cfg.schema_mode == "json_schema":
            response = await client.chat.completions.parse(
                model=cfg.model,
                messages=messages,  # type: ignore[arg-type]
                response_format=schema,
                extra_body=cfg.extra_body,
                **temperature,
            )
            message = response.choices[0].message
            text = message.parsed.model_dump_json() if message.parsed is not None else (message.content or "")
        else:
            # json_object mode guarantees valid JSON only; the schema goes into
            # the prompt and our validation/repair round enforces it.
            schema_hint = {
                "role": "system",
                "content": "Respond with a single JSON object that conforms to this JSON schema:\n"
                + json.dumps(schema.model_json_schema(), ensure_ascii=False),
            }
            response = await client.chat.completions.create(
                model=cfg.model,
                messages=[schema_hint, *messages],  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                extra_body=cfg.extra_body,
                **temperature,
            )
            text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return {"text": text, "usage": usage.model_dump() if usage is not None else None}

    async def _complete_json(self, messages: list[dict[str, str]], schema: type[SchemaT]) -> dict[str, Any]:
        """One structured-output call; returns {"text": raw_json_str, "usage": {...}}."""
        cfg = self.config
        request = {
            "messages": messages,
            "schema": schema.model_json_schema(),
            "model": cfg.model,
            "api": cfg.api,
            "schema_mode": cfg.schema_mode,
            "temperature": cfg.temperature,
        }

        async def fetch() -> dict[str, Any]:
            import openai

            async def once():
                try:
                    return await self._fetch_once(messages, schema)
                except openai.OpenAIError as exc:
                    raise classify_upstream_error(exc, cfg.name) from exc

            result = None
            async for attempt in retrying(self.attempts):
                with attempt:
                    result = await once()
            assert result is not None
            return result

        try:
            result = await self.cache.call(provider=cfg.name, model=cfg.model, request=request, fetch=fetch)
        except Exception as exc:
            if isinstance(exc, (UpstreamError, ReplayMiss, MalformedModelOutput)):
                raise
            raise classify_upstream_error(exc, cfg.name) from exc
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
                    self.config.name, f"schema validation failed twice: {second_error}"
                ) from second_error


# Backwards-compatible constructor used before multi-provider support.
def OpenAILLM(*, api_key: str | None, model: str, cache: ModelCallCache, attempts: int = 3, timeout: float = 60.0) -> StructuredLLM:
    return StructuredLLM(
        LLMProviderConfig(name="openai", model=model, api_key=api_key),
        cache=cache,
        attempts=attempts,
        timeout=timeout,
    )
