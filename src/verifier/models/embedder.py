"""Sentence embeddings via DashScope's OpenAI-compatible endpoint.

Used only as a *prior* for evidence selection: embedding distance measures
unpolarized relatedness (shared entities, paraphrase), which is exactly what
the per-sentence relevance probe misses on linking premises ("the Acorn
project uses the bronze policy" scores low as evidence alone, but shares the
entity with the claim). Embeddings never decide verdicts — they cannot see
negation or numbers.
"""

from __future__ import annotations

import math

from verifier.harness.errors import MalformedModelOutput, ReplayMiss, UpstreamError
from verifier.models.cache import ModelCallCache
from verifier.models.retry import classify_upstream_error, retrying

_BATCH = 10  # DashScope embedding batch limit is small; stay safe


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class Embedder:
    name = "qwen-embed"

    def __init__(self, *, api_key: str | None, model: str, base_url: str, cache: ModelCallCache, attempts: int = 3, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.cache = cache
        self.attempts = attempts
        self.timeout = timeout
        self.total_texts = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        request = {"model": self.model, "texts": texts}

        async def fetch():
            import openai

            async def once():
                try:
                    vectors: list[list[float]] = []
                    for i in range(0, len(texts), _BATCH):
                        response = await self._get_client().embeddings.create(
                            model=self.model, input=texts[i : i + _BATCH]
                        )
                        vectors.extend(d.embedding for d in response.data)
                    return vectors
                except openai.OpenAIError as exc:
                    raise classify_upstream_error(exc, self.name) from exc

            vectors = None
            async for attempt in retrying(self.attempts):
                with attempt:
                    vectors = await once()
            assert vectors is not None
            return {"vectors": vectors}

        try:
            result = await self.cache.call(provider=self.name, model=self.model, request=request, fetch=fetch)
        except Exception as exc:
            if isinstance(exc, (UpstreamError, ReplayMiss, MalformedModelOutput)):
                raise
            raise classify_upstream_error(exc, self.name) from exc
        self.total_texts += len(texts)
        return result["vectors"]
