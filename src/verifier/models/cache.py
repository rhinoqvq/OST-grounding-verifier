"""Cache & replay of raw model responses (stored in SQLite).

Modes:
- off:    always call the model, store nothing.
- record: call the model, store the raw response keyed by request hash.
- replay: never call the model; a missing key is an error. This makes every
  metric recomputable from the committed cache without any API key.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Awaitable, Callable

from verifier.harness.errors import ReplayMiss
from verifier.harness.runctx import CURRENT_RUN_ID


class ModelCallCache:
    def __init__(self, conn: sqlite3.Connection, mode: str = "off"):
        if mode not in ("off", "record", "replay"):
            raise ValueError(f"unknown cache mode {mode!r}")
        self.conn = conn
        self.mode = mode

    @staticmethod
    def key(provider: str, model: str, request: Any) -> str:
        canonical = json.dumps(
            {"provider": provider, "model": model, "request": request},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT response_json FROM model_calls WHERE cache_key=?", (cache_key,)
        ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def put(
        self,
        cache_key: str,
        *,
        provider: str,
        model: str,
        request: Any,
        response: Any,
        usage: Any = None,
        latency_ms: float | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO model_calls "
            "(cache_key, provider, model, request_json, response_json, usage_json, "
            "latency_ms, created, run_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                cache_key,
                provider,
                model,
                json.dumps(request, ensure_ascii=False, default=str),
                json.dumps(response, ensure_ascii=False, default=str),
                json.dumps(usage, ensure_ascii=False, default=str) if usage is not None else None,
                latency_ms,
                time.time(),
                CURRENT_RUN_ID.get(),
            ),
        )
        self.conn.commit()

    async def call(
        self,
        *,
        provider: str,
        model: str,
        request: Any,
        fetch: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Run a model call through the cache. `fetch` returns a JSON-safe dict
        {"response": ..., "usage": ...}; the same shape is what replay returns."""
        cache_key = self.key(provider, model, request)
        if self.mode == "replay":
            hit = self.get(cache_key)
            if hit is None:
                raise ReplayMiss(f"replay mode: no cached response for {provider}/{model}")
            return hit
        if self.mode == "record":
            hit = self.get(cache_key)
            if hit is not None:
                return hit
        started = time.perf_counter()
        result = await fetch()
        latency_ms = (time.perf_counter() - started) * 1000
        result.setdefault("latency_ms", latency_ms)
        if self.mode == "record":
            self.put(
                cache_key,
                provider=provider,
                model=model,
                request=request,
                response=result,
                usage=result.get("usage"),
                latency_ms=latency_ms,
            )
        return result
