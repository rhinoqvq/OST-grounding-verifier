"""HTTP service: POST /verify per the assignment spec.

Error boundary (anything not a clean verdict is an explicit error, never a
fake success):
- 422 invalid request body (FastAPI/Pydantic)
- 413 input exceeds configured limits
- 502 upstream returned garbage we could not repair, or failed fatally
- 503 upstream transient failure survived retries
- 500 internal failure (extraction impossible, replay miss)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from verifier.harness.config import Settings
from verifier.harness.errors import (
    ExtractionFailed,
    InputTooLarge,
    MalformedModelOutput,
    ReplayMiss,
    UpstreamError,
)
from verifier.harness.pipeline import Dependencies, run_request
from verifier.models.cache import ModelCallCache
from verifier.models.jev import JevDecider
from verifier.models.openai_llm import LLMProviderConfig, StructuredLLM
from verifier.output.schemas import VerifyRequest, VerifyResponse
from verifier.rag.store import connect, init_db


def llm_config(settings: Settings) -> LLMProviderConfig:
    if settings.llm_provider == "openai":
        return LLMProviderConfig(
            name="openai", model=settings.openai_model, api_key=settings.openai_api_key,
            temperature=0.0 if settings.openai_temperature_zero else None,
        )
    if settings.llm_provider == "qwen":
        return LLMProviderConfig(
            name="qwen",
            model=settings.qwen_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.qwen_base_url,
            api="chat",
            schema_mode="json_object" if settings.qwen_schema_mode == "json_object" else "json_schema",
            # structured output is unavailable in thinking mode (DashScope docs)
            extra_body={"enable_thinking": False},
        )
    raise ValueError(f"unknown llm_provider {settings.llm_provider!r}")


def build_dependencies(settings: Settings) -> Dependencies:
    conn = connect(settings.db_path)
    init_db(conn)
    cache = ModelCallCache(conn, mode=settings.cache_mode)
    llm = StructuredLLM(
        llm_config(settings),
        cache=cache,
        attempts=settings.upstream_attempts,
        timeout=settings.openai_timeout_s,
    )
    decider = JevDecider(
        api_key=settings.typesafe_api_key,
        model=settings.jev_model,
        cache=cache,
        attempts=settings.upstream_attempts,
        timeout=settings.jev_timeout_s,
        expected_served=settings.jev_expected_served,
    )
    embedder = None
    if settings.evidence_selector == "hybrid":
        from verifier.models.embedder import Embedder

        embedder = Embedder(
            api_key=settings.dashscope_api_key,
            model=settings.embed_model,
            base_url=settings.embed_base_url or settings.qwen_base_url,
            cache=cache,
            attempts=settings.upstream_attempts,
        )
    return Dependencies(settings=settings, llm=llm, decider=decider, conn=conn, embedder=embedder)


def create_app(settings: Settings | None = None, dependencies: Dependencies | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = dependencies or build_dependencies(settings)
        yield
        decider = app.state.deps.decider
        if hasattr(decider, "aclose"):
            await decider.aclose()
        if app.state.deps.conn is not None:
            app.state.deps.conn.close()

    app = FastAPI(title="grounding-verifier", lifespan=lifespan)

    @app.post("/verify", response_model=VerifyResponse)
    async def post_verify(request: VerifyRequest) -> VerifyResponse:
        response, _trace = await run_request(request, app.state.deps)
        return response

    @app.get("/health")
    async def get_health() -> dict[str, str]:
        return {"status": "ok"}

    register_error_handlers(app)
    return app


def register_error_handlers(app: FastAPI) -> None:
    def _json(status: int, error: str, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": error, "detail": detail})

    @app.exception_handler(InputTooLarge)
    async def _too_large(_: Request, exc: InputTooLarge):
        return _json(413, "input_too_large", str(exc))

    @app.exception_handler(UpstreamError)
    async def _upstream(_: Request, exc: UpstreamError):
        status = 503 if exc.retryable else 502
        return _json(status, "upstream_error", str(exc))

    @app.exception_handler(MalformedModelOutput)
    async def _malformed(_: Request, exc: MalformedModelOutput):
        return _json(502, "malformed_model_output", str(exc))

    @app.exception_handler(ExtractionFailed)
    async def _extraction(_: Request, exc: ExtractionFailed):
        return _json(500, "extraction_failed", str(exc))

    @app.exception_handler(ReplayMiss)
    async def _replay(_: Request, exc: ReplayMiss):
        return _json(500, "replay_miss", str(exc))


app = create_app()
