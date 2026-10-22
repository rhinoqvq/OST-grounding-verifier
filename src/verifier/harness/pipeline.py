"""Shared harness wiring: the dependency container, the GV_METHOD dispatcher
and run persistence. Each method's own pipeline lives under
verifier.methods.{baseline,minimal_llm,minimal_jev,optimized}.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass

from verifier.harness import runctx
from verifier.harness.config import Settings
from verifier.harness.trace import Trace
from verifier.models.decider import Decider
from verifier.models.openai_llm import StructuredLLM
from verifier.output.schemas import VerifyRequest, VerifyResponse
from verifier.rag import store as rag_store


@dataclass
class Dependencies:
    settings: Settings
    llm: StructuredLLM
    decider: Decider
    conn: sqlite3.Connection | None = None
    embedder: object | None = None  # models.embedder.Embedder when evidence_selector == "hybrid"


async def run_request(request: VerifyRequest, deps: Dependencies) -> tuple[VerifyResponse, Trace]:
    """Route the request to the configured method (GV_METHOD)."""
    token = runctx.CURRENT_RUN_ID.set(uuid.uuid4().hex)
    try:
        method = deps.settings.method
        if method == "optimized":
            from verifier.methods.optimized.pipeline import run_verification

            return await run_verification(request, deps)
        if method == "minimal_llm":
            from verifier.methods.minimal_llm import run_minimal_llm

            return await run_minimal_llm(request, deps)
        if method == "baseline":
            from verifier.methods.baseline import run_baseline

            return await run_baseline(request, deps)
        if method == "minimal_jev":
            from verifier.methods.minimal_jev import run_minimal_jev

            return await run_minimal_jev(request, deps)
        if method == "minimal_jev_plus":
            from verifier.methods.minimal_jev import run_minimal_jev

            return await run_minimal_jev(request, deps, plus=True)
        if method in ("minimal_jev_nc", "minimal_jev_me"):
            from verifier.methods.minimal_jev import run_minimal_jev

            return await run_minimal_jev(request, deps, variant=method.rsplit("_", 1)[1])
        raise NotImplementedError(f"method {method!r} is not implemented yet")
    finally:
        runctx.CURRENT_RUN_ID.reset(token)


# -- persistence ---------------------------------------------------------------

def _save(deps: Dependencies, request: VerifyRequest, response: VerifyResponse | None, status: str, trace: Trace) -> None:
    if deps.conn is None:
        return
    tree = getattr(trace, "_tree", None)
    judgments = getattr(trace, "_judgments", {})
    try:
        rag_store.save_run(
            deps.conn,
            request=request.model_dump(),
            response=response.model_dump() if response is not None else None,
            status=status,
            tree=tree,
            judgments=[j.as_dict() for j in judgments.values()],
            trace=trace.as_dict(),
        )
    except sqlite3.Error as exc:  # persistence must never break the response
        trace.note(f"save_run failed: {exc}")
