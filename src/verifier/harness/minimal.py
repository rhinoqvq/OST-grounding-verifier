"""Method 2: the minimal LLM baseline — one model call behind the same harness.

What it keeps from the harness, deliberately identical to the optimized method:
the service and its error boundary, input limits, structured output with one
repair round, cache record/replay, the trace, run persistence, and the exit
contract. What it drops: everything else. The model splits the answer into
claims, assigns verdicts and writes citations in ONE call with a minimal
prompt; the exit contract is the only thing standing between its citations and
the response. "How many raw citations the contract had to drop" is measured on
this method — that number is the argument for the optimized architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from verifier.harness.context import check_input_limits
from verifier.harness.trace import Trace
from verifier.output.contract import DraftClaim, enforce_contract
from verifier.output.schemas import Citation, Verdict, VerifyRequest, VerifyResponse
from verifier.rag.source import TextStore

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies


class MinimalCitation(BaseModel):
    passage_index: int = Field(ge=0, description="0-based index of the passage quoted")
    quote: str = Field(description="verbatim substring of that passage")


class MinimalClaim(BaseModel):
    claim_text: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
    reasoning: str
    citations: list[MinimalCitation] = Field(default_factory=list)


class MinimalOutput(BaseModel):
    claims: list[MinimalClaim]


_SYSTEM_PROMPT = """You verify a generated answer against context passages.

Split the answer into its individual factual claims. For each claim decide:
- SUPPORTED: the passages state it.
- CONTRADICTED: the passages state the opposite.
- UNSUPPORTED: the passages neither state nor contradict it.

For SUPPORTED and CONTRADICTED claims cite the evidence: the passage number and
a quote copied VERBATIM, character for character, from that passage.
UNSUPPORTED claims get no citations."""


def build_minimal_messages(request: VerifyRequest) -> list[dict[str, str]]:
    passages = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(request.context_passages))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Passages:\n{passages or '(none)'}\n\n"
                f"Answer to verify:\n{request.generated_answer}"
            ),
        },
    ]


async def run_minimal_llm(
    request: VerifyRequest, deps: "Dependencies"
) -> tuple[VerifyResponse, Trace]:
    from verifier.harness.pipeline import _save  # same persistence as the optimized method

    trace = Trace()
    try:
        response = await _run(request, deps, trace)
        _save(deps, request, response, "ok", trace)
        return response, trace
    except Exception as exc:
        trace.note(f"failed: {type(exc).__name__}: {exc}")
        _save(deps, request, None, f"error:{type(exc).__name__}", trace)
        raise


async def _run(request: VerifyRequest, deps: "Dependencies", trace: Trace) -> VerifyResponse:
    r = trace.start_round("0-prepare")
    check_input_limits(request.generated_answer, request.context_passages, deps.settings)
    store = TextStore.from_request(request.generated_answer, request.context_passages)
    trace.end_round(r, answer_sentences=len(store.answer_sentences))
    if not store.answer_sentences:
        trace.note("answer empty or whitespace: no claims")
        return VerifyResponse(claims=[])

    r = trace.start_round("1-minimal-llm")
    output = await deps.llm.generate_structured(build_minimal_messages(request), MinimalOutput)
    trace.end_round(r, raw_claims=len(output.claims))

    r = trace.start_round("2-render")
    drafts = [
        DraftClaim(
            claim_text=c.claim_text,
            verdict=Verdict(c.verdict),
            reasoning=c.reasoning,
            citations=[
                Citation(index=cit.passage_index, quote=cit.quote)
                for cit in c.citations
                if cit.quote  # an empty quote can never verify; drop it before the contract
            ],
        )
        for c in output.claims
    ]
    claims, report = enforce_contract(drafts, request.context_passages)
    trace.end_round(
        r,
        claims=len(claims),
        contract_violations=report.violations,
        dropped_citations=report.dropped_citations,
        downgraded=report.downgraded_claims,
    )
    return VerifyResponse(claims=claims)
