"""Method 2, Jev variant (minimal_jev): the whole answer and the whole source in one request.

"A straightforward single-call LLM setup with minimal prompt tuning", with the judge model in
place of the LLM. The judge only chooses, so the one request asks two choice questions about
the answer as a whole:

- the verdict, with the assignment's three definitions word for word;
- which passage is the evidence (a passage number, or none).

The state is every passage verbatim, numbered, followed by the answer verbatim. Nothing is
split, narrowed or batched: the output is one claim, the whole answer, with the top choice as
its verdict (no threshold) and, for S / C, the chosen passage as its citation, sliced whole by
code. A request that does not fit one call (more passages than one choice question can list,
or over the request budget) is an error with no verdict (OutOfSingleCallRange). The exit
contract is the only check, as for every method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verifier.harness.batching import DEFAULT_REQUEST_TOKEN_BUDGET, question_tokens
from verifier.harness.context import check_input_limits, estimate_tokens
from verifier.harness.errors import OutOfSingleCallRange
from verifier.harness.trace import Trace
from verifier.models.decider import MAX_CHOICE_OPTIONS, ChoiceQuestion
from verifier.output.contract import DraftClaim, enforce_contract
from verifier.output.schemas import Citation, Verdict, VerifyRequest, VerifyResponse

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies

# the assignment's Field Definitions, verbatim
VERDICTS = {
    "SUPPORTED": "Fully entailed by one or more context passages.",
    "CONTRADICTED": "Directly refuted by one or more context passages.",
    "UNSUPPORTED": "Not mentioned or cannot be verified from the provided context alone "
                   "(even if true in the real world).",
}
NO_EVIDENCE = "none"
MAX_PASSAGES = MAX_CHOICE_OPTIONS - 1  # one option is NO_EVIDENCE


def build_state(passages: list[str], answer: str) -> str:
    lines = ["Context passages:"]
    for i, passage in enumerate(passages):
        lines += [f"--- passage {i + 1} ---", passage]
    lines += ["--- answer ---", answer]
    return "\n".join(lines)


def build_questions(passages: list[str]) -> list[ChoiceQuestion]:
    evidence = {str(i + 1): f"passage {i + 1}" for i, p in enumerate(passages) if p.strip()}
    evidence[NO_EVIDENCE] = "no passage"
    return [
        ChoiceQuestion(question_id="verdict", instructions="Verdict for the answer against the context passages.",
                       options=dict(VERDICTS)),
        ChoiceQuestion(question_id="evidence", instructions="Which passage is the evidence for this verdict?",
                       options=evidence),
    ]


async def run_single_call(request: VerifyRequest, deps: "Dependencies") -> tuple[VerifyResponse, Trace]:
    from verifier.harness.pipeline import _save  # same persistence as the other methods

    trace = Trace()
    try:
        response = await _run(request, deps, trace)
        _save(deps, request, response, "ok", trace)
        return response, trace
    except Exception as exc:
        trace.note(f"failed: {type(exc).__name__}: {exc}")
        _save(deps, request, None, getattr(exc, "status", f"error:{type(exc).__name__}"), trace)
        raise


async def _run(request: VerifyRequest, deps: "Dependencies", trace: Trace) -> VerifyResponse:
    answer, passages = request.generated_answer, request.context_passages
    r = trace.start_round("0-prepare")
    check_input_limits(answer, passages, deps.settings)
    trace.end_round(r, passages=len(passages), answer_chars=len(answer))
    if not answer.strip():
        trace.note("answer empty or whitespace: no claims")
        return VerifyResponse(claims=[])
    if not any(p.strip() for p in passages):
        trace.note("context empty: the answer is UNSUPPORTED by definition")
        claims, _ = enforce_contract([DraftClaim(answer.strip(), Verdict.UNSUPPORTED, "The context is empty.", [])],
                                     passages)
        return VerifyResponse(claims=claims)

    if len(passages) > MAX_PASSAGES:
        raise OutOfSingleCallRange("single-call passages", len(passages), MAX_PASSAGES)
    state, questions = build_state(passages, answer), build_questions(passages)
    tokens = estimate_tokens(state) + sum(question_tokens(q) for q in questions)
    if tokens > DEFAULT_REQUEST_TOKEN_BUDGET:
        raise OutOfSingleCallRange("single-call request (tokens)", tokens, DEFAULT_REQUEST_TOKEN_BUDGET)

    r = trace.start_round("1-single-call")
    answers = await deps.decider.ask(state, questions)
    v, e = answers["verdict"], answers["evidence"]
    verdict = Verdict(v.choice)
    prob = (v.probabilities or {}).get(v.choice, 0.0)
    picked = e.choice if e.choice and e.choice != NO_EVIDENCE else None
    trace.end_round(r, request_tokens=tokens, verdict=v.probabilities, evidence=e.choice)

    r = trace.start_round("2-render")
    citations = []
    if verdict is not Verdict.UNSUPPORTED and picked is not None:
        index = int(picked) - 1
        citations = [Citation(index=index, quote=passages[index], start_char=0, end_char=len(passages[index]))]
    reasoning = f"Judge: {verdict.value} (p={prob:.2f}); evidence: {f'passage {picked}' if picked else 'none'}."
    claims, report = enforce_contract([DraftClaim(answer.strip(), verdict, reasoning, citations)], passages)
    trace.end_round(r, claims=len(claims), contract_violations=report.violations,
                    downgraded=report.downgraded_claims)
    return VerifyResponse(claims=claims)
