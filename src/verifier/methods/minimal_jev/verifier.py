"""Method 2, Jev variant: the minimal baseline with the judge model in place of the LLM.

Same shape as minimal_llm — one request, a minimal prompt, the exit contract as the
only check — so the two isolate what the model contributes, and the optimized method
shows what the pipeline adds on the same judge. The judge only chooses, so it can
neither split the answer nor write a quote:

- each answer sentence is one claim;
- per sentence the one request asks two choice questions: the verdict (the same three
  definitions as minimal_llm's prompt) and which source sentence is the evidence;
- the quote is that sentence, sliced from the source by code.

Nothing from the optimized method is used: no decomposition, no source tree, no staged
checks, no thresholds (the top choice is the answer). A source too large for one
request is narrowed to the sentences with the most lexical overlap with the answer —
selection only, never rewriting; the trace records how many were kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from verifier.harness.batching import DEFAULT_REQUEST_TOKEN_BUDGET, ask_batched, question_tokens
from verifier.harness.context import build_source_state, check_input_limits, estimate_tokens
from verifier.harness.trace import Trace
from verifier.models.decider import MAX_CHOICE_OPTIONS, ChoiceQuestion, Question
from verifier.output.citations import make_citation
from verifier.output.contract import DraftClaim, enforce_contract
from verifier.output.schemas import Verdict, VerifyRequest, VerifyResponse
from verifier.rag.retrieve import lexical_scores
from verifier.rag.source import Sentence, TextStore

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies

VERDICTS = {
    "SUPPORTED": "the passages state it",
    "CONTRADICTED": "the passages state the opposite",
    "UNSUPPORTED": "the passages neither state nor contradict it",
}
NO_EVIDENCE = "none"


def verdict_question(k: int, text: str) -> ChoiceQuestion:
    return ChoiceQuestion(
        question_id=f"v{k}",
        instructions=f'Verify this sentence from an answer against the source passages.\nSentence: "{text}"',
        options=dict(VERDICTS),
    )


def evidence_question(k: int, text: str, sentences: list[Sentence]) -> ChoiceQuestion:
    options = {s.sentence_id: f"source sentence [{s.sentence_id}]" for s in sentences}
    options[NO_EVIDENCE] = "no source sentence states it or its opposite"
    return ChoiceQuestion(
        question_id=f"e{k}",
        instructions=(
            "Which source sentence is the evidence that states this sentence from an answer, "
            f'or states its opposite?\nSentence: "{text}"'
        ),
        options=options,
    )


def fit_source(store: TextStore, answer_texts: list[str], *, max_tokens: int, limit: int) -> list[Sentence]:
    """All source sentences when they fit; otherwise the ones with the most lexical overlap
    with any answer sentence, at most `limit` of them and `max_tokens` of state, kept in
    source order."""
    sentences = store.all_source_sentences()
    if len(sentences) <= limit and estimate_tokens(build_source_state(store, sentences)) <= max_tokens:
        return sentences
    best = [0.0] * len(sentences)
    for text in answer_texts:
        for i, score in enumerate(lexical_scores(text, sentences)):
            best[i] = max(best[i], score)
    ranked = sorted(range(len(sentences)), key=lambda i: -best[i])
    kept: list[int] = []
    budget = max_tokens - estimate_tokens(build_source_state(store, []))
    for i in ranked:
        if len(kept) == limit:
            break
        cost = estimate_tokens(sentences[i].text) + 8  # id and line overhead
        if cost > budget:
            continue
        kept.append(i)
        budget -= cost
    return [sentences[i] for i in sorted(kept)]


@dataclass(frozen=True)
class SentenceCall:
    """The judge's call on one whole answer sentence."""

    verdict: Verdict           # the top choice
    evidence: str | None       # source sentence id; None for U or when no sentence was picked
    prob: float

    @property
    def supported(self) -> bool:
        """S that survives the exit contract: an S needs a citation."""
        return self.verdict is Verdict.SUPPORTED and self.evidence is not None


async def judge_sentences(store: TextStore, deps: "Dependencies") -> tuple[list[SentenceCall], dict]:
    """One call per answer sentence, in sentence order: verdict and evidence questions in
    one request (batched only for a very long answer). The optimized method's sentence
    guard sends exactly this request, so the two methods share one cached call."""
    settings = deps.settings
    texts = [s.text for s in store.answer_sentences]
    # one request: the state and every question (each evidence question lists the kept
    # sentences as options) must fit the budget together; shrink the kept set until they do.
    # The option cost counts the options a question can actually carry (no more than the
    # source has): costing every question at the 255-option cap left a 13+ sentence answer
    # no room for any source, and the empty request then passed the size check (2026-10-23:
    # 214 RAGTruth answers were judged against nothing).
    sources = store.all_source_sentences()
    limit = MAX_CHOICE_OPTIONS - 1  # one option is NO_EVIDENCE
    while True:
        base = sum(question_tokens(q) for k, t in enumerate(texts)
                   for q in (verdict_question(k, t), evidence_question(k, t, [])))
        option_cost = len(texts) * (min(limit, len(sources)) + 1) * 8
        kept = fit_source(store, texts, max_tokens=DEFAULT_REQUEST_TOKEN_BUDGET - base - option_cost - 500,
                          limit=limit)
        state = build_source_state(store, kept)
        questions: list[Question] = []
        for k, t in enumerate(texts):
            questions += [verdict_question(k, t), evidence_question(k, t, kept)]
        total = estimate_tokens(state) + sum(question_tokens(q) for q in questions)
        if (total <= DEFAULT_REQUEST_TOKEN_BUDGET and kept) or limit <= 10:
            break
        limit //= 2
    if len(questions) <= settings.max_questions_per_request:
        answers = await deps.decider.ask(state, questions)
        requests = 1
    else:  # a long answer: more questions than one request takes
        answers = await ask_batched(deps.decider, state, questions,
                                    max_per_request=settings.max_questions_per_request,
                                    max_parallel=settings.max_parallel_requests)
        requests = -(-len(questions) // settings.max_questions_per_request)
    calls = []
    for k in range(len(texts)):
        v, e = answers[f"v{k}"], answers[f"e{k}"]
        verdict = Verdict(v.choice)
        picked = verdict is not Verdict.UNSUPPORTED and e.choice and e.choice != NO_EVIDENCE
        calls.append(SentenceCall(verdict, e.choice if picked else None, v.probabilities.get(v.choice, 0.0)))
    return calls, {"questions": len(questions), "requests": requests, "source_sentences_kept": len(kept)}


async def run_minimal_jev(
    request: VerifyRequest, deps: "Dependencies"
) -> tuple[VerifyResponse, Trace]:
    from verifier.harness.pipeline import _save  # same persistence as the other methods

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
    settings = deps.settings
    r = trace.start_round("0-prepare")
    check_input_limits(request.generated_answer, request.context_passages, settings)
    store = TextStore.from_request(request.generated_answer, request.context_passages)
    claims_src = store.answer_sentences
    trace.end_round(r, answer_sentences=len(claims_src), source_sentences=len(store.all_source_sentences()))
    if not claims_src:
        trace.note("answer empty or whitespace: no claims")
        return VerifyResponse(claims=[])
    texts = [s.text for s in claims_src]

    if not store.all_source_sentences():
        trace.note("context empty: every claim UNSUPPORTED by definition")
        drafts = [DraftClaim(t, Verdict.UNSUPPORTED, "The context is empty.", []) for t in texts]
        claims, _ = enforce_contract(drafts, request.context_passages)
        return VerifyResponse(claims=claims)

    r = trace.start_round("1-minimal-jev")
    calls, info = await judge_sentences(store, deps)
    trace.end_round(r, **info)

    r = trace.start_round("2-render")
    drafts = []
    for t, call in zip(texts, calls):
        citations = [make_citation(store, call.evidence)] if call.evidence else []
        evidence = call.evidence or "none"
        drafts.append(DraftClaim(t, call.verdict, f"Judge: {call.verdict.value} (p={call.prob:.2f}); "
                                 f"evidence: {evidence}.", citations))
    claims, report = enforce_contract(drafts, request.context_passages)
    trace.end_round(r, claims=len(claims), contract_violations=report.violations,
                    downgraded=report.downgraded_claims)
    return VerifyResponse(claims=claims)
