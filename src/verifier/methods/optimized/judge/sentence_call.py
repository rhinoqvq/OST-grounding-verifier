"""The optimized method's whole-sentence call: stage 1 of the two-stage judgment, and the
sentence guard's check when two-stage is off.

One request per answer (batched only for a very long answer): the whole source as the
state, and per answer sentence (the main system's sentences) two choice questions, the
verdict and which source sentence is the evidence. The source is never narrowed: when it
has more sentences than an evidence question can list, or the request would not fit the
budget, the answer gets no whole-sentence call and every sentence takes the full path.

Until 2026-10-23 this borrowed the single-call method's request, which narrows a large
source to the sentences with the most lexical overlap. When the source fits, the request
here is the same one byte for byte (same state, questions and batching), so recorded calls
replay; the two methods no longer share code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from verifier.harness.batching import DEFAULT_REQUEST_TOKEN_BUDGET, ask_batched, question_tokens
from verifier.harness.context import build_source_state, estimate_tokens
from verifier.models.decider import MAX_CHOICE_OPTIONS, ChoiceQuestion, Question
from verifier.output.schemas import Verdict
from verifier.rag.source import Sentence, TextStore

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies

VERDICTS = {
    "SUPPORTED": "the passages state it",
    "CONTRADICTED": "the passages state the opposite",
    "UNSUPPORTED": "the passages neither state nor contradict it",
}
NO_EVIDENCE = "none"
MAX_SOURCE_SENTENCES = MAX_CHOICE_OPTIONS - 1  # one option is NO_EVIDENCE


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


def sentence_call_request(store: TextStore) -> tuple[str, list[Question]] | str:
    """(state, questions) for the whole source, or why it does not fit one request."""
    texts = [s.text for s in store.answer_sentences]
    sources = store.all_source_sentences()
    if len(sources) > MAX_SOURCE_SENTENCES:
        return f"source has {len(sources)} sentences > {MAX_SOURCE_SENTENCES} evidence options"
    state = build_source_state(store, sources)
    # the budget test the narrowing version applied before it would narrow: the state must
    # fit beside every question with its options, and the whole request must fit
    base = sum(question_tokens(q) for k, t in enumerate(texts)
               for q in (verdict_question(k, t), evidence_question(k, t, [])))
    option_cost = len(texts) * (len(sources) + 1) * 8
    state_tokens = estimate_tokens(state)
    if state_tokens > DEFAULT_REQUEST_TOKEN_BUDGET - base - option_cost - 500:
        return f"source state {state_tokens} tokens does not fit beside {len(texts) * 2} questions"
    questions: list[Question] = []
    for k, t in enumerate(texts):
        questions += [verdict_question(k, t), evidence_question(k, t, sources)]
    total = state_tokens + sum(question_tokens(q) for q in questions)
    if total > DEFAULT_REQUEST_TOKEN_BUDGET:
        return f"request {total} tokens > budget {DEFAULT_REQUEST_TOKEN_BUDGET}"
    return state, questions


async def whole_sentence_calls(store: TextStore, deps: "Dependencies") -> tuple[list[SentenceCall] | None, dict]:
    """One call per answer sentence, in sentence order; None (with the reason in the info)
    when the whole source does not fit one request."""
    settings = deps.settings
    built = sentence_call_request(store)
    if isinstance(built, str):
        return None, {"skipped": built}
    state, questions = built
    if len(questions) <= settings.max_questions_per_request:
        answers = await deps.decider.ask(state, questions)
        requests = 1
    else:  # a long answer: more questions than one request takes, same state in each
        answers = await ask_batched(deps.decider, state, questions,
                                    max_per_request=settings.max_questions_per_request,
                                    max_parallel=settings.max_parallel_requests)
        requests = -(-len(questions) // settings.max_questions_per_request)
    calls = []
    for k in range(len(store.answer_sentences)):
        v, e = answers[f"v{k}"], answers[f"e{k}"]
        verdict = Verdict(v.choice)
        picked = verdict is not Verdict.UNSUPPORTED and e.choice and e.choice != NO_EVIDENCE
        calls.append(SentenceCall(verdict, e.choice if picked else None, v.probabilities.get(v.choice, 0.0)))
    info = {"questions": len(questions), "requests": requests,
            "source_sentences_kept": len(store.all_source_sentences())}
    return calls, info
