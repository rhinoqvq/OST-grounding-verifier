"""The optimized method's whole-sentence call: stage 1 of the two-stage judgment, and the
sentence guard's check when two-stage is off.

One request per answer (batched only for a very long answer): per answer sentence (the main
system's sentences) two choice questions, the verdict and which source sentence is the
evidence, against the source. A source too large for one request is narrowed to the
sentences with the most lexical overlap with the answer (selection only, never rewriting).
That is a design choice of this method: stage 1 only settles sentences it calls S with
evidence, and every other sentence still gets the full judgment against the whole source.

Until 2026-10-23 this was the single-call method's own request (methods/minimal_jev); this
module now builds it itself, byte for byte the same in every case, so recorded calls replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from verifier.harness.batching import DEFAULT_REQUEST_TOKEN_BUDGET, ask_batched, question_tokens
from verifier.harness.context import build_source_state, estimate_tokens
from verifier.models.decider import MAX_CHOICE_OPTIONS, ChoiceQuestion, Question
from verifier.output.schemas import Verdict
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


def sentence_call_request(store: TextStore) -> tuple[str, list[Question], list[Sentence]]:
    """(state, questions, kept source sentences): the state and every question (each evidence
    question lists the kept sentences) fit the budget together; the kept set shrinks until
    they do. The option cost counts the options a question can actually carry (no more than
    the source has)."""
    texts = [s.text for s in store.answer_sentences]
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
            return state, questions, kept
        limit //= 2


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


async def whole_sentence_calls(store: TextStore, deps: "Dependencies") -> tuple[list[SentenceCall], dict]:
    """One call per answer sentence, in sentence order."""
    settings = deps.settings
    state, questions, kept = sentence_call_request(store)
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
    return calls, {"questions": len(questions), "requests": requests, "source_sentences_kept": len(kept)}
