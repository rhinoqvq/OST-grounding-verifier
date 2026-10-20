"""Round 3: verdicts and evidence, in one batched Jev round.

Per unit: one 3-way choice (supported / contradicted / not_in_context), per the
agreed rule R1: CONTRADICTED only when the source explicitly states something
that cannot be true together with the claim; weaker-than-claimed or merely
implied counts as not_in_context.

Per (unit, candidate sentence): one yes/no "is this sentence relevant evidence"
— relevance, not support, so evidence that only works combined across passages
is still collected. Both question sets are independent given the same state, so
they ship in the same round.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verifier.harness.context import describe_unit
from verifier.models.decider import Answer, ChoiceQuestion, Question, YesNoQuestion
from verifier.output.schemas import Verdict
from verifier.rag.source import Sentence, TextStore
from verifier.rag.tree import Node

VERDICT_OPTIONS = {
    "supported": (
        "Every part of the claim (subject, numbers, conditions, negation) follows from the "
        "source sentences alone. Combining several source sentences is fine. Paraphrase is fine."
    ),
    "contradicted": (
        "The source explicitly states something that cannot be true together with the claim "
        "(opposite assertion, different number for the same thing, inverted condition)."
    ),
    "not_in_context": (
        "The source neither establishes nor contradicts the claim: it is not mentioned, is only "
        "implied, or the source says something weaker (e.g. 'most' vs the claim's 'all'). "
        "True-in-the-real-world but absent from the source belongs here."
    ),
}


def verdict_question(store: TextStore, unit: Node) -> ChoiceQuestion:
    return ChoiceQuestion(
        question_id=f"v::{unit.node_id}",
        instructions=(
            "Judge this claim strictly against the source passages in the state — "
            f"never against outside knowledge.\n{describe_unit(store, unit)}"
        ),
        options=VERDICT_OPTIONS,
    )


def relevance_question(store: TextStore, unit: Node, sentence: Sentence) -> YesNoQuestion:
    return YesNoQuestion(
        question_id=f"r::{unit.node_id}::{sentence.sentence_id}",
        instructions=(
            f"Source sentence [{sentence.sentence_id}] reads: \"{sentence.text}\"\n"
            "Would a careful fact-checker quote this sentence as evidence when explaining why the "
            "claim below is supported or contradicted? Merely being about the same topic is not "
            f"evidence; the sentence must bear on whether the claim holds.\n{describe_unit(store, unit)}"
        ),
    )


def judge_questions(
    store: TextStore,
    units: list[Node],
    candidates: dict[str, list[Sentence]],
) -> list[Question]:
    questions: list[Question] = []
    for unit in units:
        questions.append(verdict_question(store, unit))
        for sentence in candidates.get(unit.node_id, []):
            questions.append(relevance_question(store, unit, sentence))
    return questions


@dataclass
class UnitJudgment:
    node_id: str
    node_version: int
    verdict: Verdict
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    evidence_sentence_ids: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_version": self.node_version,
            "verdict": self.verdict.value,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "evidence_sentence_ids": self.evidence_sentence_ids,
            "flags": self.flags,
        }


_LABEL_TO_VERDICT = {
    "supported": Verdict.SUPPORTED,
    "contradicted": Verdict.CONTRADICTED,
    "not_in_context": Verdict.UNSUPPORTED,
}


def decide_verdict(
    unit: Node,
    answers: dict[str, Answer],
    candidates: list[Sentence],
    *,
    verdict_min_prob: float,
    relevance_min_prob: float,
    evidence_floor_prob: float,
    max_evidence: int,
) -> UnitJudgment:
    judgment = UnitJudgment(node_id=unit.node_id, node_version=unit.version, verdict=Verdict.UNSUPPORTED)
    answer = answers.get(f"v::{unit.node_id}")
    if answer is None:
        judgment.flags.append("no_verdict_answer")
        return judgment
    judgment.probabilities = dict(answer.probabilities)
    judgment.confidence = answer.confidence

    ranked = sorted(
        (
            ((answers.get(f"r::{unit.node_id}::{s.sentence_id}") or Answer("")).yes_prob or 0.0, s.sentence_id)
            for s in candidates
        ),
        key=lambda t: -t[0],
    )
    judgment.evidence_sentence_ids = [
        sid for p, sid in ranked[:max_evidence] if p >= relevance_min_prob
    ]

    top_label = max(answer.probabilities, key=answer.probabilities.get, default=None)
    top_prob = answer.probabilities.get(top_label, 0.0) if top_label else 0.0
    if top_label is None or top_prob < verdict_min_prob:
        # the calibrated distribution is too flat: say "cannot verify" rather than guess
        judgment.verdict = Verdict.UNSUPPORTED
        judgment.flags.append(f"low_confidence(top={top_label},p={top_prob:.2f})")
        return judgment

    verdict = _LABEL_TO_VERDICT[top_label]
    if verdict is not Verdict.UNSUPPORTED and not judgment.evidence_sentence_ids:
        # Definite verdict but no sentence cleared the relevance bar. The
        # relevance probe is a separate, weaker signal than the verdict itself,
        # so before throwing a confident verdict away, accept the best-ranked
        # sentence if it is at least plausible (above the floor); flag it so
        # evaluation can count how often this path fires and how it performs.
        if ranked and ranked[0][0] >= evidence_floor_prob:
            judgment.evidence_sentence_ids = [ranked[0][1]]
            judgment.flags.append(f"evidence_rank_fallback(p={ranked[0][0]:.2f})")
        else:
            # nothing even plausibly citable: no basis for a definite verdict
            judgment.verdict = Verdict.UNSUPPORTED
            judgment.flags.append(f"no_evidence_for_{verdict.value.lower()}")
            return judgment
    judgment.verdict = verdict
    return judgment
