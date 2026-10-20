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


def grounding_question(store: TextStore, unit: Node, part_index: int) -> YesNoQuestion:
    """Diagnostic part of a kept-whole unit (conditional/disjunction/reported):
    asks for *presence in the source*, not truth. Grounding composes across any
    structure, truth does not — this is what catches a fabricated disjunct or
    condition inside a unit whose overall verdict looks SUPPORTED."""
    part_text = store.slice(unit.diagnostic_spans[part_index])
    return YesNoQuestion(
        question_id=f"g::{unit.node_id}::{part_index}",
        instructions=(
            f"The answer contains this unit (kept whole because splitting would change its meaning):\n"
            f"\"{unit.display_text}\"\n"
            f"Question: does the source contain the following piece of that unit's content — stated "
            f"anywhere, including inside a conditional, an alternative/option, or reported speech?\n"
            f"Content piece: \"{part_text}\""
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
        for i in range(len(unit.diagnostic_spans)):
            questions.append(grounding_question(store, unit, i))
    return questions


# --- staged relation check -------------------------------------------------
# A relation unit ("A —so→ B") packs a multi-hop judgment into one question:
# find the stated link, match the antecedent, match the consequent. One bounded
# question cannot carry that chain (measured: instantiation judged C, pure
# narrowing judged U), so the check is decomposed into single-hop questions
# that code composes under the settled rule:
#   SUPPORTED     verbatim link, general-rule instantiation, pure narrowing
#   UNSUPPORTED   sibling-trigger transfer, novel link, agent shift, widening
#   CONTRADICTED  only when the source states an incompatible link

_LINK_KIND = {
    "cause": "causal link (one side causes or explains the other)",
    "contrast": "contrast (the sides are set against each other)",
    "sequence": "temporal order (one side happens before the other)",
    "condition": "conditional link (one side holds under the other)",
}


def relation_link_question(store: TextStore, unit: Node) -> ChoiceQuestion:
    """Stage 1: which source sentence states a link of this kind at all —
    in ANY direction or polarity (an opposite link still counts here; stage 2
    sorts compatibility out)."""
    kind = _LINK_KIND.get(unit.relation_type.value, "link")
    options = {
        s.sentence_id: f'"{s.text}"' for s in store.all_source_sentences()
    }
    options["no_such_link"] = (
        "No source sentence states a link of this kind between contents like these; "
        "sentences that merely mention the parts separately do not count."
    )
    return ChoiceQuestion(
        question_id=f"rel1::{unit.node_id}",
        instructions=(
            f"{describe_unit(store, unit)}\n"
            f"Which source sentence ITSELF states a {kind} between contents of this kind — "
            "in any direction or polarity? Pick the sentence that states the connection "
            "(because / so that / but / then ...), not one that merely mentions the parts."
        ),
        options=options,
    )


SIDE_OPTIONS = {
    "matches": (
        "In the claimed link this side plays the same role and refers to the same thing as the "
        "source link's corresponding side — or to a specific instance/subset of it (a generic "
        "statement covers its instances)."
    ),
    "opposite": (
        "Same role, same aspect, but with the OPPOSITE or negated value (fast vs slow, enabled "
        "vs disabled, inverted direction)."
    ),
    "different": (
        "It refers to a DIFFERENT thing: another trigger, agent or mechanism that the source "
        "keeps separate, a generalization, or a wider scope than the source side covers."
    ),
}


def relation_stage2_questions(
    store: TextStore, unit: Node, link_sentence: Sentence, left_text: str, right_text: str
) -> list[Question]:
    """Stage 2 for a found link sentence: classify each side against the
    stated link's corresponding side. Polarity is folded into the options —
    'opposite' (same aspect, inverted value) is what licenses CONTRADICTED,
    while 'different' (sibling trigger, widening) can only mean UNSUPPORTED."""
    header = (
        f'Source sentence [{link_sentence.sentence_id}] states a link: "{link_sentence.text}"\n'
        f"Claimed link from the answer: {unit.display_text}\n"
    )
    tail = "Compare it with the corresponding side of the source link and classify."
    return [
        ChoiceQuestion(
            question_id=f"rel2::{unit.node_id}",
            instructions=f'{header}Side under check: "{left_text}"\n{tail}',
            options=SIDE_OPTIONS,
        ),
        ChoiceQuestion(
            question_id=f"rel3::{unit.node_id}",
            instructions=f'{header}Side under check: "{right_text}"\n{tail}',
            options=SIDE_OPTIONS,
        ),
    ]


def staged_link_choice(answer: Answer | None, *, min_prob: float) -> tuple[str | None, str]:
    """Resolve stage 1: (sentence_id, 'chosen') | (None, 'no_link'|'flat')."""
    if answer is None or not answer.probabilities:
        return None, "flat"
    top = max(answer.probabilities, key=answer.probabilities.__getitem__)
    p = answer.probabilities[top]
    if p < min_prob:
        return None, "flat"
    if top == "no_such_link":
        return None, "no_link"
    return top, "chosen"


def decide_relation_verdict(
    unit: Node,
    answers: dict[str, Answer],
    *,
    min_prob: float,
) -> UnitJudgment:
    judgment = UnitJudgment(node_id=unit.node_id, node_version=unit.version, verdict=Verdict.UNSUPPORTED)
    link_answer = answers.get(f"rel1::{unit.node_id}")
    if link_answer is not None:
        judgment.probabilities = dict(link_answer.probabilities)
        judgment.confidence = link_answer.confidence
    sid, status = staged_link_choice(link_answer, min_prob=min_prob)
    if sid is None:
        judgment.flags.append("staged_no_link" if status == "no_link" else "staged_flat_link")
        return judgment

    sides: list[tuple[str, str, float]] = []  # (which, top option, its prob)
    for which, qid in (("left", f"rel2::{unit.node_id}"), ("right", f"rel3::{unit.node_id}")):
        answer = answers.get(qid)
        probs = answer.probabilities if answer and answer.probabilities else {}
        top = max(probs, key=probs.__getitem__) if probs else "different"
        p = probs.get(top, 0.0)
        judgment.probabilities[f"{which}_{top}"] = p
        sides.append((which, top if p >= min_prob else "flat", p))

    detail = ",".join(f"{w}={t}({p:.2f})" for w, t, p in sides)
    # 'different' (sibling trigger, agent switch, widening) can only mean the
    # link is unstated; 'opposite' (same aspect, inverted value) is what
    # licenses CONTRADICTED; both sides matching makes the stated link cover
    # the claimed one.
    if any(t in ("different", "flat") for _, t, _ in sides):
        judgment.flags.append(f"staged_side_mismatch({sid},{detail})")
        return judgment
    if any(t == "opposite" for _, t, _ in sides):
        judgment.verdict = Verdict.CONTRADICTED
        judgment.evidence_sentence_ids = [sid]
        judgment.flags.append(f"staged_incompatible({sid},{detail})")
        return judgment
    judgment.verdict = Verdict.SUPPORTED
    judgment.evidence_sentence_ids = [sid]
    judgment.flags.append(f"staged_matched({sid})")
    return judgment


def apply_grounding_gate(
    unit: Node,
    judgment: "UnitJudgment",
    answers: dict[str, Answer],
    *,
    grounding_min_prob: float,
) -> None:
    """An ungrounded diagnostic part caps a SUPPORTED verdict at UNSUPPORTED.
    CONTRADICTED is left alone (the block already flags a conflict), and
    UNSUPPORTED cannot get worse."""
    if judgment.verdict is not Verdict.SUPPORTED or not unit.diagnostic_spans:
        return
    for i in range(len(unit.diagnostic_spans)):
        answer = answers.get(f"g::{unit.node_id}::{i}")
        if answer is None or answer.yes_prob is None:
            continue
        if answer.yes_prob < grounding_min_prob:
            judgment.verdict = Verdict.UNSUPPORTED
            judgment.evidence_sentence_ids = []
            judgment.flags.append(f"ungrounded_part_{i}(p={answer.yes_prob:.2f})")
            return


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
    embed_ranked: list[str] | None = None,
    embed_top_k: int = 2,
    embed_floor_prob: float = 0.25,
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
    # Embedding prior: the top embedding-ranked sentences join the evidence if
    # the probe gave them at least a floor score. This rescues linking premises
    # ("X uses policy Y") that look unevidential alone and score 0.3-0.5 on the
    # probe, while the floor keeps out pure topic noise.
    if embed_ranked:
        probe = {sid: p for p, sid in ranked}
        for sid in embed_ranked[:embed_top_k]:
            if sid in judgment.evidence_sentence_ids:
                continue
            if probe.get(sid, 0.0) >= embed_floor_prob:
                judgment.evidence_sentence_ids.append(sid)
                judgment.flags.append(f"embed_prior({sid},probe={probe.get(sid, 0.0):.2f})")

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
