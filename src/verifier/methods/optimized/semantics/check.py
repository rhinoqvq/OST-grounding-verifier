"""Round 2: Jev checks the proposed decomposition against the answer alone.

Two kinds of questions:
- fidelity, per unit: did this unit change meaning / add content / lose a
  qualifier compared to its sentence?
- coverage, per sentence: do the units for this sentence, taken together,
  leave out any assertion the sentence makes?

MVP failure policy: a sentence with any failed unit or a coverage gap falls
back to one whole-sentence unit (the original text cannot lose information).
LLM-driven rebuilding is a later iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

from verifier.harness.context import describe_unit
from verifier.models.decider import Answer, ChoiceQuestion, Question, YesNoQuestion
from verifier.rag.source import TextStore
from verifier.rag.tree import Node, NodeKind, NodeStatus, SemanticTree

FIDELITY_OPTIONS = {
    "faithful": "The unit restates part of the sentence exactly: same subject, same numbers and units, same negation, same qualifiers (approximately / by default / most / only ...). Nothing added.",
    "changed_meaning": "The unit alters what the sentence says: different subject or object binding, changed number, flipped or dropped negation.",
    "added_information": "The unit asserts something the sentence does not say (a detail, cause, or completion the writer did not state).",
    "lost_qualifier": "The unit drops a condition, scope or hedge that the sentence attaches (an if-clause, 'only when', 'approximately', 'by default', 'most').",
}


def fidelity_question(store: TextStore, unit: Node) -> ChoiceQuestion:
    return ChoiceQuestion(
        question_id=f"fid::{unit.node_id}",
        instructions=(
            f"The answer sentence [{unit.sentence_id}] was decomposed. "
            f"Judge this proposed unit against that sentence only.\n{describe_unit(store, unit)}"
        ),
        options=FIDELITY_OPTIONS,
    )


def coverage_question(store: TextStore, sentence_id: str, units: list[Node]) -> YesNoQuestion:
    unit_lines = "\n".join(f"- {u.display_text or store.slice(u.anchor_spans[0])}" for u in units)
    return YesNoQuestion(
        question_id=f"cov::{sentence_id}",
        instructions=(
            f"Sentence [{sentence_id}] was decomposed into these units:\n{unit_lines}\n"
            "Do these units, taken together, leave out any assertion that the sentence makes? "
            "(Greetings and meta-talk about passages do not count as assertions.)"
        ),
    )


def check_questions(store: TextStore, tree: SemanticTree) -> list[Question]:
    questions: list[Question] = []
    by_sentence: dict[str, list[Node]] = {}
    for node in tree.nodes.values():
        if node.kind is NodeKind.SENTENCE or node.origin != "answer":
            continue
        questions.append(fidelity_question(store, node))
        by_sentence.setdefault(node.sentence_id, []).append(node)
    for sentence_id, units in by_sentence.items():
        questions.append(coverage_question(store, sentence_id, units))
    return questions


@dataclass
class CheckOutcome:
    accepted_units: list[Node]
    fallback_sentence_ids: set[str]
    notes: list[str]


def apply_check_results(
    tree: SemanticTree,
    answers: dict[str, Answer],
    *,
    fidelity_min_prob: float,
) -> CheckOutcome:
    notes: list[str] = []
    fallback: set[str] = set()
    accepted: list[Node] = []

    units = [
        n for n in tree.nodes.values() if n.kind is not NodeKind.SENTENCE and n.origin == "answer"
    ]
    for unit in units:
        answer = answers.get(f"fid::{unit.node_id}")
        if answer is None:
            notes.append(f"{unit.node_id}: no fidelity answer; sentence falls back")
            fallback.add(unit.sentence_id)
            continue
        p_faithful = answer.probabilities.get("faithful", 0.0)
        if p_faithful >= fidelity_min_prob:
            unit.status = NodeStatus.CHECKED
            accepted.append(unit)
        else:
            unit.status = NodeStatus.INVALID
            top = answer.choice or "?"
            notes.append(
                f"{unit.node_id}: fidelity rejected (P(faithful)={p_faithful:.2f}, verdict={top}); "
                f"sentence {unit.sentence_id} falls back"
            )
            fallback.add(unit.sentence_id)

    for question_id, answer in answers.items():
        if question_id.startswith("cov::") and (answer.yes_prob or 0.0) >= 0.5:
            sentence_id = question_id.removeprefix("cov::")
            notes.append(f"{sentence_id}: coverage gap reported (p={answer.yes_prob:.2f}); falls back")
            fallback.add(sentence_id)

    accepted = [u for u in accepted if u.sentence_id not in fallback]
    # a relation whose member fell back cannot survive on its own
    surviving_ids = {u.node_id for u in accepted}
    kept: list[Node] = []
    for unit in accepted:
        if unit.kind.value == "relation" and any(m not in surviving_ids for m in unit.member_ids):
            notes.append(f"{unit.node_id}: member fell back; relation dropped, sentence falls back")
            fallback.add(unit.sentence_id)
        else:
            kept.append(unit)
    kept = [u for u in kept if u.sentence_id not in fallback]
    return CheckOutcome(accepted_units=kept, fallback_sentence_ids=fallback, notes=notes)
