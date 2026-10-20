"""Jev-only decomposition: code proposes cut points, Jev classifies them.

The generative-LLM proposer rewrites the answer into units (flexible, slow,
can misbind pronouns). This variant never generates text at all:

1. code finds candidate cut points in each answer sentence — connectives and
   their leading punctuation (and/but/because/so/which/or/if/...);
2. Jev answers one choice question per candidate: do the two sides carry
   independently checkable assertions, are they linked (cause/contrast/
   sequence), must the sentence stay whole (condition, disjunction, reported
   speech), or is this no boundary at all;
3. code applies the accepted cuts: fragments are verbatim answer spans, a
   linked cut also yields a relation node over its two neighbours.

Fragments carry the whole sentence as inherited context, so the judge always
sees the original wording. There is no fidelity round: a verbatim fragment
cannot add information, and fragments plus connectives tile the sentence, so
coverage holds by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from verifier.harness.batching import ask_batched
from verifier.harness.context import build_answer_state
from verifier.models.decider import Answer, ChoiceQuestion, Decider
from verifier.rag.source import Sentence, Span, TextStore
from verifier.rag.tree import Node, NodeKind, NodeStatus, RelationType, SemanticTree

# connective -> (what a "linked" cut means here)
_CONNECTIVES = [
    "because", "so that", "so", "therefore", "thus",
    "but", "however", "whereas", "while", "although",
    "which", "who", "whose",
    "and", "or", "unless", "if", "when", "after which", "then",
]
_CUT_RE = re.compile(
    r"[,;:]?\s+\b(" + "|".join(re.escape(c) for c in sorted(_CONNECTIVES, key=len, reverse=True)) + r")\b\s",
    re.IGNORECASE,
)

_MIN_FRAGMENT_WORDS = 3

CUT_OPTIONS = {
    "two_facts": (
        "LEFT and RIGHT each make a complete, independently checkable assertion; splitting here "
        "changes no meaning (typical for 'and' joining two statements about the same subject)."
    ),
    "cause_link": "Both sides are asserted AND the sentence asserts that one causes/explains the other (because/so/therefore).",
    "contrast_link": "Both sides are asserted AND set against each other (but/however/whereas/although).",
    "sequence_link": "Both sides are asserted AND ordered in time (then/after which).",
    "keep_whole": (
        "Splitting would change meaning: a conditional (if/unless/when), a disjunction (or), "
        "reported or hedged content, or a negation whose scope covers both sides."
    ),
    "not_boundary": (
        "The marker does not separate two assertions here: it joins noun phrases inside one fact, "
        "introduces a defining clause, or one side alone would be an incomplete fragment."
    ),
}

_LINK_RELATION = {
    "cause_link": RelationType.CAUSE,
    "contrast_link": RelationType.CONTRAST,
    "sequence_link": RelationType.SEQUENCE,
}


@dataclass(frozen=True)
class CutCandidate:
    question_id: str
    sentence_id: str
    marker_start: int  # char offsets into the answer document
    marker_end: int
    connective: str


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][\w./-]*", text))


def find_candidates(store: TextStore) -> list[CutCandidate]:
    candidates: list[CutCandidate] = []
    for sentence in store.answer_sentences:
        for match in _CUT_RE.finditer(sentence.text):
            left = sentence.text[: match.start()]
            right = sentence.text[match.end():]
            if _word_count(left) < _MIN_FRAGMENT_WORDS or _word_count(right) < _MIN_FRAGMENT_WORDS:
                continue
            candidates.append(
                CutCandidate(
                    question_id=f"cut::{sentence.sentence_id}::{sentence.span.start + match.start()}",
                    sentence_id=sentence.sentence_id,
                    marker_start=sentence.span.start + match.start(),
                    marker_end=sentence.span.start + match.end(),
                    connective=match.group(1).lower(),
                )
            )
    return candidates


def cut_question(store: TextStore, candidate: CutCandidate) -> ChoiceQuestion:
    sentence = store.sentence_by_id(candidate.sentence_id)
    left = store.answer[sentence.span.start : candidate.marker_start].strip()
    right = store.answer[candidate.marker_end : sentence.span.end].strip()
    return ChoiceQuestion(
        question_id=candidate.question_id,
        instructions=(
            f"Answer sentence [{candidate.sentence_id}]: \"{sentence.text}\"\n"
            f"Candidate split at the marker '{candidate.connective}':\n"
            f"LEFT: \"{left}\"\nRIGHT: \"{right}\"\n"
            "Classify what this marker does in this sentence."
        ),
        options=CUT_OPTIONS,
    )


@dataclass
class JevProposalOutcome:
    tree: SemanticTree
    units: list[Node]
    notes: list[str]


def _trim_span(store: TextStore, start: int, end: int) -> Span | None:
    text = store.answer[start:end]
    lead = len(text) - len(text.lstrip(" ,;:."))
    trail = len(text) - len(text.rstrip(" ,;:"))
    if start + lead >= end - trail:
        return None
    return Span(store.ANSWER_DOC, start + lead, end - trail)


def build_tree_from_cuts(
    store: TextStore,
    candidates: list[CutCandidate],
    answers: dict[str, Answer],
    *,
    min_prob: float,
) -> JevProposalOutcome:
    tree = SemanticTree(store=store)
    notes: list[str] = []
    units: list[Node] = []

    for sentence in store.answer_sentences:
        sentence_node = tree.add(
            Node(
                node_id=f"sent::{sentence.sentence_id}",
                origin="answer",
                sentence_id=sentence.sentence_id,
                kind=NodeKind.SENTENCE,
                anchor_spans=[sentence.span],
                display_text=sentence.text,
                status=NodeStatus.CHECKED,
            )
        )

        accepted: list[tuple[CutCandidate, str]] = []
        for candidate in (c for c in candidates if c.sentence_id == sentence.sentence_id):
            answer = answers.get(candidate.question_id)
            if answer is None or answer.choice is None:
                continue
            p = answer.probabilities.get(answer.choice, 0.0)
            if answer.choice in ("two_facts", *_LINK_RELATION) and p >= min_prob:
                accepted.append((candidate, answer.choice))
                notes.append(f"{candidate.question_id}: cut as {answer.choice} (p={p:.2f})")
            elif answer.choice == "keep_whole" and p >= min_prob:
                notes.append(f"{candidate.question_id}: keep_whole (p={p:.2f})")

        if not accepted:
            continue  # select_units falls back to the whole sentence

        accepted.sort(key=lambda t: t[0].marker_start)
        bounds = [sentence.span.start, *[c.marker_start for c, _ in accepted], sentence.span.end]
        starts = [sentence.span.start, *[c.marker_end for c, _ in accepted]]
        fragment_nodes: list[Node | None] = []
        for i in range(len(accepted) + 1):
            span = _trim_span(store, starts[i], bounds[i + 1])
            if span is None or _word_count(store.slice(span)) < _MIN_FRAGMENT_WORDS:
                fragment_nodes.append(None)
                notes.append(f"{sentence.sentence_id}: fragment {i} too small; kept inside neighbours")
                continue
            node = tree.add(
                Node(
                    node_id=f"frag::{sentence.sentence_id}::{i}",
                    origin="answer",
                    sentence_id=sentence.sentence_id,
                    kind=NodeKind.FACT,
                    anchor_spans=[span],
                    inherited_spans=[sentence.span],  # judge always sees the original sentence
                    parent_id=sentence_node.node_id,
                    display_text=store.slice(span),
                    status=NodeStatus.CHECKED,
                )
            )
            fragment_nodes.append(node)

        made = [n for n in fragment_nodes if n is not None]
        if len(made) < 2:
            # a lone fragment is worse than the sentence itself: undo
            for node in made:
                del tree.nodes[node.node_id]
            notes.append(f"{sentence.sentence_id}: cuts produced <2 usable fragments; whole sentence")
            continue
        units.extend(made)

        for i, (candidate, choice) in enumerate(accepted):
            if choice not in _LINK_RELATION:
                continue
            left, right = fragment_nodes[i], fragment_nodes[i + 1]
            if left is None or right is None:
                continue
            relation = tree.add(
                Node(
                    node_id=f"rel::{sentence.sentence_id}::{i}",
                    origin="answer",
                    sentence_id=sentence.sentence_id,
                    kind=NodeKind.RELATION,
                    anchor_spans=[Span(store.ANSWER_DOC, candidate.marker_start, candidate.marker_end)],
                    inherited_spans=[sentence.span],
                    relation_type=_LINK_RELATION[choice],
                    member_ids=[left.node_id, right.node_id],
                    parent_id=sentence_node.node_id,
                    display_text=(
                        f'The answer links these as {_LINK_RELATION[choice].value} '
                        f'("{candidate.connective}"): "{left.display_text}" / "{right.display_text}"'
                    ),
                    status=NodeStatus.CHECKED,
                )
            )
            units.append(relation)

    return JevProposalOutcome(tree=tree, units=units, notes=notes)


async def propose_with_jev(
    decider: Decider,
    store: TextStore,
    *,
    min_prob: float,
    max_per_request: int,
    max_parallel: int,
) -> JevProposalOutcome:
    candidates = find_candidates(store)
    answers: dict[str, Answer] = {}
    if candidates:
        answers = await ask_batched(
            decider,
            build_answer_state(store),
            [cut_question(store, c) for c in candidates],
            max_per_request=max_per_request,
            max_parallel=max_parallel,
        )
    return build_tree_from_cuts(store, candidates, answers, min_prob=min_prob)
