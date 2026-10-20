"""Jev-only decomposition: code proposes cut points, Jev classifies them.

The generative-LLM proposer rewrites the answer into units (flexible, slow,
can misbind pronouns). This variant never generates text at all:

1. code finds candidate cut points in each answer sentence — connectives and
   their leading punctuation (and/but/because/so/which/or/if/...). Every match
   is a candidate: no word-count filter, the cut decision belongs to Jev;
2. Jev answers one choice question per candidate: do the two sides carry
   independently checkable assertions, are they linked (cause/contrast/
   sequence), do they coordinate around a shared written-once beginning
   ("X accepts A and B" — the 'and' questions offer shared-part options with
   the assembled second claim spelled out), must the sentence stay whole
   (condition, disjunction, reported speech), or is this no boundary at all;
3. code applies the accepted cuts: fragments are verbatim answer spans; a
   linked cut also yields a relation node over its two neighbours; a shared-
   part cut gives the right fragment discontiguous anchors (shared prefix +
   its own span), both verbatim. Plain coordination yields no relation node:
   conjunction is truth-functional, the conjuncts say it all.

Fragments carry the whole sentence as inherited context, so the judge always
sees the original wording. There is no fidelity round: a verbatim fragment
cannot add information, and fragments plus connectives tile the sentence, so
coverage holds by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verifier.harness.batching import ask_batched
from verifier.harness.context import build_answer_state
from verifier.models.decider import Answer, ChoiceQuestion, Decider, YesNoQuestion
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

# Applied when BUILDING fragments (a produced fragment must be able to stand
# as an assertion), never when proposing candidates: every regex match reaches
# Jev, and a cut whose fragments fail this guard is undone (whole sentence).
_MIN_FRAGMENT_WORDS = 3

# Connectives whose questions also offer shared-part options ("X accepts A and
# B" -> carry "X accepts" over to B). Only 'and': splitting a disjunction would
# assert both sides of an 'or', which asserts neither — 'or' stays keep_whole.
_SHARE_CONNECTIVES = {"and"}
# 6 covers subject + copula + participle + preposition ("A tied election is
# resolved by ..."); a live a12 run showed 4 forces a meaning-bending prefix.
_MAX_SHARE_PREFIX_TOKENS = 6

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


def _share_options(store: TextStore, sentence: Sentence, candidate: CutCandidate) -> dict[str, str]:
    """Extra options for and/or markers: the two sides may coordinate around a
    beginning written only once ("X accepts A and B"). Each option spells out
    the assembled second claim for Jev to judge; the key carries the prefix's
    end offset so code can cut the spans afterwards without parsing anything.
    Jev only ever chooses — assembling the claim from verbatim spans is code's
    job, and a collective reading (totals, 'together') belongs in keep_whole."""
    if candidate.connective not in _SHARE_CONNECTIVES:
        return {}
    left_tokens = [
        t for t in store.answer_tokens
        if sentence.span.start <= t.span.start and t.span.end <= candidate.marker_start
    ]
    right = store.answer[candidate.marker_end : sentence.span.end].strip()
    if len(left_tokens) < 2 or not right:
        return {}
    options: dict[str, str] = {}
    for i in range(1, min(len(left_tokens), _MAX_SHARE_PREFIX_TOKENS + 1)):
        end = left_tokens[i - 1].span.end
        prefix = store.answer[sentence.span.start : end]
        options[f"share_{end}"] = (
            "LEFT and RIGHT coordinate around a shared beginning written only once; splitting is "
            f'correct and the second claim then reads: "{prefix} {right}"'
        )
    return options


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
        options={**CUT_OPTIONS, **_share_options(store, sentence, candidate)},
    )


@dataclass(frozen=True)
class RecombCheck:
    """Recombination fidelity check for one accepted shared-part cut: the only
    operation on this path that produces a sentence the answer never wrote, so
    the only place the fidelity round comes back ("do the pieces together say
    exactly what the original says"). Failing it undoes the cut."""

    question_id: str       # recomb::<cut question id>
    cut_question_id: str
    region_text: str       # the verbatim answer region the two fragments tile
    claim_texts: list[str]


def recomb_question(check: RecombCheck) -> YesNoQuestion:
    numbered = "\n".join(f'{i + 1}. "{t}"' for i, t in enumerate(check.claim_texts))
    return YesNoQuestion(
        question_id=check.question_id,
        instructions=(
            f'This part of the answer reads: "{check.region_text}"\n'
            f"It was split into separate claims, carrying the shared beginning over:\n{numbered}\n"
            "Do these claims TOGETHER assert exactly what that part asserts — nothing added, "
            "nothing lost, no change of meaning? Watch for collective readings that break when "
            "split (totals, 'together', 'between them', mixtures)."
        ),
    )


@dataclass
class JevProposalOutcome:
    tree: SemanticTree
    units: list[Node]
    notes: list[str]
    shared_checks: list[RecombCheck] = field(default_factory=list)


def _containing_unit(units: list[Node], position: int) -> Node | None:
    best: Node | None = None
    for unit in units:
        for span in unit.anchor_spans:
            if span.start <= position < span.end:
                return unit
            if span.start <= position and (best is None or span.start > best.anchor_spans[0].start):
                best = unit
    return best


def _attach_diagnostics(store: TextStore, host: Node, markers: list[CutCandidate], notes: list[str]) -> None:
    """Split the host's anchor at keep_whole markers into diagnostic parts.
    Parts are never output and never truth-judged; each gets a grounding
    question in the judge round, and an ungrounded part caps the host's
    verdict at UNSUPPORTED (grounding composes across any structure)."""
    anchor = host.anchor_spans[0]
    inside = sorted(
        (m for m in markers if anchor.start < m.marker_start < anchor.end),
        key=lambda m: m.marker_start,
    )
    if not inside:
        return
    bounds = [anchor.start, *[m.marker_start for m in inside], anchor.end]
    starts = [anchor.start, *[m.marker_end for m in inside]]
    parts: list[Span] = []
    for i in range(len(inside) + 1):
        span = _trim_span(store, starts[i], bounds[i + 1])
        if span is not None and _word_count(store.slice(span)) >= 2:
            parts.append(span)
    if len(parts) >= 2:
        host.diagnostic_spans.extend(parts)
        notes.append(
            f"{host.node_id}: {len(parts)} diagnostic parts at "
            f"{'/'.join(m.connective for m in inside)}"
        )


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
    rejected_shared: set[str] | None = None,
) -> JevProposalOutcome:
    tree = SemanticTree(store=store)
    notes: list[str] = []
    units: list[Node] = []
    shared_checks: list[RecombCheck] = []

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
        shared_prefix_end: dict[str, int] = {}
        kept_whole: list[CutCandidate] = []
        for candidate in (c for c in candidates if c.sentence_id == sentence.sentence_id):
            answer = answers.get(candidate.question_id)
            if answer is None or not answer.probabilities:
                continue
            # 'and' questions carry share_* options, and two_facts / share_* are
            # the same READING of the marker ("two facts here"), differing only
            # in how the fragments get assembled. Mass splits between them, so
            # the cut decision sums the family and compares it to the strongest
            # rival reading (relative majority); argmax then picks the style.
            # The assembled style is backstopped by an ABSOLUTE gate — the
            # recombination fidelity check — which is what licenses deciding
            # the cut itself by majority instead of an absolute bar.
            share = {k: v for k, v in answer.probabilities.items() if k.startswith("share_")}
            share_sum = sum(share.values())
            if share:
                two_facts_p = answer.probabilities.get("two_facts", 0.0)
                split_support = share_sum + two_facts_p
                rival_max = max(
                    (
                        v for k, v in answer.probabilities.items()
                        if not k.startswith("share_") and k != "two_facts"
                    ),
                    default=0.0,
                )
                if split_support >= rival_max:
                    if share_sum >= two_facts_p:
                        if rejected_shared and candidate.question_id in rejected_shared:
                            notes.append(
                                f"{candidate.question_id}: shared_part cut undone "
                                "(recombination check failed)"
                            )
                            continue
                        best = max(share, key=share.__getitem__)
                        accepted.append((candidate, "shared_part"))
                        shared_prefix_end[candidate.question_id] = int(best.rsplit("_", 1)[1])
                        notes.append(
                            f"{candidate.question_id}: cut as shared_part "
                            f"(family={split_support:.2f} vs rival={rival_max:.2f}, {best})"
                        )
                    else:
                        accepted.append((candidate, "two_facts"))
                        notes.append(
                            f"{candidate.question_id}: cut as two_facts "
                            f"(family={split_support:.2f} vs rival={rival_max:.2f})"
                        )
                    continue
            if answer.choice is None:
                continue
            p = answer.probabilities.get(answer.choice, 0.0)
            if answer.choice in ("two_facts", *_LINK_RELATION) and p >= min_prob:
                accepted.append((candidate, answer.choice))
                notes.append(f"{candidate.question_id}: cut as {answer.choice} (p={p:.2f})")
            elif answer.choice == "keep_whole" and p >= min_prob:
                # downgrade to a DIAGNOSTIC cut: the unit stays whole for the
                # verdict, but each side gets a grounding check later
                kept_whole.append(candidate)
                notes.append(f"{candidate.question_id}: keep_whole -> diagnostic parts (p={p:.2f})")

        if not accepted:
            if kept_whole:
                _attach_diagnostics(store, sentence_node, kept_whole, notes)
            continue  # select_units falls back to the whole sentence (same node)

        accepted.sort(key=lambda t: t[0].marker_start)
        bounds = [sentence.span.start, *[c.marker_start for c, _ in accepted], sentence.span.end]
        starts = [sentence.span.start, *[c.marker_end for c, _ in accepted]]
        fragment_nodes: list[Node | None] = []
        for i in range(len(accepted) + 1):
            span = _trim_span(store, starts[i], bounds[i + 1])
            # the fragment right of a shared-part cut borrows the written-once
            # beginning: discontiguous anchors, both pieces verbatim
            prefix_span: Span | None = None
            if i > 0 and accepted[i - 1][1] == "shared_part":
                end = shared_prefix_end[accepted[i - 1][0].question_id]
                prefix_span = _trim_span(store, sentence.span.start, end)
            own_text = store.slice(span) if span is not None else ""
            display = (
                f"{store.slice(prefix_span)} {own_text}"
                if prefix_span is not None and span is not None
                else own_text
            )
            if span is None or _word_count(display) < _MIN_FRAGMENT_WORDS:
                fragment_nodes.append(None)
                notes.append(f"{sentence.sentence_id}: fragment {i} too small; kept inside neighbours")
                continue
            node = tree.add(
                Node(
                    node_id=f"frag::{sentence.sentence_id}::{i}",
                    origin="answer",
                    sentence_id=sentence.sentence_id,
                    kind=NodeKind.FACT,
                    anchor_spans=[prefix_span, span] if prefix_span is not None else [span],
                    inherited_spans=[sentence.span],  # judge always sees the original sentence
                    parent_id=sentence_node.node_id,
                    display_text=display,
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
            if kept_whole:
                _attach_diagnostics(store, sentence_node, kept_whole, notes)
            continue
        units.extend(made)

        # every surviving shared-part cut gets a recombination fidelity check:
        # the assembled claim is the one sentence the answer never wrote
        for i, (candidate, kind) in enumerate(accepted):
            if kind != "shared_part":
                continue
            left_node, right_node = fragment_nodes[i], fragment_nodes[i + 1]
            if left_node is None or right_node is None:
                continue
            shared_checks.append(
                RecombCheck(
                    question_id=f"recomb::{candidate.question_id}",
                    cut_question_id=candidate.question_id,
                    region_text=store.answer[starts[i] : bounds[i + 2]].strip(),
                    claim_texts=[left_node.display_text, right_node.display_text],
                )
            )

        # keep_whole markers inside a fragment become that fragment's diagnostics
        for candidate in kept_whole:
            host = _containing_unit(made, candidate.marker_start) or sentence_node
            _attach_diagnostics(store, host, [candidate], notes)

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

    return JevProposalOutcome(tree=tree, units=units, notes=notes, shared_checks=shared_checks)


async def propose_with_jev(
    decider: Decider,
    store: TextStore,
    *,
    min_prob: float,
    recomb_min_prob: float,
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
    outcome = build_tree_from_cuts(store, candidates, answers, min_prob=min_prob)
    if not outcome.shared_checks:
        return outcome
    # recombination round: only fires when a shared-part cut was accepted; the
    # check sees the answer only, like everything else in decomposition
    recomb = await ask_batched(
        decider,
        build_answer_state(store),
        [recomb_question(c) for c in outcome.shared_checks],
        max_per_request=max_per_request,
        max_parallel=max_parallel,
    )
    rejected = {
        check.cut_question_id
        for check in outcome.shared_checks
        if ((recomb.get(check.question_id) or Answer("")).yes_prob or 0.0) < recomb_min_prob
    }
    if not rejected:
        return outcome
    return build_tree_from_cuts(
        store, candidates, answers, min_prob=min_prob, rejected_shared=rejected
    )
