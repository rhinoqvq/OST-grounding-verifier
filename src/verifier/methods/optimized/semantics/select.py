"""Round 2.5 (pure code): choose the final units to verify.

Per answer sentence, in order of preference:
1. its accepted decomposition units (facts, relations, kept-whole units), if
   they also pass the deterministic coverage check;
2. otherwise one whole-sentence unit (the original text loses nothing);
3. nothing, if the sentence is entirely non-claim (greetings, meta-talk).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from verifier.rag.source import Sentence, Span, TextStore, strip_list_marker
from verifier.rag.tree import Node, NodeKind, NodeStatus, SemanticTree

# a letter or digit in any script ("北京", "Москва" are content as much as "Paris")
_CONTENT_CHAR = re.compile(r"[^\W_]")
_WORD = re.compile(r"[^\W_]+")

# words that assert nothing on their own (the baseline's function-word list; yes/no
# and negations are not in it): a sentence of only these, like a dangling "is:"
# before a list, is not a claim
_FUNCTION_WORDS = frozenset(
    "a an the and or but if then else when while of in on at to from by for with "
    "as is are was were be been being it its this that these those there here "
    "he she they them his her their we you your i us our".split()
)

# a list enumerator the segmenter left on its own ("3.", "4.)")
_LONE_ENUMERATOR = re.compile(r"\d{1,2}[.)]+")

# an uncovered run of content this long means the decomposition missed something
_GAP_CHAR_LIMIT = 12


def _covered_mask(sentence: Sentence, spans: list[Span]) -> list[bool]:
    mask = [False] * (sentence.span.end - sentence.span.start)
    for span in spans:
        if span.doc != sentence.span.doc:
            continue
        lo = max(span.start, sentence.span.start) - sentence.span.start
        hi = min(span.end, sentence.span.end) - sentence.span.start
        for i in range(lo, max(lo, hi)):
            mask[i] = True
    return mask


def coverage_gaps(
    store: TextStore,
    sentence: Sentence,
    unit_spans: list[Span],
    non_claim_spans: list[Span],
) -> list[str]:
    """Deterministic complement to the Jev coverage question: find content runs
    of the sentence not covered by any unit anchor/inherited span or non-claim
    span. Returns the uncovered fragments (empty = fully covered)."""
    mask = _covered_mask(sentence, [*unit_spans, *non_claim_spans])
    text = sentence.text
    gaps: list[str] = []
    run_start: int | None = None
    for i, covered in enumerate([*mask, True]):
        if not covered and run_start is None:
            run_start = i
        elif covered and run_start is not None:
            fragment = text[run_start:i]
            if len(_CONTENT_CHAR.findall(fragment)) >= _GAP_CHAR_LIMIT:
                gaps.append(fragment.strip())
            run_start = None
    return gaps


def is_entirely_non_claim(
    sentence: Sentence, non_claim_spans: list[Span], *, whole_answer: bool = False
) -> bool:
    """True when the sentence asserts nothing outside its meta-discourse spans: no
    word or number is left, only function words are, or (in an answer of several
    sentences) only a list enumerator. Length is not the test: a bare "3", "80%",
    "Paris" or "北京" answer is a claim. The function words are English only."""
    mask = _covered_mask(sentence, non_claim_spans)
    rest = "".join(" " if covered else c for c, covered in zip(sentence.text, mask))
    rest = strip_list_marker(rest.strip())
    if not whole_answer and _LONE_ENUMERATOR.fullmatch(rest.strip()):
        return True
    return all(w in _FUNCTION_WORDS for w in _WORD.findall(rest.lower()))


@dataclass
class Selection:
    units: list[Node]            # final units to verify, in answer order
    notes: list[str]


def make_sentence_unit(tree: SemanticTree, sentence: Sentence) -> Node:
    node_id = f"sent::{sentence.sentence_id}"
    if node_id in tree.nodes:
        return tree.nodes[node_id]
    return tree.add(
        Node(
            node_id=node_id,
            origin="answer",
            sentence_id=sentence.sentence_id,
            kind=NodeKind.SENTENCE,
            anchor_spans=[sentence.span],
            display_text=sentence.text,
            status=NodeStatus.CHECKED,
        )
    )


def select_units(
    store: TextStore,
    tree: SemanticTree,
    accepted_units: list[Node],
    fallback_sentence_ids: set[str],
    non_claim_spans: list[Span],
) -> Selection:
    notes: list[str] = []
    by_sentence: dict[str, list[Node]] = {}
    for unit in accepted_units:
        by_sentence.setdefault(unit.sentence_id, []).append(unit)

    final: list[Node] = []
    for sentence in store.answer_sentences:
        sid = sentence.sentence_id
        if is_entirely_non_claim(
            sentence, non_claim_spans, whole_answer=len(store.answer_sentences) == 1
        ):
            notes.append(f"{sid}: non-claim; skipped")
            continue
        units = by_sentence.get(sid, [])
        if sid in fallback_sentence_ids or not units:
            if sid in fallback_sentence_ids:
                notes.append(f"{sid}: falls back to whole sentence")
            elif not units:
                notes.append(f"{sid}: no accepted units; whole sentence")
            final.append(make_sentence_unit(tree, sentence))
            continue
        # an inherited span that covers the whole sentence is judge CONTEXT,
        # not interpretation coverage: every jev fragment inherits its full
        # sentence, so counting those spans would blind this check for good
        # (measured: a dropped fragment's words escaped verification unseen)
        spans = [sp for u in units for sp in u.anchor_spans]
        spans += [
            sp
            for u in units
            for sp in u.inherited_spans
            if not (
                sp.doc == sentence.span.doc
                and sp.start <= sentence.span.start
                and sp.end >= sentence.span.end
            )
        ]
        gaps = coverage_gaps(store, sentence, spans, non_claim_spans)
        if gaps:
            notes.append(f"{sid}: uncovered content {gaps!r}; falls back to whole sentence")
            final.append(make_sentence_unit(tree, sentence))
            continue
        final.extend(sorted(units, key=lambda u: min(sp.start for sp in u.anchor_spans)))
    return Selection(units=final, notes=notes)
