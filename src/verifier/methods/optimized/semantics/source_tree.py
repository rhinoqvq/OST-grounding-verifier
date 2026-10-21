"""Source-side semantic tree: the answer's decomposition run on the source.

Each passage (a window of whole sentences when the passage is long) goes
through the same steps as the answer — cut points, cut questions, relation
reading, recombination check, coverage check — with that passage alone as the
state. Questions, options and thresholds are the answer's; only the wording
names the text a source passage (SOURCE_VOICE), never "the answer". The
resulting nodes are mapped back onto the passage: fragments become
citable source units below sentence level, relations keep their type and
sides, sentence ids become the passage's own.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field

from verifier.harness.config import Settings
from verifier.models.decider import Answer, Decider, Question
from verifier.rag.source import Span, TextStore
from verifier.rag.tree import Node, NodeKind, RelationType
from verifier.methods.optimized.semantics.propose_jev import SOURCE_VOICE, propose_with_jev
from verifier.methods.optimized.semantics.select import select_units

# A window's text rides along in every question of its decomposition round;
# this keeps it far below the decider's per-request ceiling.
_WINDOW_CHARS = 40_000


class PrefixedDecider:
    """Keeps source-side question ids apart from the answer side's."""

    def __init__(self, inner: Decider, prefix: str):
        self.inner = inner
        self.prefix = prefix
        self.name = getattr(inner, "name", "decider")

    async def ask(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        renamed = [dataclasses.replace(q, question_id=f"{self.prefix}{q.question_id}") for q in questions]
        answers = await self.inner.ask(state, renamed)
        out: dict[str, Answer] = {}
        for q in questions:
            answer = answers.get(f"{self.prefix}{q.question_id}")
            if answer is not None:
                out[q.question_id] = dataclasses.replace(answer, question_id=q.question_id)
        return out


@dataclass
class SourceUnit:
    unit_id: str                  # citable id: "P0.S3" (whole sentence), "P0.S3.f1", "P0.S3.r0"
    kind: NodeKind
    passage: int
    sentence_id: str              # the passage sentence it lies in
    span: Span                    # verbatim extent in the passage
    text: str                     # display text (a shared part may be recomposed)
    relation_type: RelationType | None = None
    member_ids: list[str] = field(default_factory=list)


@dataclass
class SourceTree:
    units: list[SourceUnit]
    notes: list[str]

    def leaves(self) -> list[SourceUnit]:
        return [u for u in self.units if u.kind is not NodeKind.RELATION]

    def relations(self) -> list[SourceUnit]:
        return [u for u in self.units if u.kind is NodeKind.RELATION]


def _windows(store: TextStore, passage: int) -> list[tuple[int, int]]:
    sentences = store.passage_sentences[passage]
    text = store.passages[passage]
    if len(text) <= _WINDOW_CHARS:
        return [(0, len(text))]
    windows: list[tuple[int, int]] = []
    start = sentences[0].span.start
    end = start
    for s in sentences:
        if s.span.end - start > _WINDOW_CHARS and end > start:
            windows.append((start, end))
            start = s.span.start
        end = s.span.end
    windows.append((start, end))
    return windows


def _suffix(node: Node) -> str:
    return node.node_id.rsplit("::", 1)[1]


async def _build_window(
    decider: Decider, store: TextStore, passage: int, window: int,
    bounds: tuple[int, int], settings: Settings,
) -> SourceTree:
    w_start, w_end = bounds
    doc = f"context:{passage}"
    wstore = TextStore.from_request(store.passages[passage][w_start:w_end], [])
    outcome = await propose_with_jev(
        PrefixedDecider(decider, f"src{passage}.{window}|"),
        wstore,
        min_prob=settings.jev_cut_min_prob,
        recomb_min_prob=settings.fidelity_min_prob,
        max_per_request=settings.max_questions_per_request,
        max_parallel=settings.max_parallel_requests,
        nc_min_prob=settings.nc_min_prob,
        voice=SOURCE_VOICE,
    )
    selection = select_units(wstore, outcome.tree, outcome.units, set(), outcome.non_claim_spans)

    passage_sentences = store.passage_sentences[passage]

    def passage_sentence(window_sentence_id: str):
        ws = wstore.sentence_by_id(window_sentence_id)
        lo, hi = ws.span.start + w_start, ws.span.end + w_start
        for s in passage_sentences:
            if s.span.start <= lo and hi <= s.span.end:
                return s
        return min(passage_sentences, key=lambda s: abs(s.span.start - lo))

    def shift(span: Span) -> Span:
        return Span(doc, span.start + w_start, span.end + w_start)

    units: list[SourceUnit] = []
    ids: dict[str, str] = {}
    ordered = sorted(selection.units, key=lambda n: n.kind is NodeKind.RELATION)
    for node in ordered:
        sentence = passage_sentence(node.sentence_id)
        if node.kind is NodeKind.SENTENCE:
            unit = SourceUnit(sentence.sentence_id, node.kind, passage, sentence.sentence_id,
                              sentence.span, sentence.text)
        elif node.kind is NodeKind.RELATION:
            unit = SourceUnit(
                f"{sentence.sentence_id}.r{_suffix(node)}", node.kind, passage, sentence.sentence_id,
                sentence.span, node.display_text or sentence.text, relation_type=node.relation_type,
                member_ids=[ids[m] for m in node.member_ids if m in ids],
            )
        else:
            spans = [shift(sp) for sp in node.anchor_spans]
            cover = Span(doc, min(sp.start for sp in spans), max(sp.end for sp in spans))
            unit = SourceUnit(
                f"{sentence.sentence_id}.f{_suffix(node)}", node.kind, passage, sentence.sentence_id,
                cover, node.display_text or store.slice(cover),
            )
        ids[node.node_id] = unit.unit_id
        units.append(unit)
    notes = [f"{doc}[{w_start}:{w_end}] {n}" for n in [*outcome.notes, *selection.notes]]
    return SourceTree(units=units, notes=notes)


async def build_source_tree(decider: Decider, store: TextStore, settings: Settings) -> SourceTree:
    gate = asyncio.Semaphore(settings.max_parallel_requests)

    async def one(passage: int, window: int, bounds: tuple[int, int]) -> SourceTree:
        async with gate:
            return await _build_window(decider, store, passage, window, bounds, settings)

    jobs = [
        one(i, wi, bounds)
        for i in range(len(store.passages))
        if store.passage_sentences[i]
        for wi, bounds in enumerate(_windows(store, i))
    ]
    parts = await asyncio.gather(*jobs)
    tree = SourceTree(units=[u for p in parts for u in p.units], notes=[n for p in parts for n in p.notes])
    for unit in tree.units:
        if unit.kind is NodeKind.FACT:
            store.register_source_unit(unit.unit_id, unit.span)
    return tree
