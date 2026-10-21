"""Two-tree check: map answer-tree nodes onto source-tree nodes.

A fact unit is mapped when some source leaf — or, falling back, the source
sentence that leaf belongs to — entails it; the question is the ordinary
three-way verdict, asked about that one source node, with its passage in the
state only to resolve references. A relation unit is mapped when a source
relation of the same type has both sides matching (the staged side checks,
with the source relation's sentence as the stated link). Only SUPPORTED is
settled here; everything left unmapped goes to the whole-source judge.
"""

from __future__ import annotations

import re
from collections import defaultdict

from verifier.harness.context import build_source_state, describe_unit, estimate_tokens
from verifier.models.decider import Answer, ChoiceQuestion, Question, YesNoQuestion
from verifier.output.schemas import Verdict
from verifier.rag.retrieve import top_by_overlap
from verifier.rag.source import Sentence, TextStore
from verifier.rag.tree import Node
from verifier.methods.optimized.judge.entail import (
    VERDICT_OPTIONS,
    UnitJudgment,
    decide_relation_verdict,
    relation_stage2_questions,
)
from verifier.methods.optimized.semantics.source_tree import SourceTree, SourceUnit

_LEAF_PREFIX = "m::"
# A claim about the passages themselves ("Passage 3 does not mention...",
# "(passage 2 & 3)") is about how the source is laid out, which no single
# source node states; it goes straight to the whole-source judge.
_PASSAGE_TALK = re.compile(r"\bpassages?\b", re.IGNORECASE)


def unit_text(store: TextStore, unit: Node) -> str:
    return unit.display_text or " ".join(store.slice(sp) for sp in unit.anchor_spans)


def mappable(store: TextStore, unit: Node) -> bool:
    anchors = " ".join(store.slice(sp) for sp in unit.anchor_spans)
    return not _PASSAGE_TALK.search(f"{unit.display_text or ''} {anchors}")


def presence_question(store: TextStore, unit: Node) -> YesNoQuestion:
    """The grounding obligation for a mapped unit without diagnostic parts:
    the unit as a whole is the one part whose presence is checked."""
    return YesNoQuestion(
        question_id=f"g::{unit.node_id}::whole",
        instructions=(
            f'The answer contains this claim: "{unit_text(store, unit)}"\n'
            "Question: does the source contain this claim's content — stated anywhere, including "
            "inside a conditional, an alternative/option, or reported speech?"
        ),
    )


def mapping_question(store: TextStore, unit: Node, source: SourceUnit, parent: Sentence | None) -> ChoiceQuestion:
    where = f' (part of source sentence [{parent.sentence_id}]: "{parent.text}")' if parent is not None else ""
    return ChoiceQuestion(
        question_id=f"{_LEAF_PREFIX}{unit.node_id}::{source.unit_id}",
        instructions=(
            f'Source unit [{source.unit_id}]{where}: "{source.text}"\n'
            f"{describe_unit(store, unit)}\n"
            "Judge the claim against THIS source unit only; the rest of the passage in the state is "
            "there to resolve references (pronouns, shared subjects), not as extra evidence."
        ),
        options=VERDICT_OPTIONS,
    )


def passage_state(store: TextStore, passage: int, focus: set[str], max_tokens: int) -> str:
    """The passage as the state; a long passage is cut to the focus sentences
    and their neighbours until it fits."""
    sentences = store.passage_sentences[passage]
    state = build_source_state(store, sentences)
    if estimate_tokens(state) <= max_tokens:
        return state
    positions = [i for i, s in enumerate(sentences) if s.sentence_id in focus]
    for radius in (8, 4, 2, 1, 0):
        keep = sorted({j for i in positions for j in range(i - radius, i + radius + 1) if 0 <= j < len(sentences)})
        state = build_source_state(store, [sentences[j] for j in keep])
        if estimate_tokens(state) <= max_tokens:
            break
    return state


def fact_mapping_plan(
    store: TextStore, tree: SourceTree, units: list[Node], *, leaves_per_unit: int
) -> dict[int, list[tuple[Question, str]]]:
    """Per passage: (mapping question, source sentence id it focuses on)."""
    leaves = tree.leaves()
    by_id = {u.unit_id: u for u in leaves}
    as_sentences = [store.sentence_by_id(u.unit_id) for u in leaves]
    plan: dict[int, list[tuple[Question, str]]] = defaultdict(list)
    for unit in units:
        if not mappable(store, unit):
            continue
        picked = top_by_overlap(unit_text(store, unit), as_sentences, leaves_per_unit)
        asked: set[str] = set()
        for sentence in picked:
            leaf = by_id[sentence.sentence_id]
            parent = store.sentence_by_id(leaf.sentence_id)
            if leaf.unit_id not in asked:
                fragment = leaf.unit_id != parent.sentence_id
                plan[leaf.passage].append(
                    (mapping_question(store, unit, leaf, parent if fragment else None), parent.sentence_id)
                )
                asked.add(leaf.unit_id)
            if parent.sentence_id not in asked:
                whole = SourceUnit(parent.sentence_id, leaf.kind, leaf.passage, parent.sentence_id,
                                   parent.span, parent.text)
                plan[leaf.passage].append((mapping_question(store, unit, whole, None), parent.sentence_id))
                asked.add(parent.sentence_id)
    return plan


def decide_fact_mapping(unit: Node, answers: dict[str, Answer], *, min_prob: float) -> UnitJudgment | None:
    """SUPPORTED with the entailing node as evidence, preferring a fragment over
    its whole sentence (the finer citation); None when nothing maps."""
    prefix = f"{_LEAF_PREFIX}{unit.node_id}::"
    best: tuple[int, float, str, Answer] | None = None
    for qid, answer in answers.items():
        if not qid.startswith(prefix) or not answer.probabilities:
            continue
        p = answer.probabilities.get("supported", 0.0)
        top = max(answer.probabilities, key=answer.probabilities.__getitem__)
        if top != "supported" or p < min_prob:
            continue
        source_id = qid[len(prefix):]
        rank = (1 if ".f" in source_id else 0, p)
        if best is None or rank > best[:2]:
            best = (*rank, source_id, answer)
    if best is None:
        return None
    _, p, source_id, answer = best
    return UnitJudgment(
        node_id=unit.node_id,
        node_version=unit.version,
        verdict=Verdict.SUPPORTED,
        probabilities=dict(answer.probabilities),
        confidence=answer.confidence,
        evidence_sentence_ids=[source_id],
        flags=[f"tree_map({source_id},p={p:.2f})"],
    )


def relation_mapping_plan(
    store: TextStore, tree: SourceTree, units: list[Node], member_text: dict[str, str]
) -> dict[str, tuple[SourceUnit, list[Question]]]:
    """Per answer relation: the best same-type source relation and the staged
    side questions against its sentence."""
    plan: dict[str, tuple[SourceUnit, list[Question]]] = {}
    relations = tree.relations()
    for unit in units:
        same = [r for r in relations if r.relation_type is unit.relation_type]
        if not same or not mappable(store, unit):
            continue
        pseudo = [Sentence(r.unit_id, r.span, r.text) for r in same]
        best_id = top_by_overlap(unit_text(store, unit), pseudo, 1)[0].sentence_id
        rel = next(r for r in same if r.unit_id == best_id)
        left = member_text.get(unit.member_ids[0], "") if unit.member_ids else ""
        right = member_text.get(unit.member_ids[1], "") if len(unit.member_ids) > 1 else ""
        link = store.sentence_by_id(rel.sentence_id)
        plan[unit.node_id] = (rel, relation_stage2_questions(store, unit, link, left, right))
    return plan


def decide_relation_mapping(
    unit: Node, rel: SourceUnit, answers: dict[str, Answer], *, min_prob: float
) -> UnitJudgment | None:
    """The source relation's sentence plays the stated link that stage 1 would
    otherwise have to find; the staged composition decides the rest."""
    staged = dict(answers)
    staged[f"rel1::{unit.node_id}"] = Answer(
        f"rel1::{unit.node_id}", choice=rel.sentence_id, probabilities={rel.sentence_id: 1.0}
    )
    judgment = decide_relation_verdict(unit, staged, min_prob=min_prob)
    if judgment.verdict is not Verdict.SUPPORTED:
        return None
    judgment.probabilities.pop(rel.sentence_id, None)
    judgment.flags.append(f"tree_relation({rel.unit_id})")
    return judgment


def isomorphic(group: list[Node], judgments: dict[str, UnitJudgment], store: TextStore) -> bool:
    """Every unit of an answer sentence mapped onto nodes of ONE source
    sentence: the split's structure is preserved there, so the whole-sentence
    recheck adds nothing."""
    parents: set[str] = set()
    for unit in group:
        judgment = judgments.get(unit.node_id)
        if judgment is None or not any(f.startswith(("tree_map(", "tree_relation(")) for f in judgment.flags):
            return False
        for eid in judgment.evidence_sentence_ids:
            parents.add(eid.split(".f", 1)[0].split(".r", 1)[0])
    return len(parents) == 1
