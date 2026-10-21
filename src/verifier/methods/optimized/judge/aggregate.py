"""Verdict aggregation over the semantic tree.

Directional soundness (the whole design rests on this):
- a part CONTRADICTED / UNSUPPORTED propagates up reliably;
- all parts SUPPORTED does NOT imply the parent is supported (splitting can
  drop "the same one", a cause link, a shared scope), so such parents get
  re-checked (round 4) and the parent verdict wins when it disagrees.
"""

from __future__ import annotations

from verifier.methods.optimized.judge.entail import UnitJudgment
from verifier.output.schemas import Verdict
from verifier.rag.tree import Node, NodeKind

_SEVERITY = {Verdict.CONTRADICTED: 2, Verdict.UNSUPPORTED: 1, Verdict.SUPPORTED: 0}


def most_severe(verdicts: list[Verdict]) -> Verdict:
    return max(verdicts, key=lambda v: _SEVERITY[v]) if verdicts else Verdict.UNSUPPORTED


def relation_verdict(
    relation: Node, judgments: dict[str, UnitJudgment]
) -> Verdict:
    """A relation claim ("A because B") holds only if the relation itself and
    every member holds; its verdict is the most severe among them."""
    own = judgments[relation.node_id].verdict
    members = [judgments[m].verdict for m in relation.member_ids if m in judgments]
    return most_severe([own, *members])


def sentences_needing_recheck(
    units: list[Node], judgments: dict[str, UnitJudgment]
) -> dict[str, list[Node]]:
    """Sentences whose facts were split (>=2 fact units) and all judged
    SUPPORTED: the only unreliable direction, so the whole sentence gets one
    more question in round 4."""
    by_sentence: dict[str, list[Node]] = {}
    for unit in units:
        if unit.kind in (NodeKind.FACT, NodeKind.RELATION):
            by_sentence.setdefault(unit.sentence_id, []).append(unit)
    return {
        sid: group
        for sid, group in by_sentence.items()
        if len([u for u in group if u.kind is NodeKind.FACT]) >= 2
        and all(judgments[u.node_id].verdict is Verdict.SUPPORTED for u in group if u.node_id in judgments)
    }


def apply_parent_recheck(
    sentence_judgment: UnitJudgment,
    child_units: list[Node],
    judgments: dict[str, UnitJudgment],
) -> tuple[bool, list[str]]:
    """If the whole-sentence verdict is not SUPPORTED while every child was,
    the split lost information: the children are replaced by the sentence unit
    (caller swaps the units). Returns (override, notes)."""
    if sentence_judgment.verdict is Verdict.SUPPORTED:
        return False, []
    notes = [
        f"parent recheck overrides children {[u.node_id for u in child_units]}: "
        f"sentence verdict {sentence_judgment.verdict.value}"
    ]
    sentence_judgment.flags.append("parent_override")
    return True, notes
