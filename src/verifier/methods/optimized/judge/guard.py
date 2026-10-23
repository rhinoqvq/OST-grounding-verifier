"""Whole-sentence guard: an alarm raised by a structural layer stands only if the judge,
asked about the whole sentence, does not call it supported.

The layers under the sentence (the grounding gate's part probes, fragments cut out of the
sentence, relation checks, the confidence threshold, the C filter) catch errors a
whole-sentence call misses, but on clean answers they mostly fire where that call says S
(QA-900, 2026-10-23: 98 of 116 gate alarms, 61 of 194 fragment alarms, 56 of 93 relation
alarms on clean answers sat in sentences the single call judged S; gate alarms hit a gold
span 2 times in 145). The guard asks the method's own whole-sentence call
(judge/sentence_call.py) once per answer and releases those alarms when it says S with
evidence; a source too large for that one request gets no guard. A whole-sentence U or C
with none of these flags is the judge's own call under the stricter wording and is left
alone, unless the scope is "all" (rc2 candidate): then it is released too, as is a
parent-recheck override.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verifier.output.schemas import Verdict
from verifier.rag.tree import Node, NodeKind

if TYPE_CHECKING:
    from verifier.methods.optimized.judge.sentence_call import SentenceCall
    from verifier.methods.optimized.judge.entail import UnitJudgment

# flag prefix -> layer name, checked in this order (the layer that changed the verdict last wins)
_FLAG_LAYERS = (("ungrounded_part", "gate"), ("low_confidence", "low_confidence"),
                ("c_filter_demoted", "c_filter"))


def guarded_layer(unit: Node, judgment: "UnitJudgment", scope: str = "structural") -> str | None:
    """The structural layer behind a non-S verdict, or None for a whole-sentence call (with
    scope "all", that call is a layer too: "parent_override" or "whole")."""
    for prefix, layer in _FLAG_LAYERS:
        if any(f.startswith(prefix) for f in judgment.flags):
            return layer
    if unit.kind is NodeKind.RELATION:
        return "relation"
    if unit.kind is NodeKind.FACT:
        return "fragment"
    if scope == "all":
        return "parent_override" if "parent_override" in judgment.flags else "whole"
    return None


def apply_sentence_guard(
    units: list[Node],
    judgments: dict[str, "UnitJudgment"],
    calls: dict[str, "SentenceCall"],
    scope: str = "structural",
) -> list[str]:
    """Release guarded alarms whose sentence the whole-sentence call supports; returns the
    released node ids. A released unit takes the call's evidence sentence, so its S carries
    a citation like any other."""
    released = []
    for unit in units:
        judgment = judgments.get(unit.node_id)
        if judgment is None or judgment.verdict is Verdict.SUPPORTED:
            continue
        layer = guarded_layer(unit, judgment, scope)
        call = calls.get(unit.sentence_id)
        if layer is None or call is None or not call.supported:
            continue
        judgment.verdict = Verdict.SUPPORTED
        judgment.evidence_sentence_ids = [call.evidence]
        judgment.flags.append(f"sentence_guard({layer})")
        released.append(unit.node_id)
    return released
