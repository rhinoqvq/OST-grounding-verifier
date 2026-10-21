"""Assemble the final response from verified pieces, in answer order."""

from __future__ import annotations

from verifier.methods.optimized.judge.entail import UnitJudgment
from verifier.output.citations import citations_for_unit
from verifier.output.contract import ContractReport, DraftClaim, enforce_contract
from verifier.output.reasoning import render_reasoning
from verifier.output.schemas import Verdict, VerifyResponse
from verifier.rag.source import TextStore
from verifier.rag.tree import Node


def _unit_sort_key(unit: Node) -> tuple[int, int]:
    start = min((sp.start for sp in unit.anchor_spans), default=0)
    return (start, 0 if unit.kind.value != "relation" else 1)


def to_response(
    store: TextStore,
    units: list[Node],
    judgments: dict[str, UnitJudgment],
    *,
    context_empty: bool = False,
) -> tuple[VerifyResponse, ContractReport, list[str]]:
    drafts: list[DraftClaim] = []
    for unit in sorted(units, key=_unit_sort_key):
        judgment = judgments[unit.node_id]
        citations = (
            []
            if judgment.verdict is Verdict.UNSUPPORTED
            else citations_for_unit(store, judgment.evidence_sentence_ids)
        )
        claim_text = unit.display_text or " ... ".join(store.slice(sp) for sp in unit.anchor_spans)
        drafts.append(
            DraftClaim(
                claim_text=claim_text,
                verdict=judgment.verdict,
                reasoning=render_reasoning(unit, judgment, context_empty=context_empty),
                citations=citations,
            )
        )
    review_notes = final_review(drafts)
    claims, report = enforce_contract(drafts, store.passages)
    return VerifyResponse(claims=claims), report, review_notes


def final_review(drafts: list[DraftClaim]) -> list[str]:
    """Pre-submit sanity pass over the assembled result as a whole."""
    notes: list[str] = []
    seen: dict[str, int] = {}
    for i, draft in enumerate(drafts):
        key = draft.claim_text.strip().lower()
        if key in seen:
            notes.append(f"claim[{i}] duplicates claim[{seen[key]}]: {draft.claim_text[:60]!r}")
        else:
            seen[key] = i
        if not draft.claim_text.strip():
            notes.append(f"claim[{i}] has empty text")
    return notes
