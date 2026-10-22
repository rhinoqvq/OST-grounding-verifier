"""Citation probe (rc2 candidate, citation_policy="probe"). A unit the two-stage call settled as
S inherits that call's one evidence sentence, which fits the whole sentence but not always the
atomic claim. The probe asks, per atomic claim, which of a few candidates states it: the call's
evidence sentence plus the two source sentences sharing the most terms with the answer sentence.
One request per answer, its state only those candidates. The verdict never changes; a claim
whose answer is "none" or too unsure keeps the inherited sentence (citation_fallback)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from verifier.harness.batching import ask_batched
from verifier.harness.context import build_source_state
from verifier.models.decider import ChoiceQuestion
from verifier.rag.retrieve import top_by_overlap
from verifier.rag.source import Sentence, TextStore
from verifier.rag.tree import Node, NodeKind

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies

NONE = "none"


def citation_candidates(store: TextStore, sentence: Sentence, evidence_id: str, k: int = 2) -> list[Sentence]:
    """The call's evidence sentence, then the k source sentences overlapping the answer sentence most."""
    picked = [store.sentence_by_id(evidence_id)]
    for s in top_by_overlap(sentence.text, store.all_source_sentences(), k + 1):
        if len(picked) > k:
            break
        if s.sentence_id != evidence_id:
            picked.append(s)
    return picked


def citation_question(store: TextStore, unit: Node, candidates: list[Sentence]) -> ChoiceQuestion:
    text = unit.display_text or " ".join(store.slice(sp) for sp in unit.anchor_spans)
    options = {s.sentence_id: f"source sentence [{s.sentence_id}] states this claim" for s in candidates}
    options[NONE] = "none of the listed source sentences states this claim"
    return ChoiceQuestion(
        question_id=f"cite::{unit.node_id}",
        instructions=f'Claim from an answer: "{text}"\nWhich listed source sentence states this claim?',
        options=options,
    )


async def probe_citations(
    deps: "Dependencies", store: TextStore, units: list[Node], evidence_of: dict[str, str]
) -> tuple[dict[str, str], dict]:
    """evidence_of: node id -> the inherited evidence sentence, for the settled atomic units.
    Returns node id -> chosen sentence id (only where the probe picked one) and trace info."""
    settings = deps.settings
    targets = [u for u in units if u.node_id in evidence_of and u.kind is not NodeKind.RELATION]
    if not targets:
        return {}, {"questions": 0}
    candidates = {
        u.node_id: citation_candidates(store, store.sentence_by_id(u.sentence_id), evidence_of[u.node_id])
        for u in targets
    }
    pool = {s.sentence_id: s for cs in candidates.values() for s in cs}
    order = {s.sentence_id: i for i, s in enumerate(store.all_source_sentences())}
    state = build_source_state(store, sorted(pool.values(), key=lambda s: order[s.sentence_id]))
    questions = [citation_question(store, u, candidates[u.node_id]) for u in targets]
    answers = await ask_batched(
        deps.decider, state, questions,
        max_per_request=settings.max_questions_per_request,
        max_parallel=settings.max_parallel_requests,
    )
    chosen = {}
    for u in targets:
        a = answers.get(f"cite::{u.node_id}")
        if a is None or not a.choice or a.choice == NONE:
            continue
        if (a.probabilities or {}).get(a.choice, 0.0) >= settings.citation_probe_min_prob:
            chosen[u.node_id] = a.choice
    return chosen, {"questions": len(questions), "candidate_sentences": len(pool), "selected": len(chosen)}
