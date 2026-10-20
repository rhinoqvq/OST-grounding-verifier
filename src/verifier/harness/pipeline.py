"""The harness: wires every module into one verification run.

Rounds (latency is bounded by rounds, not by question count):
  0  code   segment + limits
  1  LLM    propose answer structure            (sees the answer only)
  2  Jev    fidelity + coverage checks          (sees the answer only)
  2.5 code  select final units (fallback: whole sentence)
  3  Jev    verdicts + evidence relevance       (sees the sources)
  4  Jev    whole-sentence recheck where all split facts were SUPPORTED
  5  code   citations, reasoning, contract, response
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from verifier.harness.batching import ask_batched
from verifier.harness.config import Settings
from verifier.harness.context import (
    build_answer_state,
    build_source_state,
    check_input_limits,
)
from verifier.harness.errors import MalformedModelOutput
from verifier.harness.trace import Trace
from verifier.judge.aggregate import apply_parent_recheck, relation_verdict, sentences_needing_recheck
from verifier.judge.entail import UnitJudgment, decide_verdict, judge_questions
from verifier.models.decider import Decider
from verifier.models.openai_llm import StructuredLLM
from verifier.output.render import to_response
from verifier.output.schemas import Verdict, VerifyRequest, VerifyResponse
from verifier.rag import store as rag_store
from verifier.rag.retrieve import candidate_sentences
from verifier.rag.source import TextStore
from verifier.rag.tree import NodeKind, SemanticTree
from verifier.semantics.check import CheckOutcome, apply_check_results, check_questions
from verifier.semantics.propose import Proposal, propose_structure, proposal_to_tree
from verifier.semantics.select import make_sentence_unit, select_units


@dataclass
class Dependencies:
    settings: Settings
    llm: StructuredLLM
    decider: Decider
    conn: sqlite3.Connection | None = None


async def run_verification(request: VerifyRequest, deps: Dependencies) -> tuple[VerifyResponse, Trace]:
    trace = Trace()
    try:
        response = await _run(request, deps, trace)
        _save(deps, request, response, "ok", trace)
        return response, trace
    except Exception as exc:
        trace.note(f"failed: {type(exc).__name__}: {exc}")
        _save(deps, request, None, f"error:{type(exc).__name__}", trace)
        raise


async def _run(request: VerifyRequest, deps: Dependencies, trace: Trace) -> VerifyResponse:
    settings = deps.settings

    # round 0: deterministic preparation -------------------------------------
    r = trace.start_round("0-prepare")
    check_input_limits(request.generated_answer, request.context_passages, settings)
    store = TextStore.from_request(request.generated_answer, request.context_passages)
    trace_store = {"answer_sentences": len(store.answer_sentences),
                   "source_sentences": len(store.all_source_sentences())}
    trace.end_round(r, **trace_store)
    if not store.answer_sentences:
        trace.note("answer empty or whitespace: no claims")
        return VerifyResponse(claims=[])

    # round 1: LLM proposes the structure (answer only) ----------------------
    r = trace.start_round("1-propose")
    try:
        proposal = await propose_structure(deps.llm, store)
        tree, non_claim_spans, problems = proposal_to_tree(store, proposal)
        trace.end_round(r, units=len(proposal.units), dropped=len(problems), problems=problems[:10])
    except MalformedModelOutput as exc:
        # degraded mode: whole sentences become the units; still trustworthy
        tree, non_claim_spans, _ = proposal_to_tree(store, Proposal(units=[]))
        trace.fail_round(r, exc)
        trace.note("proposal malformed twice: falling back to whole-sentence units")

    # round 2: Jev checks the decomposition (answer only) --------------------
    answer_units = [
        n for n in tree.nodes.values() if n.kind is not NodeKind.SENTENCE and n.origin == "answer"
    ]
    if answer_units:
        r = trace.start_round("2-fidelity")
        questions = check_questions(store, tree)
        answers = await ask_batched(
            deps.decider,
            build_answer_state(store),
            questions,
            max_per_request=settings.max_questions_per_request,
            max_parallel=settings.max_parallel_requests,
        )
        outcome = apply_check_results(tree, answers, fidelity_min_prob=settings.fidelity_min_prob)
        trace.end_round(
            r,
            questions=len(questions),
            accepted=len(outcome.accepted_units),
            fallback_sentences=sorted(outcome.fallback_sentence_ids),
            notes=outcome.notes[:10],
        )
    else:
        outcome = CheckOutcome(accepted_units=[], fallback_sentence_ids=set(), notes=[])

    # round 2.5: select the final units (pure code) ---------------------------
    r = trace.start_round("2.5-select")
    selection = select_units(
        store, tree, outcome.accepted_units, outcome.fallback_sentence_ids, non_claim_spans
    )
    units = selection.units
    trace.end_round(r, units=[u.node_id for u in units], notes=selection.notes[:10])
    if not units:
        trace.note("answer contains no factual claims")
        return VerifyResponse(claims=[])

    # round 3: verdicts + evidence (sources) ----------------------------------
    context_empty = not store.all_source_sentences()
    judgments: dict[str, UnitJudgment] = {}
    if context_empty:
        trace.note("context empty: every claim UNSUPPORTED by definition")
        for unit in units:
            judgments[unit.node_id] = UnitJudgment(
                node_id=unit.node_id,
                node_version=unit.version,
                verdict=Verdict.UNSUPPORTED,
                flags=["context_empty"],
            )
    else:
        r = trace.start_round("3-judge")
        candidates = {
            u.node_id: candidate_sentences(
                store, u, u.display_text or store.slice(u.anchor_spans[0]),
                max_candidates=settings.max_candidate_sentences,
            )
            for u in units
        }
        questions = judge_questions(store, units, candidates)
        answers = await ask_batched(
            deps.decider,
            build_source_state(store),
            questions,
            max_per_request=settings.max_questions_per_request,
            max_parallel=settings.max_parallel_requests,
        )
        for unit in units:
            judgments[unit.node_id] = decide_verdict(
                unit,
                answers,
                candidates[unit.node_id],
                verdict_min_prob=settings.verdict_min_prob,
                relevance_min_prob=settings.relevance_min_prob,
                max_evidence=settings.max_evidence_sentences,
            )
        # relation claims inherit the most severe verdict of their members
        for unit in units:
            if unit.kind is NodeKind.RELATION:
                merged = relation_verdict(unit, judgments)
                if merged is not judgments[unit.node_id].verdict:
                    judgments[unit.node_id].flags.append(
                        f"relation_merged_from_{judgments[unit.node_id].verdict.value}"
                    )
                    judgments[unit.node_id].verdict = merged
                    if merged is Verdict.UNSUPPORTED:
                        judgments[unit.node_id].evidence_sentence_ids = []
        trace.end_round(r, questions=len(questions),
                        verdicts={u.node_id: judgments[u.node_id].verdict.value for u in units})

    # round 4: whole-sentence recheck (the one unreliable direction) ----------
    if not context_empty:
        groups = sentences_needing_recheck(units, judgments)
        if groups:
            r = trace.start_round("4-parent-recheck")
            sentence_units = {
                sid: make_sentence_unit(tree, store.sentence_by_id(sid)) for sid in groups
            }
            recheck_candidates = {
                node.node_id: candidate_sentences(
                    store, node, node.display_text,
                    max_candidates=settings.max_candidate_sentences,
                )
                for node in sentence_units.values()
            }
            questions = judge_questions(store, list(sentence_units.values()), recheck_candidates)
            answers = await ask_batched(
                deps.decider,
                build_source_state(store),
                questions,
                max_per_request=settings.max_questions_per_request,
                max_parallel=settings.max_parallel_requests,
            )
            overrides: list[str] = []
            for sid, sentence_node in sentence_units.items():
                sentence_judgment = decide_verdict(
                    sentence_node,
                    answers,
                    recheck_candidates[sentence_node.node_id],
                    verdict_min_prob=settings.verdict_min_prob,
                    relevance_min_prob=settings.relevance_min_prob,
                    max_evidence=settings.max_evidence_sentences,
                )
                override, notes = apply_parent_recheck(sentence_judgment, groups[sid], judgments)
                if override:
                    removed = {u.node_id for u in groups[sid]}
                    units = [u for u in units if u.node_id not in removed]
                    units.append(sentence_node)
                    judgments[sentence_node.node_id] = sentence_judgment
                    overrides.append(sid)
                    for note in notes:
                        trace.note(note)
            trace.end_round(r, rechecked=sorted(groups), overridden=overrides)

    # round 5: deterministic output -------------------------------------------
    r = trace.start_round("5-render")
    response, contract_report, review_notes = to_response(
        store, units, judgments, context_empty=context_empty
    )
    trace.end_round(
        r,
        claims=len(response.claims),
        contract_violations=contract_report.violations,
        downgraded=contract_report.downgraded_claims,
        review=review_notes,
    )
    _store_tree_for_save(trace, tree, judgments)
    return response


# -- persistence ---------------------------------------------------------------

def _store_tree_for_save(trace: Trace, tree: SemanticTree, judgments: dict[str, UnitJudgment]) -> None:
    trace._tree = tree  # type: ignore[attr-defined]
    trace._judgments = judgments  # type: ignore[attr-defined]


def _save(deps: Dependencies, request: VerifyRequest, response: VerifyResponse | None, status: str, trace: Trace) -> None:
    if deps.conn is None:
        return
    tree = getattr(trace, "_tree", None)
    judgments = getattr(trace, "_judgments", {})
    try:
        rag_store.save_run(
            deps.conn,
            request=request.model_dump(),
            response=response.model_dump() if response is not None else None,
            status=status,
            tree=tree,
            judgments=[j.as_dict() for j in judgments.values()],
            trace=trace.as_dict(),
        )
    except sqlite3.Error as exc:  # persistence must never break the response
        trace.note(f"save_run failed: {exc}")
