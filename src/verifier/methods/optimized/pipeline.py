"""The harness: wires every module into one verification run.

Rounds (latency is bounded by rounds, not by question count):
  0  code   segment + limits
  1  Jev    propose answer structure: code cut points, Jev classifies them
            (default proposer; with proposer=llm a generative model proposes
            and round 2 checks it)              (sees the answer only)
  2  Jev    fidelity + coverage checks          (proposer=llm only)
  2.5 code  select final units (fallback: whole sentence)
  3  Jev    verdicts + evidence relevance       (sees the sources)
  4  Jev    whole-sentence recheck where all split facts were SUPPORTED
  5  code   citations, reasoning, contract, response
"""

from __future__ import annotations

import asyncio

from verifier.harness.batching import ask_batched
from verifier.harness.config import Settings
from verifier.harness.context import (
    build_answer_state,
    build_source_state,
    check_input_limits,
    estimate_tokens,
)
from verifier.harness.errors import MalformedModelOutput, UpstreamError
from verifier.harness.trace import Trace
from verifier.methods.optimized.judge.aggregate import (
    apply_parent_recheck,
    most_severe,
    relation_verdict,
    sentences_needing_recheck,
)
from verifier.methods.optimized.judge.escalate import escalate_flat
from verifier.methods.optimized.judge.citation_probe import probe_citations
from verifier.methods.optimized.judge.guard import apply_sentence_guard
from verifier.methods.optimized.judge.sentence_call import whole_sentence_calls
from verifier.methods.optimized.judge.entail import (
    UnitJudgment,
    apply_grounding_gate,
    aux_premise_question,
    bridge_check_questions,
    bridge_find_questions,
    decide_bridge_rescue,
    decide_opposition_filter,
    decide_relation_verdict,
    decide_verdict,
    discharge_question,
    grounding_question,
    judge_questions,
    opposition_question,
    relation_link_question,
    relation_stage2_questions,
    staged_link_choice,
    verbatim_in_source,
)
from verifier.methods.optimized.judge.treemap import (
    decide_fact_mapping,
    decide_relation_mapping,
    fact_mapping_plan,
    isomorphic,
    passage_state,
    presence_question,
    relation_mapping_plan,
)
from verifier.output.render import to_response
from verifier.output.schemas import Verdict, VerifyRequest, VerifyResponse
from verifier.rag import store as rag_store
from verifier.rag.retrieve import candidate_sentences, top_by_overlap
from verifier.rag.source import TextStore
from verifier.rag.tree import Node, NodeKind, RelationType, SemanticTree
from verifier.methods.optimized.semantics.check import CheckOutcome, apply_check_results, check_questions
from verifier.methods.optimized.semantics.propose import Proposal, propose_structure, proposal_to_tree
from verifier.methods.optimized.semantics.propose_jev import propose_with_jev
from verifier.methods.optimized.semantics.select import make_sentence_unit, select_units
from verifier.methods.optimized.semantics.source_tree import SourceTree, build_source_tree


from verifier.harness.pipeline import Dependencies, _save


def _admit_upgrade(unit, judgment, answers, settings, store, trace: Trace, label: str) -> None:
    """The one admission point for every K2 upgrade (subtraction audit R6:
    rescue RULES stay separate, the submission obligations live once). Every
    upgraded verdict re-satisfies the grounding obligation here (round-2 audit
    B04) and leaves its decision note; no rescue may bypass this door."""
    apply_grounding_gate(
        unit, judgment, answers,
        grounding_min_prob=settings.grounding_min_prob, store=store,
        skip_unconditional_antecedent=settings.gate_skip_unconditional_antecedent,
        skip_non_assertive=settings.gate_skip_non_assertive_parts,
    )
    trace.note(f"{label}: {unit.node_id} -> {judgment.verdict.value}")


def _bounded_source_state(store: TextStore, units: list[Node], settings: Settings) -> str:
    """The whole source when it fits the decider's request budget; otherwise
    the sentences most relevant to the units, the largest such set that fits."""
    state = build_source_state(store)
    if estimate_tokens(state) <= settings.max_state_tokens:
        return state
    sentences = store.all_source_sentences()
    texts = [u.display_text or " ".join(store.slice(sp) for sp in u.anchor_spans) for u in units]
    for k in (40, 20, 10, 5, 2):
        keep = {s.sentence_id for t in texts for s in top_by_overlap(t, sentences, k)}
        state = build_source_state(store, [s for s in sentences if s.sentence_id in keep])
        if estimate_tokens(state) <= settings.max_state_tokens:
            break
    return state


async def _map_onto_source(
    deps: Dependencies,
    store: TextStore,
    tree: SemanticTree,
    source_tree: SourceTree,
    fact_units: list[Node],
    relation_units: list[Node],
    source_state: str,
) -> dict[str, UnitJudgment]:
    """Two-tree check: answer nodes that map onto source nodes are SUPPORTED
    here; the rest are left to the whole-source judge."""
    settings = deps.settings
    fact_plan = fact_mapping_plan(store, source_tree, fact_units, leaves_per_unit=settings.tree_map_leaves)
    member_text = {nid: node.display_text or "" for nid, node in tree.nodes.items()}
    relation_plan = relation_mapping_plan(store, source_tree, relation_units, member_text)
    by_passage: dict[int, tuple[set[str], list]] = {}
    for passage, items in fact_plan.items():
        focus, questions = by_passage.setdefault(passage, (set(), []))
        for question, sentence_id in items:
            focus.add(sentence_id)
            questions.append(question)
    for rel, questions in relation_plan.values():
        focus, passage_questions = by_passage.setdefault(rel.passage, (set(), []))
        focus.add(rel.sentence_id)
        passage_questions.extend(questions)

    async def ask(passage: int, focus: set[str], questions: list) -> dict:
        return await ask_batched(
            deps.decider,
            passage_state(store, passage, focus, settings.max_state_tokens),
            questions,
            max_per_request=settings.max_questions_per_request,
            max_parallel=settings.max_parallel_requests,
        )

    answers: dict = {}
    for part in await asyncio.gather(*(ask(p, f, q) for p, (f, q) in by_passage.items())):
        answers.update(part)

    mapped: dict[str, UnitJudgment] = {}
    for unit in fact_units:
        judgment = decide_fact_mapping(unit, answers, min_prob=settings.verdict_min_prob)
        if judgment is not None:
            mapped[unit.node_id] = judgment
    for unit in relation_units:
        if unit.node_id in relation_plan:
            judgment = decide_relation_mapping(
                unit, relation_plan[unit.node_id][0], answers, min_prob=settings.verdict_min_prob
            )
            if judgment is not None:
                mapped[unit.node_id] = judgment

    # every mapped fact owes the grounding obligation against the whole source:
    # per diagnostic part where it has them, as one part otherwise; a failed
    # check withdraws the mapping and the whole-source judge decides
    gated = [u for u in fact_units if u.node_id in mapped]
    if gated:
        questions = []
        for u in gated:
            if u.diagnostic_spans:
                questions += [grounding_question(store, u, i) for i in range(len(u.diagnostic_spans))]
            else:
                questions.append(presence_question(store, u))
        g_answers = await ask_batched(
            deps.decider,
            source_state,
            questions,
            max_per_request=settings.max_questions_per_request,
            max_parallel=settings.max_parallel_requests,
        )
        for unit in gated:
            if unit.diagnostic_spans:
                apply_grounding_gate(
                    unit, mapped[unit.node_id], g_answers,
                    grounding_min_prob=settings.grounding_min_prob, store=store,
                    skip_unconditional_antecedent=settings.gate_skip_unconditional_antecedent,
                    skip_non_assertive=settings.gate_skip_non_assertive_parts,
                )
                grounded = mapped[unit.node_id].verdict is Verdict.SUPPORTED
            else:
                answer = g_answers.get(f"g::{unit.node_id}::whole")
                p = answer.yes_prob if answer is not None and answer.yes_prob is not None else 0.0
                grounded = p >= settings.grounding_min_prob or verbatim_in_source(
                    store, " ".join(store.slice(sp) for sp in unit.anchor_spans)
                )
            if not grounded:
                del mapped[unit.node_id]
    return mapped


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

    # the source tree depends on the source alone: it is built while the
    # answer is being decomposed. "auto" builds it only for a source too long
    # for one judge state; the note lets a run be split by the path it took
    build_tree = settings.source_tree == "on"
    if settings.source_tree == "auto" and store.all_source_sentences():
        tokens = estimate_tokens(build_source_state(store))
        build_tree = tokens > settings.max_state_tokens
        trace.note(f"source tree: auto -> {'on' if build_tree else 'off'} (state {tokens:,} tokens)")
    source_tree_task = (
        asyncio.create_task(build_source_tree(deps.decider, store, settings))
        if build_tree and store.all_source_sentences()
        else None
    )
    try:
        return await _judge(store, deps, trace, source_tree_task)
    finally:
        if source_tree_task is not None and not source_tree_task.done():
            source_tree_task.cancel()


async def _judge(
    store: TextStore, deps: Dependencies, trace: Trace, source_tree_task: asyncio.Task | None
) -> VerifyResponse:
    settings = deps.settings

    # round 1: propose the structure (answer only) ----------------------------
    # proposer "llm": generative model proposes units, Jev checks them (round 2)
    # proposer "jev": code proposes cut points, Jev classifies them; fragments
    #                 are verbatim, so no fidelity round — except the one
    #                 assembling operation (shared-part cuts), which carries its
    #                 own recombination check inside propose_with_jev
    non_claim_spans: list = []
    if settings.proposer == "jev":
        r = trace.start_round("1-propose-jev")
        jev_outcome = await propose_with_jev(
            deps.decider,
            store,
            min_prob=settings.jev_cut_min_prob,
            recomb_min_prob=settings.fidelity_min_prob,
            max_per_request=settings.max_questions_per_request,
            max_parallel=settings.max_parallel_requests,
            nc_min_prob=settings.nc_min_prob,
            whole_sentences=settings.whole_sentences,
        )
        non_claim_spans = jev_outcome.non_claim_spans
        tree = jev_outcome.tree
        outcome = CheckOutcome(
            accepted_units=jev_outcome.units, fallback_sentence_ids=set(), notes=jev_outcome.notes
        )
        trace.end_round(r, units=len(jev_outcome.units), notes=jev_outcome.notes[:10])
    else:
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

        # round 2: Jev checks the decomposition (answer only) ------------------
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
    stage_calls = None   # two-stage: the whole-sentence call, asked before round 3
    settled: set[str] = set()
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
        if settings.two_stage:
            # stage 1: the whole-sentence call (the guard's request). A sentence it supports
            # with evidence is settled for every unit in it, which cites that evidence
            # sentence; the other sentences get the full judgment below
            r = trace.start_round("2.8-stage1-sentence-call")
            calls, info = await whole_sentence_calls(store, deps)
            stage_calls = {s.sentence_id: c for s, c in zip(store.answer_sentences, calls)}
            for unit in units:
                call = stage_calls.get(unit.sentence_id)
                if call is not None and call.supported:
                    judgments[unit.node_id] = UnitJudgment(
                        node_id=unit.node_id,
                        node_version=unit.version,
                        verdict=Verdict.SUPPORTED,
                        probabilities={"supported": call.prob},
                        confidence=call.prob,
                        evidence_sentence_ids=[call.evidence],
                        flags=["two_stage_sentence_supported", "inherited_citation"],
                    )
                    settled.add(unit.node_id)
            atomic = [u for u in units if u.node_id in settled and u.kind is not NodeKind.RELATION]
            per_sentence: dict[str, int] = {}
            for u in atomic:
                per_sentence[u.sentence_id] = per_sentence.get(u.sentence_id, 0) + 1
            trace.end_round(r, **info, settled_units=len(settled),
                            supported_sentences=sum(c.supported for c in calls),
                            atomic_claims_by_sentence=per_sentence)
            if settings.citation_policy == "probe" and atomic:
                r = trace.start_round("2.9-citation-probe")
                chosen, probe_info = await probe_citations(
                    deps, store, units, {u.node_id: judgments[u.node_id].evidence_sentence_ids[0] for u in atomic}
                )
                for u in atomic:
                    judgment = judgments[u.node_id]
                    if u.node_id in chosen:
                        judgment.evidence_sentence_ids = [chosen[u.node_id]]
                        judgment.flags.append("citation_selected")
                    else:
                        judgment.flags.append("citation_fallback")
                trace.end_round(r, **probe_info, fallback=len(atomic) - len(chosen))
        pending = [u for u in units if u.node_id not in settled]
        r = trace.start_round("3-judge")
        # relation units skip the generic verdict/relevance path: their check is
        # staged (find the stated link, then match sides and polarity), because
        # one bounded question cannot carry that multi-hop chain
        fact_units = [u for u in pending if u.kind is not NodeKind.RELATION]
        relation_units = [u for u in pending if u.kind is NodeKind.RELATION]
        # CONTRAST asserts no mechanism of its own ('but' ~ 'and' plus rhetoric),
        # so it skips the staged check and aggregates its members instead
        staged_relations = [
            u for u in relation_units if u.relation_type is not RelationType.CONTRAST
        ]
        source_state = _bounded_source_state(store, pending or units, settings)
        source_tree = None
        if source_tree_task is not None:
            try:
                source_tree = await source_tree_task
            except UpstreamError as exc:
                # the source tree is an aid: without it every unit goes to the whole-source judge
                trace.note(f"source tree failed ({type(exc).__name__}); units judged against the whole source")
        if source_tree is not None:
            mapped = await _map_onto_source(
                deps, store, tree, source_tree, fact_units, staged_relations, source_state
            )
            judgments.update(mapped)
            trace.note(
                f"source tree: {len(source_tree.leaves())} leaves, {len(source_tree.relations())} "
                f"relations; mapped {len(mapped)}/{len(fact_units) + len(staged_relations)} units"
            )
        judged_facts = [u for u in fact_units if u.node_id not in judgments]
        open_relations = [u for u in staged_relations if u.node_id not in judgments]
        candidates = {
            u.node_id: candidate_sentences(
                store, u, u.display_text or store.slice(u.anchor_spans[0]),
                max_candidates=settings.max_candidate_sentences,
            )
            for u in judged_facts
        }
        questions = judge_questions(store, judged_facts, candidates)
        questions += [relation_link_question(store, u) for u in open_relations]

        # hybrid evidence selector: embedding ranking as a prior. It is gated:
        # v1/v2 already handle most cases without it, so the prior only runs
        # when the request shape warrants it (pre-trigger, concurrent with the
        # judge round at zero latency cost) or when a unit's evidence turned
        # out thin (post-judge rescue, a second small call).
        hybrid = settings.evidence_selector == "hybrid" and deps.embedder is not None

        async def embed_rank(target_units) -> dict[str, list[str]]:
            from verifier.models.embedder import cosine

            sentences = store.all_source_sentences()
            unit_texts = [u.display_text or store.slice(u.anchor_spans[0]) for u in target_units]
            vectors = await deps.embedder.embed([*unit_texts, *[s.text for s in sentences]])
            unit_vecs, sent_vecs = vectors[: len(target_units)], vectors[len(target_units):]
            ranked: dict[str, list[str]] = {}
            for unit, uvec in zip(target_units, unit_vecs):
                scored = sorted(
                    ((cosine(uvec, svec), s.sentence_id) for svec, s in zip(sent_vecs, sentences)),
                    key=lambda t: -t[0],
                )
                ranked[unit.node_id] = [sid for _, sid in scored]
            return ranked

        pre_trigger = hybrid and (
            settings.embed_trigger == "always"
            or (
                settings.embed_trigger in ("multi_passage", "both")
                and (
                    sum(1 for p in store.passages if p.strip()) >= 2
                    or len(store.all_source_sentences()) >= settings.embed_min_sentences
                )
            )
        )

        ask = ask_batched(
            deps.decider,
            source_state,
            questions,
            max_per_request=settings.max_questions_per_request,
            max_parallel=settings.max_parallel_requests,
        )
        embed_ranked: dict[str, list[str]] = {}
        if pre_trigger:
            answers, embed_ranked = await asyncio.gather(ask, embed_rank(pending))
        else:
            answers = await ask

        for unit in judged_facts:
            judgments[unit.node_id] = decide_verdict(
                unit,
                answers,
                candidates[unit.node_id],
                verdict_min_prob=settings.verdict_min_prob,
                keep_leaned_supported=settings.threshold_keeps_leaned_supported,
                relevance_min_prob=settings.relevance_min_prob,
                evidence_floor_prob=settings.evidence_floor_prob,
                max_evidence=settings.max_evidence_sentences,
                embed_ranked=embed_ranked.get(unit.node_id),
                embed_top_k=settings.embed_top_k,
                embed_floor_prob=settings.embed_floor_prob,
            )
            # an ungrounded diagnostic part caps SUPPORTED at UNSUPPORTED
            apply_grounding_gate(
                unit, judgments[unit.node_id], answers,
                grounding_min_prob=settings.grounding_min_prob, store=store,
            )

        # relation stage 2: for each relation whose stage 1 found a stated-link
        # sentence, three single-hop questions (sides, polarity) in one small
        # extra request; code composes the verdict
        if relation_units:
            stage2: list = []
            for unit in open_relations:
                sid, _status = staged_link_choice(
                    answers.get(f"rel1::{unit.node_id}"), min_prob=settings.verdict_min_prob
                )
                if sid is None:
                    continue
                try:
                    link_sentence = store.sentence_by_id(sid)
                except KeyError:
                    continue
                left = tree.nodes.get(unit.member_ids[0]) if unit.member_ids else None
                right = tree.nodes.get(unit.member_ids[1]) if len(unit.member_ids) > 1 else None
                stage2 += relation_stage2_questions(
                    store, unit, link_sentence,
                    left.display_text if left else "",
                    right.display_text if right else "",
                )
            if stage2:
                answers.update(
                    await ask_batched(
                        deps.decider,
                        source_state,
                        stage2,
                        max_per_request=settings.max_questions_per_request,
                        max_parallel=settings.max_parallel_requests,
                    )
                )
            # discharge + auxiliary-premise round (round-2 audit B03/B07):
            # rel6 only when the link sentence does not assert the link in its
            # own voice (a condition may be discharged by another sentence);
            # rel7 for every found link (a side match may lean on an alias
            # stated elsewhere — the citation must carry that premise)
            extra: list = []
            for unit in open_relations:
                sid, _status = staged_link_choice(
                    answers.get(f"rel1::{unit.node_id}"), min_prob=settings.verdict_min_prob
                )
                if sid is None:
                    continue
                try:
                    link_sentence = store.sentence_by_id(sid)
                except KeyError:
                    continue
                a5 = answers.get(f"rel5::{unit.node_id}")
                assert_p = (a5.yes_prob if a5 is not None and a5.yes_prob is not None else 0.0)
                if assert_p < settings.verdict_min_prob:
                    extra.append(discharge_question(store, unit, link_sentence))
                extra.append(aux_premise_question(store, unit, link_sentence))
            if extra:
                answers.update(
                    await ask_batched(
                        deps.decider, source_state, extra,
                        max_per_request=settings.max_questions_per_request,
                        max_parallel=settings.max_parallel_requests,
                    )
                )
            for unit in relation_units:
                if unit.relation_type is RelationType.CONTRAST:
                    # R5 (subtraction audit): no placeholder own-verdict. A
                    # contrast is truth-functional, so its verdict IS the
                    # members' most severe — derived here instead of writing a
                    # neutral S for the merge loop to overwrite. Label-equal by
                    # the bottom-element identity (S is minimal in the severity
                    # lattice), but no placeholder now poses as a judged S.
                    # Evidence keeps the merge loop's semantics: U cites
                    # nothing, S/C cite the member union.
                    member_js = [judgments[m] for m in unit.member_ids if m in judgments]
                    verdict = most_severe([j.verdict for j in member_js])
                    evidence: list[str] = []
                    if verdict is not Verdict.UNSUPPORTED:
                        for j in member_js:
                            for sid in j.evidence_sentence_ids:
                                if sid not in evidence:
                                    evidence.append(sid)
                    judgments[unit.node_id] = UnitJudgment(
                        node_id=unit.node_id,
                        node_version=unit.version,
                        verdict=verdict,
                        evidence_sentence_ids=evidence,
                        flags=["contrast_truth_functional"],
                    )
                elif unit in open_relations:
                    judgments[unit.node_id] = decide_relation_verdict(
                        unit, answers, min_prob=settings.verdict_min_prob
                    )
        # post-judge rescue: units whose evidence came out thin get the
        # embedding prior after all, via one extra small call
        if hybrid and not pre_trigger and settings.embed_trigger in ("thin_evidence", "both"):
            def _thin(unit) -> bool:
                j = judgments[unit.node_id]
                if any(f.startswith(("no_evidence_for_", "evidence_rank_fallback")) for f in j.flags):
                    return True
                claim_words = len((unit.display_text or "").split())
                return (
                    j.verdict is not Verdict.UNSUPPORTED
                    and len(j.evidence_sentence_ids) <= 1
                    and claim_words >= settings.embed_thin_min_words
                )

            thin_units = [u for u in judged_facts if _thin(u)]
            if thin_units:
                rescue_ranked = await embed_rank(thin_units)
                for unit in thin_units:
                    judgments[unit.node_id] = decide_verdict(
                        unit,
                        answers,
                        candidates[unit.node_id],
                        verdict_min_prob=settings.verdict_min_prob,
                        keep_leaned_supported=settings.threshold_keeps_leaned_supported,
                        relevance_min_prob=settings.relevance_min_prob,
                        evidence_floor_prob=settings.evidence_floor_prob,
                        max_evidence=settings.max_evidence_sentences,
                        embed_ranked=rescue_ranked.get(unit.node_id),
                        embed_top_k=settings.embed_top_k,
                        embed_floor_prob=settings.embed_floor_prob,
                    )
                    judgments[unit.node_id].flags.append("embed_rescue")
                    apply_grounding_gate(
                        unit, judgments[unit.node_id], answers,
                        grounding_min_prob=settings.grounding_min_prob, store=store,
                    )
                trace.note(f"embed rescue for {[u.node_id for u in thin_units]}")

        # opposition filter (K1): a confident CONTRADICTED stands only when the
        # cited text states the explicit OPPOSITE of the claim itself (R1).
        # C -> U keeps the unit flagged, so the S set and hallucination
        # coverage are untouched by construction; only the severity label is
        # corrected (measured: C outputs 25-29% correct, dominated by
        # "different entity/aspect read as exclusive"). Runs after the embed
        # rescue so a re-decided verdict cannot resurrect a demoted C.
        c_units = [
            u for u in fact_units
            if judgments[u.node_id].verdict is Verdict.CONTRADICTED
            and judgments[u.node_id].evidence_sentence_ids
        ]
        if c_units:
            answers.update(
                await ask_batched(
                    deps.decider, source_state,
                    [opposition_question(store, u, judgments[u.node_id]) for u in c_units],
                    max_per_request=settings.max_questions_per_request,
                    max_parallel=settings.max_parallel_requests,
                )
            )
            for u in c_units:
                if decide_opposition_filter(
                    judgments[u.node_id], answers, min_prob=settings.verdict_min_prob
                ):
                    trace.note(f"opposition filter: {u.node_id} C -> U")

        # bridge rescue: a flat verdict may hide a two-hop claim whose identity
        # bridge one bounded question cannot carry (measured ~0.47 on two-hop).
        # The hop is decomposed into single-hop questions; the rescue can only
        # turn a flat UNSUPPORTED into a composed S or C with both sentences
        # cited — confident ordinary verdicts are never touched.
        # A lean to S kept under the bar (low_confidence_kept_supported) is a candidate only
        # once it has lost its single-sentence evidence and become U; while it stays S it is
        # not flat and is left alone.
        flat_units = [
            u for u in fact_units
            if judgments[u.node_id].verdict is Verdict.UNSUPPORTED
            and any(f.startswith("low_confidence") for f in judgments[u.node_id].flags)
        ]
        if flat_units:
            find_qs: list = []
            for unit in flat_units:
                find_qs += bridge_find_questions(store, unit)
            answers.update(
                await ask_batched(
                    deps.decider, source_state, find_qs,
                    max_per_request=settings.max_questions_per_request,
                    max_parallel=settings.max_parallel_requests,
                )
            )
            check_qs: list = []
            for unit in flat_units:
                bid, _ = staged_link_choice(
                    answers.get(f"br1::{unit.node_id}"),
                    min_prob=settings.verdict_min_prob, none_label="no_connection",
                )
                pid, _ = staged_link_choice(
                    answers.get(f"br3::{unit.node_id}"),
                    min_prob=settings.verdict_min_prob, none_label="no_property",
                )
                if bid is None or pid is None or bid == pid:
                    continue
                try:
                    check_qs += bridge_check_questions(
                        store, unit, store.sentence_by_id(bid), store.sentence_by_id(pid)
                    )
                except KeyError:
                    continue
            if check_qs:
                answers.update(
                    await ask_batched(
                        deps.decider, source_state, check_qs,
                        max_per_request=settings.max_questions_per_request,
                        max_parallel=settings.max_parallel_requests,
                    )
                )
            for unit in flat_units:
                rescued = decide_bridge_rescue(
                    unit, answers, min_prob=settings.verdict_min_prob
                )
                if rescued is not None:
                    rescued.flags = [*judgments[unit.node_id].flags, *rescued.flags]
                    judgments[unit.node_id] = rescued
                    _admit_upgrade(unit, judgments[unit.node_id], answers, settings,
                                   store, trace, "bridge rescue")

        # member rescue (K2): a staged-matched CAUSE/SEQUENCE relation carries
        # a constructive witness (the link sentence + both side checks) that
        # the source asserts both sides (veridical connectives), so a member
        # left FLAT by its own verdict question is rescued to S with that
        # witness instead of dragging the merge — the witness-bearing judgment
        # beats the witness-free one, the same principle as the parent
        # recheck. CONTRAST has no staged witness: its members drag
        # legitimately. A confident member verdict (any C included) is never
        # touched, and a C demoted by the opposition filter is not flat, so
        # it is not rescued either.
        for unit in relation_units:
            if unit.relation_type not in (RelationType.CAUSE, RelationType.SEQUENCE):
                continue
            own = judgments[unit.node_id]
            if own.verdict is not Verdict.SUPPORTED or not any(
                f.startswith("staged_matched") for f in own.flags
            ):
                continue
            for member_id in unit.member_ids:
                member = judgments.get(member_id)
                if (
                    member is None
                    or member.verdict is not Verdict.UNSUPPORTED
                    or not any(f.startswith("low_confidence") for f in member.flags)
                ):
                    continue
                member.verdict = Verdict.SUPPORTED
                member.evidence_sentence_ids = list(own.evidence_sentence_ids)
                sid = own.evidence_sentence_ids[0] if own.evidence_sentence_ids else "?"
                member.flags.append(f"member_rescue({sid})")
                member_node = tree.nodes.get(member_id)
                if member_node is not None:
                    _admit_upgrade(member_node, member, answers, settings, store,
                                   trace, f"member rescue <- {unit.node_id} ({sid})")
                else:
                    trace.note(f"member rescue <- {unit.node_id} ({sid}): {member_id} (no node)")

        # flat-U battle (K2, different judge): facts still flat after every
        # cheaper rescue go to the generative model for a second opinion; an
        # upgrade needs a code-verified verbatim quote (see judge/escalate.py)
        if settings.flat_escalation == "llm":
            still_flat = [
                u for u in fact_units
                if judgments[u.node_id].verdict is Verdict.UNSUPPORTED
                and any(f.startswith("low_confidence") for f in judgments[u.node_id].flags)
            ]
            if still_flat:
                upgrades = await asyncio.gather(
                    *(escalate_flat(deps.llm, store, u, judgments[u.node_id]) for u in still_flat)
                )
                for u, upgraded in zip(still_flat, upgrades):
                    if upgraded:
                        _admit_upgrade(u, judgments[u.node_id], answers, settings,
                                       store, trace, "llm escalation")
                trace.note(f"llm escalation: {sum(upgrades)}/{len(still_flat)} flats rescued")

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
                    elif merged is Verdict.CONTRADICTED and not judgments[unit.node_id].evidence_sentence_ids:
                        # the C came from a member: the relation claim must cite
                        # that member's evidence or the contract will refuse it
                        for member_id in unit.member_ids:
                            member = judgments.get(member_id)
                            if member and member.verdict is Verdict.CONTRADICTED and member.evidence_sentence_ids:
                                judgments[unit.node_id].evidence_sentence_ids = list(member.evidence_sentence_ids)
                                break
        trace.end_round(r, questions=len(questions),
                        verdicts={u.node_id: judgments[u.node_id].verdict.value for u in units})

    # round 4: whole-sentence recheck (the one unreliable direction) ----------
    if not context_empty and settings.parent_recheck:
        groups = sentences_needing_recheck(units, judgments)
        # a sentence the two-stage call settled is S already: nothing to recheck
        groups = {sid: g for sid, g in groups.items() if not all(u.node_id in settled for u in g)}
        preserved = [sid for sid, group in groups.items() if isomorphic(group, judgments, store)]
        for sid in preserved:
            del groups[sid]
            trace.note(f"{sid}: every unit mapped into one source sentence; recheck skipped")
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
                source_state,
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
                    keep_leaned_supported=settings.threshold_keeps_leaned_supported,
                    relevance_min_prob=settings.relevance_min_prob,
                    evidence_floor_prob=settings.evidence_floor_prob,
                    max_evidence=settings.max_evidence_sentences,
                )
                apply_grounding_gate(
                    sentence_node, sentence_judgment, answers,
                    grounding_min_prob=settings.grounding_min_prob,
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

    # round 4b: whole-sentence guard ------------------------------------------
    if settings.sentence_guard and not context_empty:
        r = trace.start_round("4b-sentence-guard")
        if stage_calls is not None:
            # two-stage: the same call, already asked; expected to release nothing
            calls_by_sentence, info = stage_calls, {"reused_stage1_call": True}
        else:
            calls, info = await whole_sentence_calls(store, deps)
            calls_by_sentence = {s.sentence_id: c for s, c in zip(store.answer_sentences, calls)}
        released = apply_sentence_guard(units, judgments, calls_by_sentence, scope=settings.guard_scope)
        trace.end_round(r, **info, released=released)

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
