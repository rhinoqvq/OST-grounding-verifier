"""J+nc and J+ME (2026-10-23 ablations): the optimized method's way of finding evidence.

J+nc is J's judge request unchanged (same wording, same evidence question, the top choice
is the verdict) plus the meta-discourse gate (the nc request minimal_jev_plus sends). J+ME
keeps that verdict and replaces only the evidence: for a sentence judged S or C, every
candidate source sentence of the judge state gets the optimized method's relevance probe
(entail.relevance_question's wording for a whole-sentence claim), in follow-up requests
against the same state. Evidence is the top max_evidence above relevance_min_prob, else the
best one above evidence_floor_prob; an S or C with neither becomes U — decide_verdict's rule.
J's own evidence pick is not used by J+ME (it is kept in the trace for offline variants).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from verifier.harness.batching import ask_batched
from verifier.harness.context import check_input_limits
from verifier.harness.trace import Trace
from verifier.models.decider import Answer, YesNoQuestion
from verifier.output.citations import make_citation
from verifier.output.contract import DraftClaim, enforce_contract
from verifier.output.schemas import Verdict, VerifyRequest, VerifyResponse
from verifier.rag.retrieve import top_by_overlap
from verifier.rag.source import Sentence, TextStore
from verifier.methods.minimal_jev.verifier import NO_EVIDENCE, _ask_sentences, meta_discourse, verdict_question

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies


def relevance_probe(k: int, text: str, sentence: Sentence) -> YesNoQuestion:
    return YesNoQuestion(
        question_id=f"r::{k}::{sentence.sentence_id}",
        instructions=(
            f"Source sentence [{sentence.sentence_id}] reads: \"{sentence.text}\"\n"
            "Would a careful fact-checker quote this sentence as evidence when explaining why the "
            "claim below is supported or contradicted? Merely being about the same topic is not "
            f'evidence; the sentence must bear on whether the claim holds.\nClaim: "{text}"'
        ),
    )


def sufficiency_state(store: TextStore, cited: list[str]) -> str:
    """The found sentences and the sentence before each, in source order, in the usual state format."""
    from verifier.harness.context import build_source_state

    chosen: dict[str, Sentence] = {}
    for sid in cited:
        s = store.sentence_by_id(sid)
        siblings = store.passage_sentences[store.passage_index_of(s.span.doc)]
        k = next((i for i, x in enumerate(siblings) if x.sentence_id == sid), 0)
        for x in siblings[max(0, k - 1):k + 1]:
            chosen[x.sentence_id] = x
    order = {x.sentence_id: i for i, x in enumerate(store.all_source_sentences())}
    return build_source_state(store, sorted(chosen.values(), key=lambda x: order[x.sentence_id]))


def sufficiency_question(k: int, text: str, cited: list[str]) -> "ChoiceQuestion":
    from verifier.methods.optimized.judge.entail import VERDICT_OPTIONS
    from verifier.models.decider import ChoiceQuestion

    ids = ", ".join(f"[{sid}]" for sid in cited)
    return ChoiceQuestion(
        question_id=f"cs::{k}",
        instructions=(
            f"Judge this claim strictly against the source sentences {ids} in the state — never "
            "against outside knowledge. The other sentences in the state are there only to resolve "
            f'references (pronouns, shared subjects), not as extra evidence.\nClaim: "{text}"'
        ),
        options=dict(VERDICT_OPTIONS),
    )


def probe_evidence(answers: dict[str, Answer], k: int, candidates: list[Sentence], settings) -> tuple[list[str], str | None]:
    """decide_verdict's evidence rule: the top max_evidence above the relevance bar, else the
    best-ranked one above the floor (flagged), else nothing."""
    ranked = sorted(
        (((answers.get(f"r::{k}::{s.sentence_id}") or Answer("")).yes_prob or 0.0, s.sentence_id) for s in candidates),
        key=lambda t: -t[0],
    )
    chosen = [sid for p, sid in ranked[: settings.max_evidence_sentences] if p >= settings.relevance_min_prob]
    if chosen:
        return chosen, None
    if ranked and ranked[0][0] >= settings.evidence_floor_prob:
        return [ranked[0][1]], f"evidence_rank_fallback(p={ranked[0][0]:.2f})"
    return [], "no_evidence"


async def run_me(request: VerifyRequest, deps: "Dependencies", trace: Trace, *, ost_evidence: bool) -> VerifyResponse:
    settings = deps.settings
    r = trace.start_round("0-prepare")
    check_input_limits(request.generated_answer, request.context_passages, settings)
    store = TextStore.from_request(request.generated_answer, request.context_passages)
    sentences = store.answer_sentences
    trace.end_round(r, answer_sentences=len(sentences), source_sentences=len(store.all_source_sentences()))
    if not sentences:
        trace.note("answer empty or whitespace: no claims")
        return VerifyResponse(claims=[])
    texts = [s.text for s in sentences]

    # (verdict, evidence ids, note, verdict probability) per answer sentence
    decided: list[tuple[Verdict, list[str], str | None, float]]
    r = trace.start_round("1-minimal-jev-nc")
    if not store.all_source_sentences():
        trace.note("context empty: every claim UNSUPPORTED by definition")
        nc = await meta_discourse(store, deps)
        decided = [(Verdict.UNSUPPORTED, [], None, 0.0) for _ in sentences]
        trace.end_round(r)
    else:
        held: dict = {}
        nc, (answers, info) = await asyncio.gather(
            meta_discourse(store, deps), _ask_sentences(store, deps, verdict_question, held)
        )
        decided = []
        for k in range(len(sentences)):
            v, e = answers[f"v{k}"], answers[f"e{k}"]
            verdict = Verdict(v.choice)
            picked = e.choice if e.choice and e.choice != NO_EVIDENCE else None
            evidence = [picked] if picked and verdict is not Verdict.UNSUPPORTED else []
            decided.append((verdict, evidence, None, v.probabilities.get(v.choice, 0.0)))
        trace.end_round(r, **info)
        if ost_evidence:
            r = trace.start_round("2-evidence-probes")
            kept = held["kept"]
            todo = [k for k, s in enumerate(sentences)
                    if nc.get(s.sentence_id, 0.0) < settings.nc_min_prob and decided[k][0] is not Verdict.UNSUPPORTED]
            cands = {k: (kept if len(kept) <= settings.max_candidate_sentences
                         else top_by_overlap(texts[k], kept, settings.max_candidate_sentences)) for k in todo}
            questions = [relevance_probe(k, texts[k], s) for k in todo for s in cands[k]]
            probes = (await ask_batched(deps.decider, held["state"], questions,
                                        max_per_request=settings.max_questions_per_request,
                                        max_parallel=settings.max_parallel_requests)) if questions else {}
            per_sentence = []
            recovered = []  # an S that J's own pick left without evidence and the probes found some for
            for k in todo:
                verdict, j_evidence, _, p = decided[k]
                evidence, note = probe_evidence(probes, k, cands[k], settings)
                decided[k] = (verdict if evidence else Verdict.UNSUPPORTED, evidence, note, p)
                if verdict is Verdict.SUPPORTED and evidence and not j_evidence:
                    recovered.append(k)
                top = sorted((((probes.get(f"r::{k}::{s.sentence_id}") or Answer("")).yes_prob or 0.0), s.sentence_id)
                             for s in cands[k])[::-1][:5]
                per_sentence.append({"k": k, "verdict": verdict.value, "j_evidence": j_evidence,
                                     "probes": [[sid, round(q, 3)] for q, sid in top]})
            # relevant is not sufficient: a recovered S stands only if the judge, asked against the
            # found sentences alone, still says supported
            checks = await asyncio.gather(*(
                deps.decider.ask(sufficiency_state(store, decided[k][1]),
                                 [sufficiency_question(k, texts[k], decided[k][1])]) for k in recovered))
            sufficiency = {}
            for k, answers_k in zip(recovered, checks):
                probs = answers_k[f"cs::{k}"].probabilities or {}
                top = max(probs, key=probs.get, default=None)
                sufficiency[k] = (top == "supported", round(probs.get("supported", 0.0), 3))
                if top != "supported":
                    verdict, evidence, note, p = decided[k]
                    decided[k] = (Verdict.UNSUPPORTED, [], "recovered_evidence_insufficient", p)
            for row in per_sentence:
                if row["k"] in sufficiency:
                    row["recovered_check"] = sufficiency[row["k"]]
            trace.end_round(r, questions=len(questions), sentences=len(todo), recovered=len(recovered),
                            recovered_insufficient=sum(not ok for ok, _ in sufficiency.values()),
                            per_sentence=per_sentence)
    gated = {sid for sid, p in nc.items() if p >= settings.nc_min_prob}

    r = trace.start_round("3-render")
    drafts = []
    for s, (verdict, evidence, note, p) in zip(sentences, decided):
        if s.sentence_id in gated:
            continue
        citations = [make_citation(store, sid) for sid in evidence]
        drafts.append(DraftClaim(
            s.text, verdict,
            f"Judge: {verdict.value} (p={p:.2f}); evidence: {', '.join(evidence) or 'none'}"
            f"{'; ' + note if note else ''}.",
            citations,
        ))
    claims, report = enforce_contract(drafts, request.context_passages)
    trace.end_round(r, claims=len(claims), gated=len(gated), contract_violations=report.violations,
                    downgraded=report.downgraded_claims)
    return VerifyResponse(claims=claims)
