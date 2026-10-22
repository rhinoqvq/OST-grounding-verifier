"""Jev-only proposer: candidate generation, tree building, pipeline e2e."""

import pytest

from conftest import FakeLLM, scripted_decider, verdict_answer
from verifier.models.decider import Answer
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.rag.source import TextStore
from verifier.rag.tree import NodeKind
from verifier.methods.optimized.semantics.propose_jev import (
    build_tree_from_cuts,
    cut_question,
    find_candidates,
    propose_with_jev,
)

ANSWER = (
    "The backup failed because the disk was full. "
    "The scheduler retries hourly and alerts the on-call engineer. "
    "The cache serves stale data or it revalidates in the background."
)


@pytest.fixture
def store() -> TextStore:
    return TextStore.from_request(ANSWER, ["(src)"])


def choice(qid: str, label: str, p: float = 0.9) -> Answer:
    probs = dict.fromkeys(
        ["two_facts", "cause_link", "contrast_link", "sequence_link", "keep_whole", "not_boundary"],
        (1 - p) / 5,
    )
    probs[label] = p
    return Answer(qid, choice=label, probabilities=probs)


def test_find_candidates_picks_connectives(store):
    candidates = find_candidates(store)
    connectives = sorted(c.connective for c in candidates)
    assert connectives == ["and", "because", "or"]
    for c in candidates:
        assert store.answer[c.marker_start : c.marker_end].strip(" ,;").lower().startswith(c.connective.split()[0])


def test_build_tree_applies_cuts_and_relations(store):
    candidates = find_candidates(store)
    by_conn = {c.connective: c for c in candidates}
    answers = {
        by_conn["because"].question_id: choice(by_conn["because"].question_id, "cause_link"),
        by_conn["and"].question_id: choice(by_conn["and"].question_id, "two_facts"),
        by_conn["or"].question_id: choice(by_conn["or"].question_id, "keep_whole"),
    }
    outcome = build_tree_from_cuts(store, candidates, answers, min_prob=0.6)
    kinds = sorted((u.kind.value, u.display_text) for u in outcome.units)
    texts = [t for _, t in kinds]
    assert "The backup failed" in texts
    assert "the disk was full." in texts
    assert "The scheduler retries hourly" in texts
    assert "alerts the on-call engineer." in texts
    relations = [u for u in outcome.units if u.kind is NodeKind.RELATION]
    assert len(relations) == 1 and relations[0].relation_type.value == "cause"
    # 'or' sentence produced no units -> whole-sentence fallback happens in select
    assert not any(u.sentence_id == "A.S2" for u in outcome.units)
    # fragments are verbatim answer substrings
    for unit in outcome.units:
        if unit.kind is NodeKind.FACT:
            assert unit.display_text in store.answer


def test_low_probability_cut_is_rejected(store):
    candidates = find_candidates(store)
    answers = {c.question_id: choice(c.question_id, "two_facts", p=0.4) for c in candidates}
    outcome = build_tree_from_cuts(store, candidates, answers, min_prob=0.6)
    assert outcome.units == []


async def test_pipeline_with_jev_proposer(wal_decider, wal_proposal):
    """End-to-end with proposer=jev on the WAL example: the LLM must never be
    called, and the NFS claim must still come out CONTRADICTED."""
    from conftest import ANSWER as WAL_ANSWER, PASSAGE_0, PASSAGE_1
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    store = TextStore.from_request(WAL_ANSWER, [])
    cut_script = {}
    for c in find_candidates(store):
        # scripted: keep every sentence whole -> sentence units
        cut_script[c.question_id] = choice(c.question_id, "keep_whole")
    script = {
        **cut_script,
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "contradicted"),
        "r::sent::A.S0::P0.S2": Answer("r::sent::A.S0::P0.S2", yes_prob=0.9),
        "v::sent::A.S1": verdict_answer("v::sent::A.S1", "supported"),
        "r::sent::A.S1::P1.S0": Answer("r::sent::A.S1::P1.S0", yes_prob=0.9),
        "v::sent::A.S2": verdict_answer("v::sent::A.S2", "not_in_context"),
    }
    decider = scripted_decider(script)
    llm = FakeLLM()  # no proposal scripted: any call would fail the test
    deps = make_deps(llm, decider, proposer="jev")
    request = VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=WAL_ANSWER)
    response, trace = await run_verification(request, deps)
    assert llm.calls == [], "jev proposer must not call the generative model"
    assert [c.verdict for c in response.claims] == [
        Verdict.CONTRADICTED,
        Verdict.SUPPORTED,
        Verdict.UNSUPPORTED,
    ]
    round_names = [x["name"] for x in trace.as_dict()["rounds"]]
    assert "1-propose-jev" in round_names and "2-fidelity" not in round_names


DISJ_ANSWER = "Boil the potatoes for 15 minutes or until they are soft enough to peel."
DISJ_PASSAGE = "Boil the potatoes for about 15 minutes."


async def test_ungrounded_disjunct_caps_supported_at_unsupported():
    """Kept-whole disjunction with a fabricated disjunct: block verdict says
    SUPPORTED, but the grounding check on the invented part caps it at U."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    store = TextStore.from_request(DISJ_ANSWER, [DISJ_PASSAGE])
    candidates = find_candidates(store)
    assert [c.connective for c in candidates] == ["or"]
    qid = candidates[0].question_id
    script = {
        qid: choice(qid, "keep_whole"),
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "supported"),
        "r::sent::A.S0::P0.S0": Answer("r::sent::A.S0::P0.S0", yes_prob=0.9),
        "g::sent::A.S0::0": Answer("g::sent::A.S0::0", yes_prob=0.95),  # "Boil ... 15 minutes" grounded
        "g::sent::A.S0::1": Answer("g::sent::A.S0::1", yes_prob=0.10),  # "until soft" fabricated
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=[DISJ_PASSAGE], generated_answer=DISJ_ANSWER),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    claim = response.claims[0]
    assert claim.verdict is Verdict.UNSUPPORTED
    assert claim.citations == []
    assert "no grounding" in claim.reasoning or "UNSUPPORTED" in claim.reasoning


async def test_grounded_conditional_stays_supported():
    """Kept-whole conditional fully present in the source: both parts grounded,
    block verdict SUPPORTED survives the gate."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    answer = "The server accepts incoming requests if the recovery procedure succeeds completely."
    passage = "The server accepts incoming requests if the recovery procedure succeeds completely."
    store = TextStore.from_request(answer, [passage])
    candidates = find_candidates(store)
    assert [c.connective for c in candidates] == ["if"]
    qid = candidates[0].question_id
    script = {
        qid: choice(qid, "keep_whole"),
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "supported"),
        "r::sent::A.S0::P0.S0": Answer("r::sent::A.S0::P0.S0", yes_prob=0.9),
        # conftest default answers g:: with 0.9 -> grounded
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=[passage], generated_answer=answer),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.SUPPORTED


C20_ANSWER = "Production accepts protocol V1 and protocol V2."
C20_PASSAGES = [
    "Production traffic is routed exclusively to release Azure.",
    "Release Azure rejects protocol V1 and accepts protocol V2.",
]


def test_every_regex_match_is_a_candidate():
    """No word-count filter before Jev: the noun-coordination 'and' whose right
    side is only two words still becomes a candidate (the cut decision is
    Jev's, not code's)."""
    store = TextStore.from_request(C20_ANSWER, ["(src)"])
    assert [c.connective for c in find_candidates(store)] == ["and"]


def test_and_question_offers_shared_part_options():
    store = TextStore.from_request(C20_ANSWER, ["(src)"])
    q = cut_question(store, find_candidates(store)[0])
    share = {k: v for k, v in q.options.items() if k.startswith("share_")}
    assert set(share) == {"share_10", "share_18", "share_27"}
    assert 'reads: "Production accepts protocol V2."' in share["share_18"]
    # 'because' offers none (not a coordination); 'or' offers none either —
    # splitting a disjunction would assert both sides, which asserts neither
    for text in ("The backup failed because the disk was full.", DISJ_ANSWER):
        store2 = TextStore.from_request(text, ["(src)"])
        q2 = cut_question(store2, find_candidates(store2)[0])
        assert not any(k.startswith("share_") for k in q2.options)


def test_shared_part_cut_builds_discontiguous_fact():
    store = TextStore.from_request(C20_ANSWER, ["(src)"])
    candidates = find_candidates(store)
    cand = candidates[0]
    probs = {"share_18": 0.9, "not_boundary": 0.1}
    answers = {cand.question_id: Answer(cand.question_id, choice="share_18", probabilities=probs)}
    outcome = build_tree_from_cuts(store, candidates, answers, min_prob=0.6)
    facts = [u for u in outcome.units if u.kind is NodeKind.FACT]
    assert [f.display_text for f in facts] == [
        "Production accepts protocol V1",
        "Production accepts protocol V2.",
    ]
    assert [store.slice(sp) for sp in facts[1].anchor_spans] == [
        "Production accepts",
        "protocol V2.",
    ], "right fragment carries the written-once beginning as a second verbatim span"
    assert not any(u.kind is NodeKind.RELATION for u in outcome.units), (
        "conjunction is truth-functional: the conjuncts say it all, no relation node"
    )
    assert len(outcome.shared_checks) == 1
    check = outcome.shared_checks[0]
    assert check.region_text == C20_ANSWER
    assert check.claim_texts == [f.display_text for f in facts]


def test_share_mass_sum_rule():
    """Adjacent prefixes split the mass: 0.35+0.30 share vs 0.35 not_boundary
    still cuts (family sum beats the rival reading), prefix by argmax."""
    store = TextStore.from_request(C20_ANSWER, ["(src)"])
    cand = find_candidates(store)[0]
    probs = {"share_18": 0.35, "share_27": 0.30, "not_boundary": 0.35}
    answers = {cand.question_id: Answer(cand.question_id, choice="not_boundary", probabilities=probs)}
    outcome = build_tree_from_cuts(store, [cand], answers, min_prob=0.6)
    facts = [u for u in outcome.units if u.kind is NodeKind.FACT]
    assert len(facts) == 2 and facts[1].display_text == "Production accepts protocol V2."


def test_family_majority_accepts_c20_run0_distribution():
    """Anchored to a live Jev run on c20 (2026-10-21): shares 0.47+0.02+0.01 +
    two_facts 0.01 = 0.51 edges out not_boundary 0.49. The family majority
    nominates the cut; the recombination gate is the absolute check."""
    store = TextStore.from_request(C20_ANSWER, ["(src)"])
    cand = find_candidates(store)[0]
    probs = {"share_18": 0.47, "share_10": 0.02, "share_27": 0.01,
             "two_facts": 0.01, "not_boundary": 0.49}
    answers = {cand.question_id: Answer(cand.question_id, choice="not_boundary", probabilities=probs)}
    outcome = build_tree_from_cuts(store, [cand], answers, min_prob=0.6)
    facts = [u for u in outcome.units if u.kind is NodeKind.FACT]
    assert [f.display_text for f in facts] == [
        "Production accepts protocol V1",
        "Production accepts protocol V2.",
    ]


async def test_recombination_failure_undoes_shared_cut():
    """The assembled claim is the one sentence the answer never wrote, so it
    carries a fidelity check; failing it rebuilds without the cut."""
    store = TextStore.from_request(C20_ANSWER, ["(src)"])
    cand = find_candidates(store)[0]
    probs = {"share_18": 0.9, "not_boundary": 0.1}
    script = {
        cand.question_id: Answer(cand.question_id, choice="share_18", probabilities=probs),
        f"recomb::{cand.question_id}": Answer(f"recomb::{cand.question_id}", yes_prob=0.2),
    }
    outcome = await propose_with_jev(
        scripted_decider(script), store,
        min_prob=0.6, recomb_min_prob=0.6, max_per_request=50, max_parallel=2,
    )
    assert outcome.units == [], "failed recombination undoes the cut (whole sentence)"
    assert any("recombination check failed" in n for n in outcome.notes)


async def test_pipeline_splits_shared_coordination():
    """c20 end-to-end: one merged sentence becomes two claims with opposite
    verdicts, the assembled claim judged and cited like any other."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    store = TextStore.from_request(C20_ANSWER, C20_PASSAGES)
    cand = find_candidates(store)[0]
    probs = {"share_18": 0.9, "not_boundary": 0.1}
    script = {
        cand.question_id: Answer(cand.question_id, choice="share_18", probabilities=probs),
        # recomb:: passes via the conftest default (0.9)
        "v::frag::A.S0::0": verdict_answer("v::frag::A.S0::0", "contradicted"),
        "r::frag::A.S0::0::P1.S0": Answer("r::frag::A.S0::0::P1.S0", yes_prob=0.9),
        "v::frag::A.S0::1": verdict_answer("v::frag::A.S0::1", "supported"),
        "r::frag::A.S0::1::P1.S0": Answer("r::frag::A.S0::1::P1.S0", yes_prob=0.9),
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=C20_PASSAGES, generated_answer=C20_ANSWER),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    assert [c.claim_text for c in response.claims] == [
        "Production accepts protocol V1",
        "Production accepts protocol V2.",
    ]
    assert [c.verdict for c in response.claims] == [Verdict.CONTRADICTED, Verdict.SUPPORTED]
    assert response.claims[1].citations, "assembled claim still gets verbatim citations"


R04_ANSWER = "Because the gateway compresses large responses, those transfers finish faster."

STAGED_ANSWER = "The cache evicts entries whenever a key expires, so memory usage stays bounded."
STAGED_PASSAGES = [
    "The cache evicts entries periodically so that memory usage stays bounded. "
    "The cache also evicts entries whenever a key expires."
]


def test_sentence_initial_connective_is_a_candidate():
    """'Because A, B' used to produce no candidate (the regex wanted leading
    whitespace); now the comma is the split point and the fragments exclude
    the connective, which anchors the relation node instead."""
    store = TextStore.from_request(R04_ANSWER, ["(src)"])
    cands = find_candidates(store)
    assert [c.connective for c in cands] == ["because"]
    cand = cands[0]
    assert cand.lead_start is not None
    assert store.answer[cand.lead_start : cand.lead_end].strip() == "Because"
    answers = {cand.question_id: choice(cand.question_id, "cause_link")}
    outcome = build_tree_from_cuts(store, cands, answers, min_prob=0.6)
    facts = [u for u in outcome.units if u.kind is NodeKind.FACT]
    assert [f.display_text for f in facts] == [
        "the gateway compresses large responses",
        "those transfers finish faster.",
    ]
    rel = [u for u in outcome.units if u.kind is NodeKind.RELATION][0]
    assert store.slice(rel.anchor_spans[0]).strip() == "Because"


def _staged_relation(store):
    from verifier.rag.tree import NodeKind as NK

    cands = find_candidates(store)
    cand = [c for c in cands if c.connective == "so"][0]
    answers = {c.question_id: choice(c.question_id, "not_boundary") for c in cands}
    answers[cand.question_id] = choice(cand.question_id, "cause_link")
    outcome = build_tree_from_cuts(store, cands, answers, min_prob=0.6)
    return [u for u in outcome.units if u.kind is NK.RELATION][0]


def side(qid: str, label: str, p: float = 0.9) -> Answer:
    probs = dict.fromkeys(("matches", "opposite", "different"), (1 - p) / 2)
    probs[label] = p
    return Answer(qid, choice=label, probabilities=probs)


def test_staged_relation_composition():
    """The settled rule, enforced by code over single-hop answers: a
    'different' side (sibling trigger, widening) -> U; both sides 'matches'
    -> S with the link sentence cited; an 'opposite' side (same aspect,
    inverted value) -> C; no stated link -> U."""
    from verifier.methods.optimized.judge.entail import decide_relation_verdict

    store = TextStore.from_request(STAGED_ANSWER, STAGED_PASSAGES)
    rel = _staged_relation(store)
    rid = rel.node_id

    def link(choice_label, p=0.9):
        probs = dict.fromkeys(("P0.S0", "P0.S1", "no_such_link"), (1 - p) / 2)
        probs[choice_label] = p
        return Answer(f"rel1::{rid}", choice=choice_label, probabilities=probs)

    answers = {
        f"rel1::{rid}": link("P0.S0"),
        f"rel2::{rid}": side(f"rel2::{rid}", "different"),  # expiry != periodic
        f"rel3::{rid}": side(f"rel3::{rid}", "matches"),
        f"rel5::{rid}": Answer(f"rel5::{rid}", yes_prob=0.9),  # link is asserted
    }
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED
    assert any(f.startswith("staged_side_mismatch") for f in j.flags)
    assert j.evidence_sentence_ids == []

    answers[f"rel2::{rid}"] = side(f"rel2::{rid}", "matches")
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.SUPPORTED and j.evidence_sentence_ids == ["P0.S0"]

    # 'different' outranks 'opposite': a sibling trigger with an inverted other
    # side is still an unstated link, not a contradiction
    answers[f"rel2::{rid}"] = side(f"rel2::{rid}", "different")
    answers[f"rel3::{rid}"] = side(f"rel3::{rid}", "opposite")
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED

    answers[f"rel2::{rid}"] = side(f"rel2::{rid}", "matches")
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.CONTRADICTED and j.evidence_sentence_ids == ["P0.S0"]

    answers[f"rel1::{rid}"] = link("no_such_link")
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED and "staged_no_link" in j.flags


async def test_pipeline_staged_relation_end_to_end():
    """r03 shape through the whole pipeline: stage 1 finds the mechanism
    sentence, stage 2's left-side check kills the sibling-trigger transfer,
    the relation claim comes out U (not C) while both fragments stay S."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    store = TextStore.from_request(STAGED_ANSWER, STAGED_PASSAGES)
    cand = find_candidates(store)[0]
    rid = "rel::A.S0::0"
    script = {
        cand.question_id: choice(cand.question_id, "cause_link"),
        "v::frag::A.S0::0": verdict_answer("v::frag::A.S0::0", "supported"),
        "r::frag::A.S0::0::P0.S1": Answer("r::frag::A.S0::0::P0.S1", yes_prob=0.9),
        "v::frag::A.S0::1": verdict_answer("v::frag::A.S0::1", "supported"),
        "r::frag::A.S0::1::P0.S0": Answer("r::frag::A.S0::1::P0.S0", yes_prob=0.9),
        f"rel1::{rid}": Answer(
            f"rel1::{rid}", choice="P0.S0",
            probabilities={"P0.S0": 0.9, "P0.S1": 0.05, "no_such_link": 0.05},
        ),
        f"rel2::{rid}": side(f"rel2::{rid}", "different"),  # sibling trigger
        # rel3 passes via the conftest default (matches 0.9)
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=STAGED_PASSAGES, generated_answer=STAGED_ANSWER),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    rel_claims = [c for c in response.claims if c.claim_text.startswith("The answer links")]
    assert len(rel_claims) == 1
    assert rel_claims[0].verdict is Verdict.UNSUPPORTED
    assert "different trigger" in rel_claims[0].reasoning
    others = [c.verdict for c in response.claims if not c.claim_text.startswith("The answer links")]
    assert others == [Verdict.SUPPORTED, Verdict.SUPPORTED]


async def test_contrast_relation_is_truth_functional():
    """'but' asserts no mechanism: the relation skips the staged check, and a
    contradicted member propagates up WITH its evidence (the contract refuses a
    CONTRADICTED claim that cites nothing)."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    answer = "The scheduler retries hourly but alerts the on-call engineer."
    passages = ["The scheduler retries hourly. It never alerts anyone."]
    store = TextStore.from_request(answer, passages)
    cand = [c for c in find_candidates(store) if c.connective == "but"][0]
    script = {
        cand.question_id: choice(cand.question_id, "contrast_link"),
        "v::frag::A.S0::0": verdict_answer("v::frag::A.S0::0", "supported"),
        "r::frag::A.S0::0::P0.S0": Answer("r::frag::A.S0::0::P0.S0", yes_prob=0.9),
        "v::frag::A.S0::1": verdict_answer("v::frag::A.S0::1", "contradicted"),
        "r::frag::A.S0::1::P0.S1": Answer("r::frag::A.S0::1::P0.S1", yes_prob=0.9),
    }
    decider = scripted_decider(script)
    response, _ = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), decider, proposer="jev"),
    )
    rel = [c for c in response.claims if c.claim_text.startswith("The answer links")][0]
    assert rel.verdict is Verdict.CONTRADICTED
    assert rel.citations, "merged C inherits the contradicted member's evidence"
    asked = [q.question_id for _, questions in decider.seen for q in questions]
    assert not any(q.startswith("rel1::") for q in asked), (
        "contrast relations never reach the staged link question"
    )


BRIDGE_PASSAGES = [
    "Production traffic is routed exclusively to release Azure.",
    "Release Azure rejects protocol V1 and accepts protocol V2.",
]
BRIDGE_ANSWER = "Production accepts protocol V2."


def _flat(uid: str) -> Answer:
    return Answer(
        f"v::{uid}", choice="not_in_context",
        probabilities={"supported": 0.46, "contradicted": 0.08, "not_in_context": 0.46},
        confidence=0.4,
    )


async def test_bridge_rescue_composes_two_hop_support():
    """c20-V2 shape: the ordinary verdict is flat; the rescue finds the bridge
    and property sentences and composes SUPPORTED with both cited."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    uid = "sent::A.S0"
    script = {
        f"v::{uid}": _flat(uid),
        f"br1::{uid}": Answer(f"br1::{uid}", choice="P0.S0",
                              probabilities={"P0.S0": 0.9, "P1.S0": 0.05, "no_connection": 0.05}),
        f"br3::{uid}": Answer(f"br3::{uid}", choice="P1.S0",
                              probabilities={"P0.S0": 0.05, "P1.S0": 0.9, "no_property": 0.05}),
        # br2 (yes 0.9) and br4 (matches 0.9) pass via conftest defaults
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=BRIDGE_PASSAGES, generated_answer=BRIDGE_ANSWER),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    claim = response.claims[0]
    assert claim.verdict is Verdict.SUPPORTED
    assert {c.quote for c in claim.citations} == set(BRIDGE_PASSAGES)
    assert "identity hop" in claim.reasoning


async def test_bridge_rescue_gates_hold():
    """A partial bridge (br2 low) and a missing bridge both leave the flat
    UNSUPPORTED untouched; a confident ordinary verdict never triggers the
    rescue at all."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    uid = "sent::A.S0"
    # partial bridge: br2 says the coverage is not complete
    script = {
        f"v::{uid}": _flat(uid),
        f"br1::{uid}": Answer(f"br1::{uid}", choice="P0.S0",
                              probabilities={"P0.S0": 0.9, "P1.S0": 0.05, "no_connection": 0.05}),
        f"br3::{uid}": Answer(f"br3::{uid}", choice="P1.S0",
                              probabilities={"P0.S0": 0.05, "P1.S0": 0.9, "no_property": 0.05}),
        f"br2::{uid}": Answer(f"br2::{uid}", yes_prob=0.06),
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=BRIDGE_PASSAGES, generated_answer=BRIDGE_ANSWER),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.UNSUPPORTED

    # no bridge found (conftest default: no_connection): flat stands
    script2 = {f"v::{uid}": _flat(uid)}
    decider = scripted_decider(script2)
    response, _ = await run_verification(
        VerifyRequest(context_passages=BRIDGE_PASSAGES, generated_answer=BRIDGE_ANSWER),
        make_deps(FakeLLM(), decider, proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.UNSUPPORTED
    asked = [q.question_id for _, questions in decider.seen for q in questions]
    assert not any(q.startswith("br2::") for q in asked), "no stage B without a found pair"

    # confident verdict: the rescue never fires
    script3 = {
        f"v::{uid}": verdict_answer(f"v::{uid}", "supported"),
        f"r::{uid}::P0.S0": Answer(f"r::{uid}::P0.S0", yes_prob=0.9),
    }
    decider3 = scripted_decider(script3)
    response, _ = await run_verification(
        VerifyRequest(context_passages=BRIDGE_PASSAGES, generated_answer=BRIDGE_ANSWER),
        make_deps(FakeLLM(), decider3, proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.SUPPORTED
    asked3 = [q.question_id for _, questions in decider3.seen for q in questions]
    assert not any(q.startswith("br1::") for q in asked3)


def test_embed_prior_rescues_linking_premise(wal_store):
    """A linking premise with probe 0.4 (< 0.6 bar) joins the evidence when the
    embedding prior ranks it top-2 and it clears the 0.25 floor."""
    from verifier.methods.optimized.judge.entail import decide_verdict
    from verifier.rag.tree import Node, NodeKind, NodeStatus
    from verifier.rag.source import Span

    unit = Node(
        node_id="u1", origin="answer", sentence_id="A.S0", kind=NodeKind.FACT,
        anchor_spans=[Span("answer", 0, 10)], display_text="claim", status=NodeStatus.CHECKED,
    )
    answers = {
        "v::u1": verdict_answer("v::u1", "supported"),
        "r::u1::P0.S0": Answer("r::u1::P0.S0", yes_prob=0.9),   # direct evidence
        "r::u1::P0.S1": Answer("r::u1::P0.S1", yes_prob=0.4),   # linking premise, below bar
        "r::u1::P0.S2": Answer("r::u1::P0.S2", yes_prob=0.1),   # noise
    }
    sentences = [wal_store.all_source_sentences()[i] for i in range(3)]
    judgment = decide_verdict(
        unit, answers, sentences,
        verdict_min_prob=0.55, relevance_min_prob=0.60, evidence_floor_prob=0.30,
        max_evidence=4,
        embed_ranked=["P0.S1", "P0.S0", "P0.S2"], embed_top_k=2, embed_floor_prob=0.25,
    )
    assert "P0.S0" in judgment.evidence_sentence_ids
    assert "P0.S1" in judgment.evidence_sentence_ids, "premise rescued by embedding prior"
    assert "P0.S2" not in judgment.evidence_sentence_ids, "noise stays out (below floor)"
    assert any(f.startswith("embed_prior(P0.S1") for f in judgment.flags)


class FakeEmbedder:
    """Deterministic embedder: vector = bag of word-hash buckets."""

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in t.lower().split():
                v[hash(w) % 64] += 1.0
            out.append(v)
        return out


async def test_pre_trigger_fires_only_on_multi_passage(wal_decider, wal_proposal):
    """embed_trigger=both: single short passage -> no embedding call;
    two passages -> embedding runs (concurrently with the judge round)."""
    from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    # two passages: pre-trigger fires
    deps = make_deps(FakeLLM(proposal=wal_proposal), wal_decider, proposer="jev",
                     evidence_selector="hybrid", embed_trigger="both")
    deps.embedder = FakeEmbedder()
    await run_verification(
        VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER), deps
    )
    assert deps.embedder.calls == 1

    # one short passage, strong evidence everywhere: no embedding at all
    from conftest import scripted_decider, verdict_answer
    from verifier.models.decider import Answer

    script = {
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "supported"),
        "r::sent::A.S0::P0.S0": Answer("r::sent::A.S0::P0.S0", yes_prob=0.9),
    }
    deps2 = make_deps(FakeLLM(), scripted_decider(script), proposer="jev",
                      evidence_selector="hybrid", embed_trigger="both")
    deps2.embedder = FakeEmbedder()
    await run_verification(
        VerifyRequest(context_passages=["The cache stores recent results."],
                      generated_answer="The cache stores recent results."),
        deps2,
    )
    assert deps2.embedder.calls == 0


async def test_thin_evidence_rescue_runs_embedding(wal_decider):
    """Single passage (no pre-trigger) + long claim with no passing evidence:
    the rescue path fires one embedding call and restores the evidence."""
    from conftest import FakeLLM, scripted_decider, verdict_answer
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification
    from verifier.models.decider import Answer

    passage = "The uploader keeps retrying failed chunks until every chunk succeeds."
    answer = "The uploader keeps retrying failed chunks until every chunk of the file succeeds."
    script = {
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "supported"),
        # probe gives the only sentence 0.4: below the 0.6 bar, above 0.25 floor
        "r::sent::A.S0::P0.S0": Answer("r::sent::A.S0::P0.S0", yes_prob=0.4),
    }
    deps = make_deps(FakeLLM(), scripted_decider(script), proposer="jev",
                     evidence_selector="hybrid", embed_trigger="both")
    deps.embedder = FakeEmbedder()
    response, trace = await run_verification(
        VerifyRequest(context_passages=[passage], generated_answer=answer), deps
    )
    assert deps.embedder.calls == 1, "rescue must fire exactly one embedding call"
    claim = response.claims[0]
    assert claim.verdict is Verdict.SUPPORTED
    assert claim.citations, "rescued evidence should produce a citation"


def test_small_fragment_undoes_cut_instead_of_dropping():
    """m07 shape: 'and' + 'but' both accepted would leave a 1-word fragment
    ('failed'); the guard undoes the 'but' cut so every word stays inside a
    unit (the tiling invariant), instead of silently dropping the fragment."""
    answer = "The migration started in June and ran for seventeen weeks but failed."
    store = TextStore.from_request(answer, ["(src)"])
    cands = find_candidates(store)
    by_conn = {c.connective: c for c in cands}
    answers = {
        by_conn["and"].question_id: choice(by_conn["and"].question_id, "two_facts"),
        by_conn["but"].question_id: choice(by_conn["but"].question_id, "contrast_link"),
    }
    outcome = build_tree_from_cuts(store, cands, answers, min_prob=0.6)
    facts = [u for u in outcome.units if u.kind is NodeKind.FACT]
    assert [f.display_text for f in facts] == [
        "The migration started in June",
        "ran for seventeen weeks but failed.",
    ]
    assert any("cut undone" in n for n in outcome.notes)
    assert not any(u.kind is NodeKind.RELATION for u in outcome.units)
    # tiling invariant: every content char is inside a fragment or a marker
    covered: set[int] = set()
    for u in facts:
        for sp in u.anchor_spans:
            covered.update(range(sp.start, sp.end))
    for c in (by_conn["and"], by_conn["but"]):
        covered.update(range(c.marker_start, c.marker_end))
    missing = [i for i, ch in enumerate(store.answer) if ch.isalnum() and i not in covered]
    assert missing == []


def test_sequence_relation_order_composition():
    """Sequence is asymmetric: matching sides alone cannot tell 'A then B'
    from 'B then A' (measured: an inverted order composed S, mathprobe m09).
    The order stage decides: inverted -> C, unstated -> U, same order -> S."""
    from verifier.methods.optimized.judge.entail import decide_relation_verdict, relation_stage2_questions

    answer = "The fix was deployed, then the servers were restarted."
    passages = ["The servers were restarted first, and only then was the fix deployed."]
    store = TextStore.from_request(answer, passages)
    cands = find_candidates(store)
    cand = [c for c in cands if c.connective == "then"][0]
    answers = {c.question_id: choice(c.question_id, "not_boundary") for c in cands}
    answers[cand.question_id] = choice(cand.question_id, "sequence_link")
    outcome = build_tree_from_cuts(store, cands, answers, min_prob=0.6)
    rel = [u for u in outcome.units if u.kind is NodeKind.RELATION][0]
    rid = rel.node_id
    link = store.all_source_sentences()[0]

    # the order question is asked for sequence relations only (rel5 for all)
    qids = [q.question_id for q in relation_stage2_questions(store, rel, link, "L", "R")]
    assert qids == [f"rel2::{rid}", f"rel3::{rid}", f"rel4::{rid}", f"rel5::{rid}"]
    cause_store = TextStore.from_request(STAGED_ANSWER, STAGED_PASSAGES)
    cause_rel = _staged_relation(cause_store)
    cause_qids = [
        q.question_id
        for q in relation_stage2_questions(
            cause_store, cause_rel, cause_store.all_source_sentences()[0], "L", "R"
        )
    ]
    assert not any(q.startswith("rel4::") for q in cause_qids)

    def order(label, p=0.9):
        probs = dict.fromkeys(("left_first", "right_first", "no_order"), (1 - p) / 2)
        probs[label] = p
        return Answer(f"rel4::{rid}", choice=label, probabilities=probs)

    judge_answers = {
        f"rel1::{rid}": Answer(f"rel1::{rid}", choice="P0.S0",
                               probabilities={"P0.S0": 0.9, "no_such_link": 0.1}),
        f"rel2::{rid}": side(f"rel2::{rid}", "matches"),
        f"rel3::{rid}": side(f"rel3::{rid}", "matches"),
        f"rel4::{rid}": order("right_first"),
        f"rel5::{rid}": Answer(f"rel5::{rid}", yes_prob=0.9),
    }
    j = decide_relation_verdict(rel, judge_answers, min_prob=0.55)
    assert j.verdict is Verdict.CONTRADICTED
    assert any(f.startswith("staged_order_inverted") for f in j.flags)
    assert j.evidence_sentence_ids == ["P0.S0"]

    judge_answers[f"rel4::{rid}"] = order("no_order")
    j = decide_relation_verdict(rel, judge_answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED
    assert any(f.startswith("staged_order_unstated") for f in j.flags)

    judge_answers[f"rel4::{rid}"] = order("left_first")
    j = decide_relation_verdict(rel, judge_answers, min_prob=0.55)
    assert j.verdict is Verdict.SUPPORTED


async def test_bridge_binding_gate_blocks_third_entity_property():
    """Conflict-189 shape: a clean alias bridge (br2 yes) plus br4 that ignores
    subjects by design — only the binding premise (br5) stops a property stated
    about an unrelated third entity from transferring. br5 low -> flat U stands."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    passages = [
        "Acme is also known as Acme Global in overseas filings.",
        "GammaCo's shares dropped ten percent on Friday.",
    ]
    answer = "Acme Global's shares dropped ten percent on Friday."
    uid = "sent::A.S0"
    script = {
        f"v::{uid}": _flat(uid),
        f"br1::{uid}": Answer(f"br1::{uid}", choice="P0.S0",
                              probabilities={"P0.S0": 0.9, "P1.S0": 0.05, "no_connection": 0.05}),
        f"br3::{uid}": Answer(f"br3::{uid}", choice="P1.S0",
                              probabilities={"P0.S0": 0.05, "P1.S0": 0.9, "no_property": 0.05}),
        # br2 (yes) and br4 (matches) pass via defaults: without the binding
        # gate this composes a false SUPPORTED
        f"br5::{uid}": Answer(f"br5::{uid}", yes_prob=0.05),
    }
    decider = scripted_decider(script)
    response, _ = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), decider, proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.UNSUPPORTED
    asked = [q.question_id for _, questions in decider.seen for q in questions]
    assert any(q.startswith("br5::") for q in asked)


def test_coverage_check_ignores_whole_sentence_inherited_spans():
    """A unit whose anchors miss part of the sentence must trigger the
    whole-sentence fallback even though it inherits the full sentence as
    context (the inherited span used to blind the check for good)."""
    from verifier.rag.source import Span
    from verifier.rag.tree import Node, NodeStatus, SemanticTree
    from verifier.methods.optimized.semantics.select import select_units

    answer = "The migration started in June and ran for seventeen weeks."
    store = TextStore.from_request(answer, ["(src)"])
    tree = SemanticTree(store=store)
    sentence = store.answer_sentences[0]
    partial_end = sentence.span.start + len("The migration started in June")
    unit = tree.add(Node(
        node_id="frag::A.S0::0", origin="answer", sentence_id=sentence.sentence_id,
        kind=NodeKind.FACT,
        anchor_spans=[Span(store.ANSWER_DOC, sentence.span.start, partial_end)],
        inherited_spans=[sentence.span],
        display_text="The migration started in June",
        status=NodeStatus.CHECKED,
    ))
    selection = select_units(store, tree, [unit], set(), [])
    assert [u.node_id for u in selection.units] == [f"sent::{sentence.sentence_id}"]
    assert any("uncovered content" in n for n in selection.notes)


def test_grounding_gate_verbatim_witness():
    """Fix (1): a diagnostic part whose text is verbatim present in a source
    passage must not be capped on a low probe score (presence is a decidable
    string property); absent content still caps (m10 stays capped)."""
    from verifier.methods.optimized.judge.entail import UnitJudgment, apply_grounding_gate
    from verifier.rag.source import Span
    from verifier.rag.tree import Node, NodeStatus

    passages = ["If the cache is full, old entries are evicted to make room."]
    answer = "If the cache is full, old entries are evicted."
    store = TextStore.from_request(answer, passages)
    sentence = store.answer_sentences[0]
    comma = store.answer.index(",")
    unit = Node(
        node_id="sent::A.S0", origin="answer", sentence_id=sentence.sentence_id,
        kind=NodeKind.SENTENCE, anchor_spans=[sentence.span],
        display_text=sentence.text, status=NodeStatus.CHECKED,
    )
    unit.diagnostic_spans = [
        Span(store.ANSWER_DOC, sentence.span.start, comma),
        Span(store.ANSWER_DOC, comma + 2, sentence.span.end - 1),
    ]
    answers = {
        "g::sent::A.S0::0": Answer("g::sent::A.S0::0", yes_prob=0.05),  # false low probe
        "g::sent::A.S0::1": Answer("g::sent::A.S0::1", yes_prob=0.9),
    }
    j = UnitJudgment(node_id=unit.node_id, node_version=1, verdict=Verdict.SUPPORTED,
                     evidence_sentence_ids=["P0.S0"])
    apply_grounding_gate(unit, j, answers, grounding_min_prob=0.5, store=store)
    assert j.verdict is Verdict.SUPPORTED
    assert any(f.startswith("grounding_witness_0") for f in j.flags)

    store2 = TextStore.from_request(answer, ["Entries are evicted when memory runs low."])
    j2 = UnitJudgment(node_id=unit.node_id, node_version=1, verdict=Verdict.SUPPORTED,
                      evidence_sentence_ids=["P0.S0"])
    apply_grounding_gate(unit, j2, answers, grounding_min_prob=0.5, store=store2)
    assert j2.verdict is Verdict.UNSUPPORTED
    assert any(f.startswith("ungrounded_part_0") for f in j2.flags)


def test_unconditional_evidence_does_not_gate_the_antecedent():
    """Gate audit 2026-10-22: 'If the latch is misaligned, ...' / 'If desired, ...'
    were capped on their antecedent although the source states the consequent
    outright. "If A, B" is weaker than B, so A need not be in the source; a source
    that gives the content under a condition of its own still gets the check."""
    from verifier.methods.optimized.judge.entail import UnitJudgment, apply_grounding_gate
    from verifier.rag.source import Span
    from verifier.rag.tree import Node, NodeStatus

    answer = "If desired, old entries are evicted."

    def gate(passage: str) -> UnitJudgment:
        store = TextStore.from_request(answer, [passage])
        sentence = store.answer_sentences[0]
        comma = store.answer.index(",")
        unit = Node(
            node_id="sent::A.S0", origin="answer", sentence_id=sentence.sentence_id,
            kind=NodeKind.SENTENCE, anchor_spans=[sentence.span],
            display_text=sentence.text, status=NodeStatus.CHECKED,
        )
        unit.diagnostic_spans = [
            Span(store.ANSWER_DOC, sentence.span.start, comma),
            Span(store.ANSWER_DOC, comma + 2, sentence.span.end - 1),
        ]
        answers = {
            "g::sent::A.S0::0": Answer("g::sent::A.S0::0", yes_prob=0.03),
            "g::sent::A.S0::1": Answer("g::sent::A.S0::1", yes_prob=0.9),
        }
        j = UnitJudgment(node_id=unit.node_id, node_version=1, verdict=Verdict.SUPPORTED,
                         evidence_sentence_ids=["P0.S0"])
        apply_grounding_gate(unit, j, answers, grounding_min_prob=0.5, store=store,
                             skip_unconditional_antecedent=True)
        return j

    outright = gate("Old entries are evicted to make room.")
    assert outright.verdict is Verdict.SUPPORTED
    assert any(f.startswith("antecedent_unconditional_source_0") for f in outright.flags)
    conditional = gate("Old entries are evicted when memory runs low.")
    assert conditional.verdict is Verdict.UNSUPPORTED


def _diagnostic_unit(answer: str, cuts: list[str]):
    """A kept-whole answer sentence with diagnostic parts split at the given
    connective phrases (each phrase dropped, as _attach_diagnostics does)."""
    from verifier.rag.source import Span
    from verifier.rag.tree import Node, NodeStatus

    store = TextStore.from_request(answer, ["Unrelated source text about something else entirely."])
    sentence = store.answer_sentences[0]
    unit = Node(
        node_id="sent::A.S0", origin="answer", sentence_id=sentence.sentence_id,
        kind=NodeKind.SENTENCE, anchor_spans=[sentence.span],
        display_text=sentence.text, status=NodeStatus.CHECKED,
    )
    bounds, pos = [], 0
    for cut in cuts:
        i = answer.index(cut, pos)
        bounds.append((i, i + len(cut)))
        pos = i + len(cut)
    starts = [0] + [e for _, e in bounds]
    ends = [s for s, _ in bounds] + [len(answer.rstrip("."))]
    unit.diagnostic_spans = [Span(store.ANSWER_DOC, a, b) for a, b in zip(starts, ends)]
    return store, unit


def test_non_assertive_parts_are_not_gated():
    """Gate audit 2026-10-22 (found on QA-900, confirmed on Summary + Data2txt):
    parts that assert nothing a source could contain were capped although no
    true error sat in them. A fabricated alternative is still content (m10)."""
    from verifier.methods.optimized.judge.entail import non_assertive_part

    def reasons(answer, cuts):
        store, unit = _diagnostic_unit(answer, cuts)
        return [non_assertive_part(store, unit, i) for i in range(len(unit.diagnostic_spans))]

    assert reasons("If desired, top with cheese.", [", "]) == ["optional", None]
    assert reasons("9. Check if the door now shuts and locks normally.", [" if "]) == ["whether", "whether"]
    assert reasons("Water might stain them, so be cautious.", [", so "]) == [None, "advice"]
    assert reasons("Adjust the clip up or down a few links.", [" or "]) == [None, "idiom"]
    assert reasons("Note that the passages do not provide specific times or temperatures.", [" or "])[0] == "about the source"
    # a fabricated disjunct stays gated, however short (m10)
    assert reasons("Production requests face a 30-second timeout or instant rejection.", [" or "]) == [None, None]
    # a real condition is not non-assertive: it is left to the antecedent rule (m11)
    assert reasons("If the cache is full, old entries are evicted.", [", "]) == [None, None]


def test_framing_words_are_dropped_before_the_presence_check():
    from verifier.methods.optimized.judge.entail import grounding_question, strip_framing, verbatim_in_source

    assert strip_framing("Therefore, based on the provided passages, you can use netting") == "you can use netting"
    assert strip_framing("However, it does not offer WiFi") == "it does not offer WiFi"
    assert strip_framing("Overall") == "Overall"  # nothing would be left: keep the text
    store = TextStore.from_request("x", ["You can use bird netting over the porch."])
    assert verbatim_in_source(store, "Therefore, you can use bird netting")
    s2, unit = _diagnostic_unit("Therefore, based on the provided passages, you can use netting or spikes.", [" or "])
    q = grounding_question(s2, unit, 0)
    assert 'Content piece: "you can use netting"' in q.instructions


def test_a_reason_clause_is_its_own_diagnostic_part():
    """'Use graph paper if desired, as it can help ...': without the break the
    antecedent part swallowed the reason clause and the whole part was capped."""
    from verifier.methods.optimized.semantics.propose_jev import CutCandidate, _attach_diagnostics
    from verifier.rag.tree import Node, NodeStatus

    answer = "Use graph paper if desired, as it can help in creating a neat graph."
    store = TextStore.from_request(answer, ["Graph paper helps."])
    sentence = store.answer_sentences[0]
    host = Node(
        node_id="sent::A.S0", origin="answer", sentence_id=sentence.sentence_id,
        kind=NodeKind.SENTENCE, anchor_spans=[sentence.span],
        display_text=sentence.text, status=NodeStatus.CHECKED,
    )
    i = answer.index(" if ")
    marker = CutCandidate("c0", sentence.sentence_id, i, i + len(" if "), "if")
    _attach_diagnostics(store, host, [marker], [])
    parts = [store.slice(sp) for sp in host.diagnostic_spans]
    assert parts == ["Use graph paper", "it can help in creating a neat graph."] or \
        parts[-1].startswith("it can help"), parts


async def test_member_rescue_on_staged_matched_relation():
    """Fix (3): a staged-matched CAUSE relation rescues a FLAT member to S with
    the link sentence as evidence (witness-bearing judgment beats the
    witness-free one), so the merge is no longer dragged to U."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    store = TextStore.from_request(STAGED_ANSWER, STAGED_PASSAGES)
    cand = find_candidates(store)[0]
    rid = "rel::A.S0::0"
    script = {
        cand.question_id: choice(cand.question_id, "cause_link"),
        # left member flat; right member supported
        "v::frag::A.S0::0": Answer(
            "v::frag::A.S0::0", choice="not_in_context",
            probabilities={"supported": 0.46, "contradicted": 0.08, "not_in_context": 0.46},
        ),
        "v::frag::A.S0::1": verdict_answer("v::frag::A.S0::1", "supported"),
        "r::frag::A.S0::1::P0.S0": Answer("r::frag::A.S0::1::P0.S0", yes_prob=0.9),
        f"rel1::{rid}": Answer(
            f"rel1::{rid}", choice="P0.S1",
            probabilities={"P0.S0": 0.05, "P0.S1": 0.9, "no_such_link": 0.05},
        ),
        # rel2/rel3 pass via conftest defaults (matches 0.9) -> staged_matched
        # bridge find questions fire for the flat member before the rescue:
        f"br1::frag::A.S0::0": Answer(
            f"br1::frag::A.S0::0", choice="no_connection",
            probabilities={"P0.S0": 0.05, "P0.S1": 0.05, "no_connection": 0.9},
        ),
        # the rescue completes an all-S sentence, which buys the round-4
        # whole-sentence recheck — a deliberate extra guard on K2 upgrades
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "supported"),
        "r::sent::A.S0::P0.S1": Answer("r::sent::A.S0::P0.S1", yes_prob=0.9),
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=STAGED_PASSAGES, generated_answer=STAGED_ANSWER),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    by_text = {c.claim_text: c for c in response.claims}
    rescued = next(c for t, c in by_text.items() if not t.startswith("The answer links")
                   and "expires" in t)
    assert rescued.verdict is Verdict.SUPPORTED
    assert "one side of a connection the source itself states" in rescued.reasoning
    rel = next(c for t, c in by_text.items() if t.startswith("The answer links"))
    assert rel.verdict is Verdict.SUPPORTED, "merge no longer dragged by the flat member"


async def test_opposition_filter_demotes_non_opposite_c():
    """Fix (4): a confident C whose cited sentence talks about a DIFFERENT
    entity is demoted to U (R1: explicit opposite only); a genuine opposite
    (conftest default) keeps its C."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    passages = ["The regulator fined BetaCorp for the data breach."]
    answer = "The regulator fined Acme for the data breach."
    uid = "sent::A.S0"
    base = {
        f"v::{uid}": verdict_answer(f"v::{uid}", "contradicted"),
        f"r::{uid}::P0.S0": Answer(f"r::{uid}::P0.S0", yes_prob=0.9),
    }
    script = {
        **base,
        f"c1::{uid}": Answer(
            f"c1::{uid}", choice="different",
            probabilities={"opposite": 0.05, "different": 0.9, "matches": 0.05},
        ),
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.UNSUPPORTED
    assert "states something different" in response.claims[0].reasoning

    # genuine opposition (default c1 answer: opposite 0.9) -> C stands
    response, _ = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), scripted_decider(dict(base)), proposer="jev"),
    )
    assert response.claims[0].verdict is Verdict.CONTRADICTED


def test_conditional_scope_undoes_interior_cuts():
    """audit09: a kept-whole conditional (lead 'if') must not let an interior
    'because' cut lift conditional content into unconditional claims — every
    accepted cut in the sentence is undone, the sentence stays whole and the
    undone markers join the diagnostic parts."""
    answer = "If the watchdog fires, the service restarts because memory is exhausted."
    store = TextStore.from_request(answer, ["(src)"])
    cands = find_candidates(store)
    by_conn = {c.connective: c for c in cands}
    answers = {
        by_conn["if"].question_id: choice(by_conn["if"].question_id, "keep_whole"),
        by_conn["because"].question_id: choice(by_conn["because"].question_id, "cause_link"),
    }
    outcome = build_tree_from_cuts(store, cands, answers, min_prob=0.6)
    assert outcome.units == []  # select falls back to the whole sentence
    assert any("kept-whole conditional" in n for n in outcome.notes)
    sent = outcome.tree.nodes["sent::A.S0"]
    assert len(sent.diagnostic_spans) >= 2  # parts still get grounding checks

    # control: a kept-whole 'or' elsewhere does NOT undo an accepted cut
    answer2 = "The backup failed because the disk was full."
    store2 = TextStore.from_request(answer2, ["(src)"])
    cands2 = find_candidates(store2)
    answers2 = {cands2[0].question_id: choice(cands2[0].question_id, "cause_link")}
    outcome2 = build_tree_from_cuts(store2, cands2, answers2, min_prob=0.6)
    assert len([u for u in outcome2.units if u.kind is NodeKind.FACT]) == 2


def test_staged_link_assertedness_gate():
    """P-assert (second self-audit n02): a link the source states only inside a
    condition or reported speech composes neither S nor C — the relation stays
    UNSUPPORTED with the unasserted flag. An asserted link (default) passes."""
    from verifier.methods.optimized.judge.entail import decide_relation_verdict

    store = TextStore.from_request(STAGED_ANSWER, STAGED_PASSAGES)
    rel = _staged_relation(store)
    rid = rel.node_id
    answers = {
        f"rel1::{rid}": Answer(f"rel1::{rid}", choice="P0.S0",
                               probabilities={"P0.S0": 0.9, "no_such_link": 0.1}),
        f"rel2::{rid}": side(f"rel2::{rid}", "matches"),
        f"rel3::{rid}": side(f"rel3::{rid}", "matches"),
        f"rel5::{rid}": Answer(f"rel5::{rid}", yes_prob=0.08),  # inside if/reported
    }
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED
    assert any(f.startswith("staged_link_unasserted") for f in j.flags)

    # even an 'opposite' side must not compose C from an unasserted link
    answers[f"rel3::{rid}"] = side(f"rel3::{rid}", "opposite")
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED

    answers[f"rel3::{rid}"] = side(f"rel3::{rid}", "matches")
    answers[f"rel5::{rid}"] = Answer(f"rel5::{rid}", yes_prob=0.9)
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.SUPPORTED


def test_witness_matches_within_single_sentence_only():
    """Phantom match (second self-audit n04, deterministic): normalization must
    not erase sentence boundaries — cross-sentence assembled content gets no
    witness, real single-sentence presence still does."""
    from verifier.methods.optimized.judge.entail import verbatim_in_source

    store = TextStore.from_request(
        "x", ["The system alerts on-call staff when errors occur. "
              "The server restarts. Daily backups run at noon."])
    assert not verbatim_in_source(store, "the server restarts daily")
    assert verbatim_in_source(store, "the server restarts")
    assert verbatim_in_source(store, "Daily backups run at noon")
    assert not verbatim_in_source(store, "the server restarts weekly")


def test_inverted_conditional_triggers_scope_guard():
    """Lexicon gap (second self-audit n05): 'Should the watchdog fire, ...'
    now yields a lead candidate; keep_whole on it undoes the interior
    'because' cut like any conditional."""
    answer = "Should the watchdog fire, the service restarts because memory is exhausted."
    store = TextStore.from_request(answer, ["(src)"])
    cands = find_candidates(store)
    by_conn = {c.connective: c for c in cands}
    assert "should" in by_conn and by_conn["should"].lead_start is not None
    answers = {
        by_conn["should"].question_id: choice(by_conn["should"].question_id, "keep_whole"),
        by_conn["because"].question_id: choice(by_conn["because"].question_id, "cause_link"),
    }
    outcome = build_tree_from_cuts(store, cands, answers, min_prob=0.6)
    assert outcome.units == []
    assert any("kept-whole conditional" in n for n in outcome.notes)


def test_witness_preserves_signs_and_comparators():
    """Round-2 audit B05: the witness normalization keeps a digit-attached sign
    and comparison operators, while a hyphen between letters stays punctuation."""
    from verifier.methods.optimized.judge.entail import verbatim_in_source

    s1 = TextStore.from_request("x", ["The single recorded temperature for sample T7 is exactly -5 C."])
    assert verbatim_in_source(s1, "is exactly -5 C")
    assert not verbatim_in_source(s1, "is exactly +5 C")
    s2 = TextStore.from_request("x", ["The only configured acceptance predicate is x <= 10."])
    assert verbatim_in_source(s2, "predicate is x <= 10")
    assert not verbatim_in_source(s2, "predicate is x >= 10")
    s3 = TextStore.from_request("x", ["The system alerts on-call staff."])
    assert verbatim_in_source(s3, "alerts on-call staff")
    assert verbatim_in_source(s3, "alerts on call staff")


def test_witness_ignores_a_leading_list_marker():
    """Gate sample 2026-10-22 (#20): "8. Click Apply" was capped although the
    source says "Click Apply" verbatim; the step number is layout, not content.
    Only a real marker is dropped: a sign or a decimal stays."""
    from verifier.methods.optimized.judge.entail import verbatim_in_source

    store = TextStore.from_request(
        "x", ["Click Apply at the bottom of the screen, you can finish syncing process."])
    assert verbatim_in_source(store, '8. Click "Apply"')
    assert verbatim_in_source(store, "* Click Apply")
    assert not verbatim_in_source(store, "8. Click Cancel")
    s1 = TextStore.from_request("x", ["The recorded temperature is exactly 5 C."])
    assert not verbatim_in_source(s1, "-5 C")
    s2 = TextStore.from_request("x", ["Add 5 cups of flour."])
    assert not verbatim_in_source(s2, "2.5 cups of flour")


def test_staged_discharge_and_aux_evidence():
    """Round-2 audit B03/B07: a condition-wrapped link composes S when another
    sentence discharges the condition (both cited); an alias premise named by
    rel7 joins the evidence; without a discharge the gate still fails closed."""
    from verifier.methods.optimized.judge.entail import decide_relation_verdict

    store = TextStore.from_request(STAGED_ANSWER, STAGED_PASSAGES)
    rel = _staged_relation(store)
    rid = rel.node_id

    def pick(qid, label, options):
        probs = dict.fromkeys(options, (1 - 0.9) / (len(options) - 1))
        probs[label] = 0.9
        return Answer(qid, choice=label, probabilities=probs)

    answers = {
        f"rel1::{rid}": pick(f"rel1::{rid}", "P0.S0", ("P0.S0", "P0.S1", "no_such_link")),
        f"rel2::{rid}": side(f"rel2::{rid}", "matches"),
        f"rel3::{rid}": side(f"rel3::{rid}", "matches"),
        f"rel5::{rid}": Answer(f"rel5::{rid}", yes_prob=0.1),  # condition-wrapped
        f"rel6::{rid}": pick(f"rel6::{rid}", "P0.S1", ("P0.S1", "no_discharge")),
    }
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.SUPPORTED
    assert j.evidence_sentence_ids == ["P0.S0", "P0.S1"]
    assert any(f.startswith("staged_link_discharged") for f in j.flags)

    answers[f"rel6::{rid}"] = pick(f"rel6::{rid}", "no_discharge", ("P0.S1", "no_discharge"))
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.UNSUPPORTED
    assert any(f.startswith("staged_link_unasserted") for f in j.flags)

    answers[f"rel5::{rid}"] = Answer(f"rel5::{rid}", yes_prob=0.9)
    answers[f"rel7::{rid}"] = pick(f"rel7::{rid}", "P0.S1", ("P0.S1", "none_needed"))
    j = decide_relation_verdict(rel, answers, min_prob=0.55)
    assert j.verdict is Verdict.SUPPORTED
    assert j.evidence_sentence_ids == ["P0.S0", "P0.S1"]
    assert any(f.startswith("aux_premise") for f in j.flags)


async def test_bridge_rescue_regated_for_grounding():
    """Round-2 audit B04: a flat unit with an ungrounded diagnostic part that
    bridge rescue upgrades to S must pass the grounding gate again — the
    rescued S is capped back to U, with both flags kept."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    passages = [
        "All production traffic is routed exclusively to the Helios cluster.",
        "The Helios cluster enforces a 30-second request timeout.",
    ]
    answer = "Production requests face a 30-second timeout or instant rejection."
    store = TextStore.from_request(answer, passages)
    cand = [c for c in find_candidates(store) if c.connective == "or"][0]
    uid = "sent::A.S0"
    script = {
        cand.question_id: choice(cand.question_id, "keep_whole"),
        f"v::{uid}": _flat(uid),
        f"g::{uid}::0": Answer(f"g::{uid}::0", yes_prob=0.9),
        f"g::{uid}::1": Answer(f"g::{uid}::1", yes_prob=0.05),  # 'instant rejection' absent
        f"br1::{uid}": Answer(f"br1::{uid}", choice="P0.S0",
                              probabilities={"P0.S0": 0.9, "P1.S0": 0.05, "no_connection": 0.05}),
        f"br3::{uid}": Answer(f"br3::{uid}", choice="P1.S0",
                              probabilities={"P0.S0": 0.05, "P1.S0": 0.9, "no_property": 0.05}),
        # br2/br4/br5 pass via defaults: the rescue composes S, the gate caps it
    }
    response, _ = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    claim = response.claims[0]
    assert claim.verdict is Verdict.UNSUPPORTED


async def test_meta_discourse_sentence_is_excluded():
    """nc gate: a pure greeting/offer sentence is dropped from the claims at
    the high bar (0.80); below the bar, or a factual sentence, stays. A false
    exclusion hides content (danger direction), hence the bar."""
    from test_pipeline import make_deps
    from verifier.methods.optimized.pipeline import run_verification

    answer = "I hope this helps with your setup! The cache stores query results."
    passages = ["The cache stores query results."]

    def nc(sid, meta_p):
        return Answer(f"nc::{sid}", choice="meta_discourse" if meta_p >= 0.5 else "factual_claim",
                      probabilities={"factual_claim": 1 - meta_p, "meta_discourse": meta_p})

    script = {
        "nc::A.S0": nc("A.S0", 0.92),  # greeting: excluded
        "v::sent::A.S1": verdict_answer("v::sent::A.S1", "supported"),
        "r::sent::A.S1::P0.S0": Answer("r::sent::A.S1::P0.S0", yes_prob=0.9),
    }
    response, trace = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), scripted_decider(script), proposer="jev"),
    )
    assert len(response.claims) == 1
    assert response.claims[0].claim_text == "The cache stores query results."

    # below the bar: both sentences stay
    script2 = dict(script)
    script2["nc::A.S0"] = nc("A.S0", 0.6)
    script2["v::sent::A.S0"] = verdict_answer("v::sent::A.S0", "not_in_context")
    response, _ = await run_verification(
        VerifyRequest(context_passages=passages, generated_answer=answer),
        make_deps(FakeLLM(), scripted_decider(script2), proposer="jev"),
    )
    assert len(response.claims) == 2


async def test_llm_escalation_requires_verbatim_witness():
    """Flat-U escalation: a second-judge 'supported' upgrades only with a
    verbatim quote verified inside the cited sentence; a fabricated quote or a
    declined answer leaves the flat U untouched."""
    from verifier.methods.optimized.judge.entail import UnitJudgment
    from verifier.methods.optimized.judge.escalate import EscalationAnswer, escalate_flat
    from verifier.rag.tree import Node, NodeKind, NodeStatus

    passages = ["The exporter writes metrics to a local file every minute."]
    store = TextStore.from_request("The exporter writes metrics every minute.", passages)
    sentence = store.answer_sentences[0]
    unit = Node(node_id="sent::A.S0", origin="answer", sentence_id=sentence.sentence_id,
                kind=NodeKind.SENTENCE, anchor_spans=[sentence.span],
                display_text=sentence.text, status=NodeStatus.CHECKED)

    class FakeEscalationLLM:
        def __init__(self, answer):
            self.answer = answer

        async def generate_structured(self, messages, schema):
            assert schema is EscalationAnswer
            return self.answer

    def flat_judgment():
        return UnitJudgment(node_id=unit.node_id, node_version=1,
                            verdict=Verdict.UNSUPPORTED, flags=["low_confidence(p=0.5)"])

    j = flat_judgment()
    ok = await escalate_flat(
        FakeEscalationLLM(EscalationAnswer(verdict="supported", sentence_id="P0.S0",
                                           quote="writes metrics to a local file")),
        store, unit, j)
    assert ok and j.verdict is Verdict.SUPPORTED and j.evidence_sentence_ids == ["P0.S0"]
    assert any(f.startswith("llm_rescue(") for f in j.flags)

    j = flat_judgment()
    ok = await escalate_flat(
        FakeEscalationLLM(EscalationAnswer(verdict="supported", sentence_id="P0.S0",
                                           quote="writes metrics to the cloud nightly")),
        store, unit, j)
    assert not ok and j.verdict is Verdict.UNSUPPORTED
    assert any(f.startswith("llm_rescue_no_witness") for f in j.flags)

    j = flat_judgment()
    ok = await escalate_flat(
        FakeEscalationLLM(EscalationAnswer(verdict="not_established")), store, unit, j)
    assert not ok and j.verdict is Verdict.UNSUPPORTED


def test_lexicon_conditional_words_can_generate_candidates():
    """Subtraction audit R4 obligation: a word with the 'conditional' role must
    also carry a candidate-generating role (cut/lead/inversion) — otherwise
    adding it to the scope guard is a silent no-op (the n05 failure mode)."""
    from verifier.methods.optimized.semantics.lexicon import _LEXICON

    for word, roles in _LEXICON.items():
        if "conditional" in roles:
            assert roles & {"cut", "lead", "inversion"}, (
                f"'{word}' is conditional but no candidate generator can see it"
            )


def test_witness_compares_words_of_any_script():
    """sa3: the witness compares every word of the part in any script. A part in
    another script got no token at all and matched any source sentence that had
    none either (a phantom witness); a mixed part must match as a whole; a part
    with no word is no witness."""
    from verifier.methods.optimized.judge.entail import verbatim_in_source

    store = TextStore.from_request(
        "x",
        [
            "北京是中国的首都。",
            "Москва — столица России.",
            "東京は日本の首都です。",
            "The 2008 Olympics drew record crowds.",
            "* * *",
        ],
    )
    assert verbatim_in_source(store, "北京是中国的首都")
    assert verbatim_in_source(store, "北京")
    assert not verbatim_in_source(store, "上海是中国的首都")
    assert verbatim_in_source(store, "Москва — столица России")
    assert not verbatim_in_source(store, "Москва — столица Франции")
    assert verbatim_in_source(store, "Москва")
    assert not verbatim_in_source(store, "Париж")  # no token once: it matched the "* * *" line
    assert verbatim_in_source(store, "日本の首都")
    assert not verbatim_in_source(store, "大阪は日本の首都です")
    # mixed: both halves have to sit in one sentence
    assert not verbatim_in_source(store, "北京 2008")
    assert verbatim_in_source(
        TextStore.from_request("x", ["北京2008年举办了奥运会。"]), "北京 2008"
    )
    # nothing to compare is no witness, even next to a source line with no word
    assert not verbatim_in_source(store, "——")
    assert not verbatim_in_source(store, "")
