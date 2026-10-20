"""Jev-only proposer: candidate generation, tree building, pipeline e2e."""

import pytest

from conftest import FakeLLM, scripted_decider, verdict_answer
from verifier.models.decider import Answer
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.rag.source import TextStore
from verifier.rag.tree import NodeKind
from verifier.semantics.propose_jev import (
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
    from verifier.harness.pipeline import run_verification

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
    from verifier.harness.pipeline import run_verification

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
    from verifier.harness.pipeline import run_verification

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
    from verifier.harness.pipeline import run_verification

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
    from verifier.judge.entail import decide_relation_verdict

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
    from verifier.harness.pipeline import run_verification

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
    from verifier.harness.pipeline import run_verification

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


def test_embed_prior_rescues_linking_premise(wal_store):
    """A linking premise with probe 0.4 (< 0.6 bar) joins the evidence when the
    embedding prior ranks it top-2 and it clears the 0.25 floor."""
    from verifier.judge.entail import decide_verdict
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
    from verifier.harness.pipeline import run_verification

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
    from verifier.harness.pipeline import run_verification
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
