"""Jev-only proposer: candidate generation, tree building, pipeline e2e."""

import pytest

from conftest import FakeLLM, scripted_decider, verdict_answer
from verifier.models.decider import Answer
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.rag.source import TextStore
from verifier.rag.tree import NodeKind
from verifier.semantics.propose_jev import build_tree_from_cuts, find_candidates

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
