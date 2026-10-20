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
