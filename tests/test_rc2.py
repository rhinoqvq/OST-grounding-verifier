"""rc2 candidate switches: the sentence guard over every alarm (guard_scope="all"), the
parent recheck off, and two-stage judging (the whole-sentence call first; a sentence it
supports with evidence settles all of its units, which inherit its citation)."""

import pytest

from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM, scripted_decider, verdict_answer
from test_sentence_guard import judged, node
from verifier.harness.config import Settings
from verifier.harness.pipeline import Dependencies
from verifier.methods.minimal_jev.verifier import SentenceCall
from verifier.methods.optimized.judge.guard import apply_sentence_guard, guarded_layer
from verifier.methods.optimized.pipeline import run_verification
from verifier.models.decider import Answer
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.rag.source import TextStore
from verifier.rag.tree import NodeKind

S, C, U = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNSUPPORTED


@pytest.fixture
def wal_request() -> VerifyRequest:
    return VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)


def deps_with(decider, llm, **overrides) -> Dependencies:
    settings = Settings(_env_file=None, openai_api_key="test", typesafe_api_key="test",
                        **{"proposer": "llm", **overrides})
    return Dependencies(settings=settings, llm=llm, decider=decider, conn=None)


def asked(decider) -> list[str]:
    return [q.question_id for _, qs in decider.seen for q in qs]


def sentence_call(k: int, verdict: str, evidence: str | None) -> dict[str, Answer]:
    probs = dict.fromkeys(["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"], 0.05)
    probs[verdict] = 0.9
    pick = evidence or "none"
    return {f"v{k}": Answer(f"v{k}", choice=verdict, probabilities=probs, confidence=0.9),
            f"e{k}": Answer(f"e{k}", choice=pick, probabilities={pick: 0.9}, confidence=0.9)}


def contract_violations(request, response) -> list[str]:
    """An S or C claim needs a citation; a citation is a verbatim slice of its passage."""
    bad = []
    for claim in response.claims:
        if claim.verdict is not Verdict.UNSUPPORTED and not claim.citations:
            bad.append(f"no citation: {claim.claim_text}")
        for cit in claim.citations:
            if request.context_passages[cit.index][cit.start_char:cit.end_char] != cit.quote:
                bad.append(f"not verbatim: {cit.quote}")
    return bad


def test_scope_all_makes_the_whole_sentence_call_a_layer():
    sent = node("s", NodeKind.SENTENCE)
    assert guarded_layer(sent, judged(sent, U)) is None
    assert guarded_layer(sent, judged(sent, U), "all") == "whole"
    assert guarded_layer(sent, judged(sent, C, "parent_override"), "all") == "parent_override"
    # a structural layer still names itself first
    assert guarded_layer(sent, judged(sent, U, "ungrounded_part_0(p=0.06)"), "all") == "gate"


def test_scope_all_releases_a_plain_whole_sentence_alarm():
    plain, over = node("s", NodeKind.SENTENCE), node("p", NodeKind.SENTENCE, "A.S1")
    calls = {"A.S0": SentenceCall(S, "P0.S1", 0.9), "A.S1": SentenceCall(S, "P0.S4", 0.8)}
    judgments = {"s": judged(plain, U), "p": judged(over, C, "parent_override")}
    assert apply_sentence_guard([plain, over], judgments, calls) == []  # structural scope
    assert apply_sentence_guard([plain, over], judgments, calls, scope="all") == ["s", "p"]
    assert "sentence_guard(whole)" in judgments["s"].flags
    assert "sentence_guard(parent_override)" in judgments["p"].flags
    assert judgments["p"].evidence_sentence_ids == ["P0.S4"]
    # no evidence, or not S: the alarm stands even with scope all
    lone = node("x", NodeKind.SENTENCE, "A.S2")
    judgments["x"] = judged(lone, U)
    assert apply_sentence_guard([lone], judgments, {"A.S2": SentenceCall(S, None, 0.7)}, scope="all") == []


async def test_parent_recheck_off_asks_no_recheck(wal_request, wal_proposal, wal_decider):
    """u2 and u3 split sentence A.S1 and are both S: the recheck would ask about the
    whole sentence; with the switch off it does not."""
    response, trace = await run_verification(
        wal_request, deps_with(wal_decider, FakeLLM(wal_proposal), sentence_guard=False, parent_recheck=False))
    assert "v::sent::A.S1" not in asked(wal_decider)
    assert "4-parent-recheck" not in [r["name"] for r in trace.as_dict()["rounds"]]
    assert len(response.claims) == 4


async def test_parent_recheck_on_is_the_default(wal_request, wal_proposal, wal_decider):
    await run_verification(wal_request, deps_with(wal_decider, FakeLLM(wal_proposal), sentence_guard=False))
    assert "v::sent::A.S1" in asked(wal_decider)


async def test_two_stage_settles_supported_sentences_and_judges_the_rest(wal_request, wal_proposal):
    """A.S0 is S with evidence in the whole-sentence call: its unit u1 is S citing P0.S2 and
    gets no OST question. A.S1 is U and A.S2 is S without evidence: their units go the full
    OST way. The 4b guard reuses the stage-1 call and releases nothing."""
    script = {**sentence_call(0, "SUPPORTED", "P0.S2"), **sentence_call(1, "UNSUPPORTED", None),
              **sentence_call(2, "SUPPORTED", None),
              "v::u2": verdict_answer("v::u2", "supported"), "r::u2::P1.S0": Answer("r::u2::P1.S0", yes_prob=0.93),
              "v::u3": verdict_answer("v::u3", "supported"), "r::u3::P1.S1": Answer("r::u3::P1.S1", yes_prob=0.94)}
    decider = scripted_decider(script)
    response, trace = await run_verification(
        wal_request, deps_with(decider, FakeLLM(wal_proposal), two_stage=True, guard_scope="all", parent_recheck=False))
    qs = asked(decider)
    # u1 settled by stage 1: no judge-side question (the answer-only decomposition check still runs)
    assert not [q for q in qs if "::u1" in q and not q.startswith("fid::")]
    assert any("::u4" in q for q in qs)                 # A.S2 (S without evidence) is judged by OST
    assert "v::u2" in qs and "v::u3" in qs
    assert qs.count("v0") == 1                          # one whole-sentence call, reused by 4b
    rounds = {r["name"]: r for r in trace.as_dict()["rounds"]}
    assert rounds["2.8-stage1-sentence-call"]["detail"]["settled_units"] == 1
    assert rounds["4b-sentence-guard"]["detail"]["released"] == []
    assert "4-parent-recheck" not in rounds
    u1 = next(c for c in response.claims if c.claim_text.startswith("SQLite WAL mode allows"))
    p0s2 = TextStore.from_request(wal_request.generated_answer, wal_request.context_passages).sentence_by_id("P0.S2")
    assert u1.verdict is S and [c.quote for c in u1.citations] == [p0s2.text]
    assert contract_violations(wal_request, response) == []


async def test_two_stage_off_keeps_rc1_behaviour(wal_request, wal_proposal, wal_decider):
    response, trace = await run_verification(
        wal_request, deps_with(wal_decider, FakeLLM(wal_proposal), sentence_guard=False))
    assert "2.8-stage1-sentence-call" not in [r["name"] for r in trace.as_dict()["rounds"]]
    assert contract_violations(wal_request, response) == []
