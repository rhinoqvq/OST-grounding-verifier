"""minimal_jev: the whole answer against the whole source in one request, one claim, the chosen
passage as its citation; a request that does not fit one call gives no verdict."""

import pytest

from conftest import ANSWER, PASSAGE_0, PASSAGE_1
from test_minimal_jev import ForbiddenLLM
from verifier.harness.config import Settings
from verifier.harness.errors import OutOfSingleCallRange
from verifier.harness.pipeline import Dependencies, run_request
from verifier.methods.minimal_jev.single import MAX_PASSAGES, VERDICTS
from verifier.models.decider import Answer, FakeDecider
from verifier.output.schemas import Verdict, VerifyRequest


def make_deps(decider) -> Dependencies:
    settings = Settings(_env_file=None, method="minimal_jev", typesafe_api_key="t")
    return Dependencies(settings=settings, llm=ForbiddenLLM(), decider=decider, conn=None)


def calls(verdict: str, evidence: str, p: float = 0.8) -> FakeDecider:
    return FakeDecider({
        "verdict": Answer("verdict", choice=verdict, probabilities={verdict: p}, confidence=p),
        "evidence": Answer("evidence", choice=evidence, probabilities={evidence: 0.9}, confidence=0.9),
    })


def wal_request() -> VerifyRequest:
    return VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)


async def test_one_request_one_claim_citing_the_whole_chosen_passage():
    decider = calls("CONTRADICTED", "2")
    response, _ = await run_request(wal_request(), make_deps(decider))
    assert decider.usage.calls == 1
    state, questions = decider.seen[0]
    assert PASSAGE_0 in state and PASSAGE_1 in state and ANSWER in state  # verbatim, unsplit
    assert [q.question_id for q in questions] == ["verdict", "evidence"]
    assert questions[0].options == VERDICTS
    assert "Fully entailed by one or more context passages." in questions[0].options.values()
    assert set(questions[1].options) == {"1", "2", "none"}
    [claim] = response.claims
    assert claim.claim_text == ANSWER.strip() and claim.verdict is Verdict.CONTRADICTED
    [cit] = claim.citations
    assert cit.index == 1 and cit.quote == PASSAGE_1
    assert PASSAGE_1[cit.start_char:cit.end_char] == cit.quote
    assert "p=0.80" in claim.reasoning and "passage 2" in claim.reasoning


async def test_top_choice_without_a_threshold():
    response, _ = await run_request(wal_request(), make_deps(calls("SUPPORTED", "1", p=0.34)))
    assert response.claims[0].verdict is Verdict.SUPPORTED
    assert response.claims[0].citations[0].quote == PASSAGE_0


async def test_supported_without_evidence_falls_to_the_contract():
    response, _ = await run_request(wal_request(), make_deps(calls("SUPPORTED", "none")))
    assert response.claims[0].verdict is Verdict.UNSUPPORTED and response.claims[0].citations == []


async def test_unsupported_carries_no_citation():
    response, _ = await run_request(wal_request(), make_deps(calls("UNSUPPORTED", "1")))
    assert response.claims[0].verdict is Verdict.UNSUPPORTED and response.claims[0].citations == []


async def test_more_passages_than_one_question_lists_is_out_of_range():
    decider = calls("SUPPORTED", "1")
    request = VerifyRequest(context_passages=[f"Passage number {i}." for i in range(MAX_PASSAGES + 1)],
                            generated_answer="Passage number 3 exists.")
    with pytest.raises(OutOfSingleCallRange) as err:
        await run_request(request, make_deps(decider))
    assert err.value.status == "out_of_single_call_range" and "passages" in str(err.value)
    assert decider.seen == []


async def test_a_request_over_the_budget_is_out_of_range():
    decider = calls("SUPPORTED", "1")
    passage = "The committee reviewed the annual budget report in great detail. " * 3_000
    with pytest.raises(OutOfSingleCallRange) as err:
        await run_request(VerifyRequest(context_passages=[passage], generated_answer="It did."), make_deps(decider))
    assert "tokens" in str(err.value) and decider.seen == []


async def test_empty_answer_and_empty_context_need_no_request():
    decider = FakeDecider({})
    response, _ = await run_request(VerifyRequest(context_passages=[PASSAGE_0], generated_answer="  "),
                                    make_deps(decider))
    assert response.claims == []
    response, _ = await run_request(VerifyRequest(context_passages=[], generated_answer=ANSWER), make_deps(decider))
    assert [c.verdict for c in response.claims] == [Verdict.UNSUPPORTED] and decider.usage.calls == 0
