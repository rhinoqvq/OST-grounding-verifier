"""Method 1 (lexical baseline): zero model calls, mechanics fully deterministic."""

import pytest
from fastapi.testclient import TestClient

from conftest import ANSWER, PASSAGE_0, PASSAGE_1
from verifier.harness.baseline import run_baseline
from verifier.harness.config import Settings
from verifier.harness.pipeline import Dependencies, run_request
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.service import create_app

PASSAGE = "The retention period is 30 days. Automatic backups are enabled by default."


class ForbiddenLLM:
    async def generate_structured(self, messages, schema):  # pragma: no cover
        raise AssertionError("baseline must not call the LLM")


class ForbiddenDecider:
    name = "forbidden"

    async def ask(self, state, questions):  # pragma: no cover
        raise AssertionError("baseline must not call the decider")


def make_deps(**overrides) -> Dependencies:
    settings = Settings(_env_file=None, method="baseline", **overrides)
    return Dependencies(settings=settings, llm=ForbiddenLLM(), decider=ForbiddenDecider(), conn=None)


async def verify(answer: str, passages: list[str], **overrides):
    request = VerifyRequest(context_passages=passages, generated_answer=answer)
    response, trace = await run_baseline(request, make_deps(**overrides))
    return response


async def test_supported_with_verbatim_sentence_citation():
    response = await verify("The retention period is 30 days.", [PASSAGE])
    claim = response.claims[0]
    assert claim.verdict is Verdict.SUPPORTED
    cite = claim.citations[0]
    assert cite.index == 0
    assert PASSAGE[cite.start_char : cite.end_char] == cite.quote
    assert "retention period is 30 days" in cite.quote


async def test_number_mismatch_is_contradicted():
    response = await verify("The retention period is 31 days.", [PASSAGE])
    claim = response.claims[0]
    assert claim.verdict is Verdict.CONTRADICTED
    assert claim.citations and "30 days" in claim.citations[0].quote
    assert "numbers disagree" in claim.reasoning


async def test_negation_mismatch_is_contradicted():
    response = await verify("Automatic backups are not enabled by default.", [PASSAGE])
    claim = response.claims[0]
    assert claim.verdict is Verdict.CONTRADICTED
    assert "negated" in claim.reasoning


async def test_low_coverage_is_unsupported_without_citations():
    response = await verify("The service encrypts data with AES-256 at rest.", [PASSAGE])
    claim = response.claims[0]
    assert claim.verdict is Verdict.UNSUPPORTED
    assert claim.citations == []


async def test_empty_context_is_all_unsupported():
    response = await verify("The retention period is 30 days.", [])
    assert [c.verdict for c in response.claims] == [Verdict.UNSUPPORTED]


async def test_empty_answer_short_circuits():
    response = await verify("   ", [PASSAGE])
    assert response.claims == []


async def test_deterministic():
    a = await verify(ANSWER, [PASSAGE_0, PASSAGE_1])
    b = await verify(ANSWER, [PASSAGE_0, PASSAGE_1])
    assert [c.model_dump() for c in a.claims] == [c.model_dump() for c in b.claims]


async def test_run_request_routes_baseline():
    request = VerifyRequest(context_passages=[PASSAGE], generated_answer="The retention period is 30 days.")
    response, _ = await run_request(request, make_deps())
    assert response.claims[0].verdict is Verdict.SUPPORTED


def test_service_with_baseline_method():
    settings = Settings(_env_file=None, method="baseline")
    deps = Dependencies(settings=settings, llm=ForbiddenLLM(), decider=ForbiddenDecider(), conn=None)
    app = create_app(settings, dependencies=deps)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/verify",
            json={"context_passages": [PASSAGE_0, PASSAGE_1], "generated_answer": ANSWER},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["claims"]) == 3  # one claim per answer sentence
    for c in body["claims"]:
        assert c["verdict"] in {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
        if c["verdict"] == "UNSUPPORTED":
            assert c["citations"] == []
        else:
            assert c["citations"]
