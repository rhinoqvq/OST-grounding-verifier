"""Method 2 (minimal LLM): one call, same harness, contract does the policing."""

import pytest
from fastapi.testclient import TestClient

from conftest import ANSWER, PASSAGE_0, PASSAGE_1
from verifier.harness.config import Settings
from verifier.methods.minimal_llm.verifier import (
    MinimalCitation,
    MinimalClaim,
    MinimalOutput,
    run_minimal_llm,
)
from verifier.harness.pipeline import Dependencies, run_request
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.service import create_app


class FakeMinimalLLM:
    """Scripted stand-in returning a fixed MinimalOutput."""

    def __init__(self, output: MinimalOutput | None = None, error: Exception | None = None):
        self.output = output
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    async def generate_structured(self, messages, schema):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        assert schema is MinimalOutput
        return self.output


class ForbiddenDecider:
    """The minimal method must never consult the judge."""

    name = "forbidden"

    async def ask(self, state, questions):  # pragma: no cover - failing is the point
        raise AssertionError("minimal_llm must not call the decider")


def make_deps(llm, **overrides) -> Dependencies:
    settings = Settings(_env_file=None, method="minimal_llm", openai_api_key="t", **overrides)
    return Dependencies(settings=settings, llm=llm, decider=ForbiddenDecider(), conn=None)


def wal_output() -> MinimalOutput:
    return MinimalOutput(
        claims=[
            MinimalClaim(
                claim_text="WAL mode works over NFS network shares.",
                verdict="CONTRADICTED",
                reasoning="The passage says WAL does not work over a network filesystem.",
                citations=[
                    MinimalCitation(
                        passage_index=0,
                        quote="WAL does not work over a network filesystem",
                    )
                ],
            ),
            MinimalClaim(
                claim_text="WAL uses two auxiliary files (-wal and -shm).",
                verdict="SUPPORTED",
                reasoning="Stated in passage 1.",
                citations=[
                    MinimalCitation(
                        passage_index=1,
                        quote="the write-ahead log file 'X-wal', and the wal-index shared-memory file 'X-shm'",
                    )
                ],
            ),
            MinimalClaim(
                claim_text="WAL files are encrypted with AES-256 at rest.",
                verdict="UNSUPPORTED",
                reasoning="No passage mentions encryption.",
                citations=[],
            ),
        ]
    )


def wal_request() -> VerifyRequest:
    return VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)


async def test_minimal_happy_path_single_call():
    llm = FakeMinimalLLM(output=wal_output())
    response, trace = await run_minimal_llm(wal_request(), make_deps(llm))

    assert len(llm.calls) == 1
    assert [c.verdict for c in response.claims] == [
        Verdict.CONTRADICTED,
        Verdict.SUPPORTED,
        Verdict.UNSUPPORTED,
    ]
    # offsets were re-derived by the contract, so the quote is anchored
    nfs = response.claims[0].citations[0]
    assert nfs.index == 0
    assert PASSAGE_0[nfs.start_char : nfs.end_char] == nfs.quote
    # the prompt carries numbered passages and the answer, nothing else
    user_message = llm.calls[0][1]["content"]
    assert "[0] " in user_message and "[1] " in user_message and ANSWER in user_message


async def test_fabricated_quote_is_downgraded():
    output = MinimalOutput(
        claims=[
            MinimalClaim(
                claim_text="WAL checkpoints every 1000 pages.",
                verdict="SUPPORTED",
                reasoning="Stated in the passages.",
                citations=[
                    MinimalCitation(passage_index=1, quote="checkpoints occur every 1000 pages exactly")
                ],
            )
        ]
    )
    response, trace = await run_minimal_llm(wal_request(), make_deps(FakeMinimalLLM(output=output)))

    claim = response.claims[0]
    assert claim.verdict is Verdict.UNSUPPORTED
    assert claim.citations == []
    assert "downgraded to UNSUPPORTED" in claim.reasoning


async def test_unsupported_with_citations_is_cleared():
    output = MinimalOutput(
        claims=[
            MinimalClaim(
                claim_text="WAL files are encrypted at rest.",
                verdict="UNSUPPORTED",
                reasoning="Not in the passages.",
                citations=[MinimalCitation(passage_index=0, quote="WAL")],
            )
        ]
    )
    response, _ = await run_minimal_llm(wal_request(), make_deps(FakeMinimalLLM(output=output)))
    assert response.claims[0].verdict is Verdict.UNSUPPORTED
    assert response.claims[0].citations == []


async def test_out_of_bounds_index_and_empty_quote_are_dropped():
    output = MinimalOutput(
        claims=[
            MinimalClaim(
                claim_text="Readers do not block writers.",
                verdict="SUPPORTED",
                reasoning="Stated in passage 0.",
                citations=[
                    MinimalCitation(passage_index=7, quote="readers do not block writers"),
                    MinimalCitation(passage_index=0, quote=""),
                ],
            )
        ]
    )
    response, _ = await run_minimal_llm(wal_request(), make_deps(FakeMinimalLLM(output=output)))
    assert response.claims[0].verdict is Verdict.UNSUPPORTED  # nothing verifiable survived


async def test_empty_answer_skips_the_model():
    llm = FakeMinimalLLM()
    request = VerifyRequest(context_passages=[PASSAGE_0], generated_answer="   ")
    response, _ = await run_minimal_llm(request, make_deps(llm))
    assert response.claims == [] and llm.calls == []


async def test_run_request_routes_by_method():
    llm = FakeMinimalLLM(output=wal_output())
    response, _ = await run_request(wal_request(), make_deps(llm))
    assert len(response.claims) == 3

    # "baseline" now routes to method 1 (no LLM call — the fake stays untouched)
    deps = make_deps(llm)
    deps.settings.method = "baseline"
    calls_before = len(llm.calls)
    response, _ = await run_request(wal_request(), deps)
    assert response.claims and len(llm.calls) == calls_before


def test_service_with_minimal_method():
    settings = Settings(_env_file=None, method="minimal_llm", openai_api_key="t")
    deps = Dependencies(
        settings=settings, llm=FakeMinimalLLM(output=wal_output()), decider=ForbiddenDecider(), conn=None
    )
    app = create_app(settings, dependencies=deps)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/verify",
            json={"context_passages": [PASSAGE_0, PASSAGE_1], "generated_answer": ANSWER},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert {c["verdict"] for c in body["claims"]} == {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
