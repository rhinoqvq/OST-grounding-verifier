"""End-to-end pipeline tests on the assignment's WAL example, fully scripted."""

import pytest

from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM, scripted_decider, verdict_answer
from verifier.harness.config import Settings
from verifier.harness.errors import InputTooLarge, MalformedModelOutput, UpstreamError
from verifier.harness.pipeline import Dependencies, run_verification
from verifier.models.decider import Answer
from verifier.output.schemas import Verdict, VerifyRequest


def make_deps(llm, decider, **settings_overrides) -> Dependencies:
    settings = Settings(
        _env_file=None, openai_api_key="test", typesafe_api_key="test", **settings_overrides
    )
    return Dependencies(settings=settings, llm=llm, decider=decider, conn=None)


@pytest.fixture
def wal_request() -> VerifyRequest:
    return VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)


async def test_wal_example_end_to_end(wal_request, wal_proposal, wal_decider):
    llm = FakeLLM(proposal=wal_proposal)
    response, trace = await run_verification(wal_request, make_deps(llm, wal_decider))

    assert len(response.claims) == 4
    verdicts = [c.verdict for c in response.claims]
    assert verdicts == [
        Verdict.CONTRADICTED,   # NFS claim
        Verdict.SUPPORTED,      # two auxiliary files
        Verdict.SUPPORTED,      # checkpoints at ~1000 pages
        Verdict.UNSUPPORTED,    # AES-256 encryption
    ]
    nfs = response.claims[0]
    assert nfs.citations[0].index == 0
    assert nfs.citations[0].quote in PASSAGE_0
    assert "network filesystem" in nfs.citations[0].quote
    aux = response.claims[1]
    assert aux.citations[0].index == 1
    assert aux.citations[0].quote in PASSAGE_1
    assert response.claims[3].citations == []
    # the split sentence passed its whole-sentence recheck (round 4, no override)
    round_names = [r["name"] for r in trace.as_dict()["rounds"]]
    assert "4-parent-recheck" in round_names


async def test_llm_never_sees_the_sources(wal_request, wal_proposal, wal_decider):
    llm = FakeLLM(proposal=wal_proposal)
    await run_verification(wal_request, make_deps(llm, wal_decider))
    for messages in llm.calls:
        joined = " ".join(m["content"] for m in messages)
        assert "network filesystem" not in joined
        assert "Automatic checkpointing" not in joined


async def test_answer_numbers_are_not_corrected(wal_proposal, wal_decider):
    """The claim must keep the answer's 1,000 wording, not the source's 1000."""
    request = VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)
    llm = FakeLLM(proposal=wal_proposal)
    response, _ = await run_verification(request, make_deps(llm, wal_decider))
    checkpoint_claim = response.claims[2]
    assert "roughly 1,000 pages" in checkpoint_claim.claim_text


async def test_empty_answer_returns_no_claims(wal_decider):
    request = VerifyRequest(context_passages=[PASSAGE_0], generated_answer="   ")
    response, _ = await run_verification(request, make_deps(FakeLLM(), wal_decider))
    assert response.claims == []


async def test_empty_context_everything_unsupported(wal_proposal):
    request = VerifyRequest(context_passages=[], generated_answer=ANSWER)
    llm = FakeLLM(proposal=wal_proposal)
    response, _ = await run_verification(request, make_deps(llm, scripted_decider()))
    assert len(response.claims) == 4
    assert all(c.verdict is Verdict.UNSUPPORTED for c in response.claims)
    assert all(c.citations == [] for c in response.claims)
    assert all("empty" in c.reasoning for c in response.claims)


async def test_non_claim_only_answer(wal_decider):
    from verifier.semantics.propose import Proposal, TokenRange

    request = VerifyRequest(context_passages=[PASSAGE_0], generated_answer="Hope this helps!")
    llm = FakeLLM(proposal=Proposal(units=[], non_claim_token_ranges=[TokenRange(first=0, last=2)]))
    response, _ = await run_verification(request, make_deps(llm, wal_decider))
    assert response.claims == []


async def test_malformed_proposal_degrades_to_sentence_units(wal_request):
    llm = FakeLLM(error=MalformedModelOutput("openai", "twice"))
    script = {
        "v::sent::A.S0": verdict_answer("v::sent::A.S0", "contradicted"),
        "r::sent::A.S0::P0.S2": Answer("r::sent::A.S0::P0.S2", yes_prob=0.9),
    }
    response, trace = await run_verification(wal_request, make_deps(llm, scripted_decider(script)))
    assert len(response.claims) == 3  # one unit per answer sentence
    assert response.claims[0].verdict is Verdict.CONTRADICTED
    assert any("falling back to whole-sentence units" in n for n in trace.as_dict()["notes"])


async def test_input_too_large(wal_decider):
    request = VerifyRequest(context_passages=["x"], generated_answer="word " * 5000)
    with pytest.raises(InputTooLarge):
        await run_verification(request, make_deps(FakeLLM(), wal_decider, max_answer_chars=100))


async def test_upstream_error_propagates(wal_request, wal_proposal):
    class FailingDecider:
        name = "fail"

        async def ask(self, state, questions):
            raise UpstreamError(provider="jev", kind="transient", retryable=True)

    with pytest.raises(UpstreamError):
        await run_verification(
            wal_request, make_deps(FakeLLM(proposal=wal_proposal), FailingDecider())
        )


async def test_low_confidence_becomes_unsupported(wal_request, wal_proposal):
    flat = {
        "v::u1": Answer("v::u1", choice="supported",
                        probabilities={"supported": 0.4, "contradicted": 0.3, "not_in_context": 0.3}),
        "r::u1::P0.S2": Answer("r::u1::P0.S2", yes_prob=0.9),
    }
    response, _ = await run_verification(
        wal_request, make_deps(FakeLLM(proposal=wal_proposal), scripted_decider(flat))
    )
    claim = response.claims[0]
    assert claim.verdict is Verdict.UNSUPPORTED
    assert "too flat" in claim.reasoning


async def test_verdict_without_evidence_downgrades(wal_request, wal_proposal):
    script = {"v::u1": verdict_answer("v::u1", "contradicted")}  # no relevant sentence scripted
    response, _ = await run_verification(
        wal_request, make_deps(FakeLLM(proposal=wal_proposal), scripted_decider(script))
    )
    claim = response.claims[0]
    assert claim.verdict is Verdict.UNSUPPORTED
    assert claim.citations == []


async def test_parent_recheck_overrides_split(wal_request, wal_proposal, wal_decider):
    """If the whole sentence fails its recheck, the split units are replaced."""
    wal_decider.script["v::sent::A.S1"] = verdict_answer("v::sent::A.S1", "not_in_context")
    response, trace = await run_verification(
        wal_request, make_deps(FakeLLM(proposal=wal_proposal), wal_decider)
    )
    texts = [c.claim_text for c in response.claims]
    assert any(t.startswith("Alongside the main database file, it uses") for t in texts)
    assert len(response.claims) == 3  # u2+u3 merged back into the sentence unit
    merged = next(c for c in response.claims if c.claim_text.startswith("Alongside"))
    assert merged.verdict is Verdict.UNSUPPORTED
