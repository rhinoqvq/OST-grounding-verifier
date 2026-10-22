"""Method 2, Jev variant: one judge request per answer, sentence claims, code-sliced quotes."""

from conftest import ANSWER, PASSAGE_0, PASSAGE_1
from verifier.harness.config import Settings
from verifier.harness.pipeline import Dependencies, run_request
from verifier.methods.minimal_jev.verifier import NO_EVIDENCE, fit_source
from verifier.models.decider import Answer, ChoiceQuestion, FakeDecider
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.rag.source import TextStore


class ForbiddenLLM:
    """The Jev variant must never call the generative model."""

    async def generate_structured(self, messages, schema):  # pragma: no cover - failing is the point
        raise AssertionError("minimal_jev must not call the LLM")


def choice(qid: str, label: str, p: float = 0.9) -> Answer:
    return Answer(qid, choice=label, probabilities={label: p}, confidence=p)


def make_deps(decider, **overrides) -> Dependencies:
    settings = Settings(_env_file=None, method="minimal_jev", typesafe_api_key="t", **overrides)
    return Dependencies(settings=settings, llm=ForbiddenLLM(), decider=decider, conn=None)


def wal_request() -> VerifyRequest:
    return VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)


def wal_decider() -> FakeDecider:
    return FakeDecider({
        "v0": choice("v0", "CONTRADICTED"), "e0": choice("e0", "P0.S2"),
        "v1": choice("v1", "SUPPORTED"), "e1": choice("e1", "P1.S0"),
        "v2": choice("v2", "UNSUPPORTED"), "e2": choice("e2", NO_EVIDENCE),
    })


async def test_one_request_per_answer_with_sentence_claims_and_sliced_quotes():
    decider = wal_decider()
    response, trace = await run_request(wal_request(), make_deps(decider))
    assert decider.usage.calls == 1
    state, questions = decider.seen[0]
    assert {q.question_id for q in questions} == {"v0", "e0", "v1", "e1", "v2", "e2"}
    assert all(isinstance(q, ChoiceQuestion) for q in questions)
    assert [c.verdict for c in response.claims] == [
        Verdict.CONTRADICTED, Verdict.SUPPORTED, Verdict.UNSUPPORTED,
    ]
    store = TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1])
    assert [c.claim_text for c in response.claims] == [s.text for s in store.answer_sentences]
    nfs = response.claims[0].citations[0]
    assert nfs.index == 0 and nfs.quote in PASSAGE_0 and "network filesystem" in nfs.quote
    assert response.claims[2].citations == []


async def test_supported_without_evidence_is_downgraded_by_the_contract():
    decider = FakeDecider({
        "v0": choice("v0", "SUPPORTED"), "e0": choice("e0", NO_EVIDENCE),
        "v1": choice("v1", "SUPPORTED"), "e1": choice("e1", "P1.S0"),
        "v2": choice("v2", "UNSUPPORTED"), "e2": choice("e2", NO_EVIDENCE),
    })
    response, _ = await run_request(wal_request(), make_deps(decider))
    assert response.claims[0].verdict is Verdict.UNSUPPORTED
    assert response.claims[0].citations == []


async def test_empty_context_needs_no_request():
    decider = FakeDecider({})
    request = VerifyRequest(context_passages=[], generated_answer=ANSWER)
    response, _ = await run_request(request, make_deps(decider))
    assert decider.usage.calls == 0
    assert all(c.verdict is Verdict.UNSUPPORTED for c in response.claims)


async def test_a_long_answer_keeps_a_source_that_fits():
    """2026-10-23: the option cost was counted at the 255-option cap, so a 13+ sentence
    answer kept no source at all and every sentence was judged against nothing."""
    answer = " ".join(f"The cafe sells item number {i} every day." for i in range(14))
    passage = " ".join(f"The cafe sells item number {i} on weekdays." for i in range(18))
    decider = FakeDecider(default_factory=lambda q: choice(
        q.question_id, "SUPPORTED" if q.question_id.startswith("v") else "P0.S0"))
    request = VerifyRequest(context_passages=[passage], generated_answer=answer)
    response, trace = await run_request(request, make_deps(decider))
    state, questions = decider.seen[0]
    assert decider.usage.calls == 1
    assert "item number 17 on weekdays" in state
    evidence = [q for q in questions if q.question_id.startswith("e")]
    assert len(evidence) == 14 and all(len(q.options) == 19 for q in evidence)
    kept = trace.as_dict()["rounds"][1]["detail"]["source_sentences_kept"]
    assert kept == 18


def test_large_source_is_narrowed_by_overlap_and_kept_in_order():
    filler = " ".join(f"Unrelated sentence number {i} about gardening." for i in range(400))
    passage = filler + " WAL does not work over a network filesystem."
    store = TextStore.from_request("WAL works over a network filesystem.", [passage])
    kept = fit_source(store, ["WAL works over a network filesystem."], max_tokens=2_000, limit=50)
    assert len(kept) <= 50
    assert any("network filesystem" in s.text for s in kept)
    order = [store.all_source_sentences().index(s) for s in kept]
    assert order == sorted(order)
