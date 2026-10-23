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
    settings = Settings(_env_file=None, method="minimal_jev_seg", typesafe_api_key="t", **overrides)
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


# --- J+ (minimal_jev_plus): the optimized method's meta-discourse gate and way of asking ---

def plus_deps(decider) -> Dependencies:
    settings = Settings(_env_file=None, method="minimal_jev_plus", typesafe_api_key="t")
    return Dependencies(settings=settings, llm=ForbiddenLLM(), decider=decider, conn=None)


def nc(sid: str, meta: float) -> Answer:
    label = "meta_discourse" if meta >= 0.5 else "factual_claim"
    return Answer(f"nc::{sid}", choice=label,
                  probabilities={"meta_discourse": meta, "factual_claim": 1 - meta}, confidence=max(meta, 1 - meta))


def dist(qid: str, **probs: float) -> Answer:
    top = max(probs, key=probs.get)
    return Answer(qid, choice=top, probabilities=dict(probs), confidence=probs[top])


async def test_plus_gates_meta_discourse_and_asks_the_optimized_way():
    decider = FakeDecider({
        "nc::A.S0": nc("A.S0", 0.9), "nc::A.S1": nc("A.S1", 0.05), "nc::A.S2": nc("A.S2", 0.05),
        "v0": dist("v0", supported=0.9, contradicted=0.05, not_in_context=0.05), "e0": choice("e0", "P0.S2"),
        # a lean to S below the 0.55 bar stays S
        "v1": dist("v1", supported=0.5, contradicted=0.2, not_in_context=0.3), "e1": choice("e1", "P1.S0"),
        # a lean to C below the bar is U
        "v2": dist("v2", supported=0.2, contradicted=0.5, not_in_context=0.3), "e2": choice("e2", "P0.S2"),
    })
    response, trace = await run_request(wal_request(), plus_deps(decider))
    assert decider.usage.calls == 2  # the answer-only nc request and the judge request
    store = TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1])
    texts = [s.text for s in store.answer_sentences]
    assert [c.claim_text for c in response.claims] == texts[1:]  # the gated sentence yields no claim
    assert [c.verdict for c in response.claims] == [Verdict.SUPPORTED, Verdict.UNSUPPORTED]
    judge = next(qs for _, qs in decider.seen if any(q.question_id == "v0" for q in qs))
    v0 = next(q for q in judge if q.question_id == "v0")
    assert set(v0.options) == {"supported", "contradicted", "not_in_context"}
    assert "never against outside knowledge" in v0.instructions
    nc_request = next(qs for _, qs in decider.seen if any(q.question_id.startswith("nc::") for q in qs))
    assert {q.question_id for q in nc_request} == {"nc::A.S0", "nc::A.S1", "nc::A.S2"}


async def test_plus_definite_verdict_without_evidence_is_unsupported():
    decider = FakeDecider({
        "nc::A.S0": nc("A.S0", 0.0), "nc::A.S1": nc("A.S1", 0.0), "nc::A.S2": nc("A.S2", 0.0),
        "v0": dist("v0", supported=0.9, contradicted=0.05, not_in_context=0.05), "e0": choice("e0", NO_EVIDENCE),
        "v1": dist("v1", supported=0.1, contradicted=0.8, not_in_context=0.1), "e1": choice("e1", NO_EVIDENCE),
        "v2": dist("v2", supported=0.1, contradicted=0.1, not_in_context=0.8), "e2": choice("e2", NO_EVIDENCE),
    })
    response, _ = await run_request(wal_request(), plus_deps(decider))
    assert [c.verdict for c in response.claims] == [Verdict.UNSUPPORTED] * 3
    assert all(c.citations == [] for c in response.claims)


async def test_minimal_jev_keeps_its_own_wording():
    decider = wal_decider()
    await run_request(wal_request(), make_deps(decider))
    _, questions = decider.seen[0]
    v0 = next(q for q in questions if q.question_id == "v0")
    assert set(v0.options) == {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}


# --- J+nc / J+ME: J's verdict with the meta-discourse gate; ME takes the optimized evidence probes ---

def me_deps(decider, method: str) -> Dependencies:
    settings = Settings(_env_file=None, method=method, typesafe_api_key="t")
    return Dependencies(settings=settings, llm=ForbiddenLLM(), decider=decider, conn=None)


def j_script(**extra) -> dict:
    return {
        "nc::A.S0": nc("A.S0", 0.9), "nc::A.S1": nc("A.S1", 0.05), "nc::A.S2": nc("A.S2", 0.05),
        "v0": choice("v0", "SUPPORTED"), "e0": choice("e0", "P0.S2"),
        "v1": choice("v1", "SUPPORTED"), "e1": choice("e1", NO_EVIDENCE),  # J's pick found nothing
        "v2": choice("v2", "UNSUPPORTED"), "e2": choice("e2", NO_EVIDENCE),
        **extra,
    }


async def test_nc_is_j_without_the_gated_sentence():
    response, _ = await run_request(wal_request(), me_deps(FakeDecider(j_script()), "minimal_jev_nc"))
    store = TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1])
    assert [c.claim_text for c in response.claims] == [s.text for s in store.answer_sentences][1:]
    # S without J's evidence pick falls to U at the contract, as in J
    assert [c.verdict for c in response.claims] == [Verdict.UNSUPPORTED, Verdict.UNSUPPORTED]


async def test_me_takes_evidence_from_relevance_probes_with_rank_fallback():
    def probe(q):  # only P0.S1 scores, and only at the fallback level
        return Answer(q.question_id, yes_prob=0.4 if q.question_id == "r::1::P0.S1" else 0.1)

    # the recovered S (J's own pick was none) must pass the sufficiency check on what was found
    for check, expected in (("supported", Verdict.SUPPORTED), ("not_in_context", Verdict.UNSUPPORTED)):
        decider = FakeDecider(j_script(**{"cs::1": dist("cs::1", **{check: 0.8, "contradicted": 0.1})}),
                              default_factory=probe)
        response, _ = await run_request(wal_request(), me_deps(decider, "minimal_jev_me"))
        assert [c.verdict for c in response.claims] == [expected, Verdict.UNSUPPORTED]
    decider = FakeDecider(j_script(**{"cs::1": dist("cs::1", supported=0.8, contradicted=0.1)}),
                          default_factory=probe)
    response, _ = await run_request(wal_request(), me_deps(decider, "minimal_jev_me"))
    store = TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1])
    assert response.claims[0].citations[0].quote == store.sentence_by_id("P0.S1").text
    asked = {q.question_id.split("::")[1] for _, qs in decider.seen for q in qs if q.question_id.startswith("r::")}
    assert asked == {"1"}  # the gated sentence (0) and the U sentence (2) get no probes
