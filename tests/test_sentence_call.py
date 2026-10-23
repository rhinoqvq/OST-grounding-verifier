"""The optimized method's own whole-sentence call (stage 1 and the sentence guard): the same
request the single call sent while the source fits, and no call at all when it does not
(the source is never narrowed; every sentence takes the full path)."""

from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM, scripted_decider
from test_rc2 import asked, deps_with
from verifier.harness.config import Settings
from verifier.harness.pipeline import Dependencies
from verifier.methods.minimal_jev.verifier import _ask_sentences
from verifier.methods.minimal_jev.verifier import verdict_question as single_call_verdict_question
from verifier.methods.optimized.judge.sentence_call import (
    MAX_SOURCE_SENTENCES,
    sentence_call_request,
    whole_sentence_calls,
)
from verifier.methods.optimized.pipeline import run_verification
from verifier.models.decider import Answer, FakeDecider
from verifier.output.schemas import VerifyRequest
from verifier.rag.source import TextStore


def any_answer(q) -> Answer:
    label = "SUPPORTED" if q.question_id.startswith("v") else "none"
    return Answer(q.question_id, choice=label, probabilities={label: 0.9}, confidence=0.9)


def plain_deps(decider) -> Dependencies:
    settings = Settings(_env_file=None, typesafe_api_key="t")
    return Dependencies(settings=settings, llm=FakeLLM(None), decider=decider, conn=None)


async def requests_of(store: TextStore) -> tuple[list, list]:
    """(what the single call sends, what the optimized method's own call sends)."""
    old, new = FakeDecider(default_factory=any_answer), FakeDecider(default_factory=any_answer)
    await _ask_sentences(store, plain_deps(old), single_call_verdict_question)
    calls, _ = await whole_sentence_calls(store, plain_deps(new))
    assert calls is not None
    key = lambda seen: sorted(seen, key=lambda r: r[1][0].question_id)  # noqa: E731
    return key(old.seen), key(new.seen)


async def test_the_request_is_the_single_calls_when_the_source_fits():
    """Same state, questions and options, so the same cache key: recorded calls replay."""
    old, new = await requests_of(TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1]))
    assert len(new) == 1 and new == old


async def test_a_long_answer_is_batched_the_same_way():
    answer = " ".join(f"The cafe sells item number {i} every day." for i in range(26))
    passage = "The cafe sells items. It opens daily. Prices vary."
    old, new = await requests_of(TextStore.from_request(answer, [passage]))
    assert len(new) == 2 and new == old  # 52 questions, at most 48 per request


async def test_a_source_over_the_option_limit_gets_no_call_and_is_not_narrowed():
    passage = " ".join(f"Unrelated sentence number {i} about gardening." for i in range(300))
    store = TextStore.from_request("WAL works over a network filesystem.", [passage])
    reason = sentence_call_request(store)
    assert isinstance(reason, str) and f"300 sentences > {MAX_SOURCE_SENTENCES}" in reason
    decider = FakeDecider(default_factory=any_answer)
    calls, info = await whole_sentence_calls(store, plain_deps(decider))
    assert calls is None and info == {"skipped": reason} and decider.seen == []
    # the single call narrows the same source instead
    _, single = await _ask_sentences(store, plain_deps(FakeDecider(default_factory=any_answer)),
                                     single_call_verdict_question)
    assert single["source_sentences_kept"] <= MAX_SOURCE_SENTENCES


async def test_a_source_over_the_token_budget_gets_no_call():
    long_sentence = "The committee reviewed the annual budget report in great detail " * 24
    passage = " ".join(f"{long_sentence.strip()} number {i}." for i in range(200))
    store = TextStore.from_request("The committee reviewed the budget.", [passage])
    assert len(store.all_source_sentences()) <= MAX_SOURCE_SENTENCES
    reason = sentence_call_request(store)
    assert isinstance(reason, str) and "does not fit" in reason


def filler(n: int) -> str:
    return " ".join(f"Unrelated sentence number {i} about gardening." for i in range(n))


async def test_stage1_skipped_sends_every_sentence_down_the_full_path(wal_proposal):
    request = VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1 + " " + filler(300)], generated_answer=ANSWER)
    decider = scripted_decider()
    response, trace = await run_verification(
        request, deps_with(decider, FakeLLM(wal_proposal), two_stage=True, guard_scope="all",
                           parent_recheck=False, citation_policy="probe", source_tree="off"))
    qs = asked(decider)
    assert not [q for q in qs if q[:1] in "ve" and q[1:].isdigit()]  # no whole-sentence call
    rounds = {r["name"]: r["detail"] for r in trace.as_dict()["rounds"]}
    assert "skipped" in rounds["2.8-stage1-sentence-call"]
    assert rounds["2.8-stage1-sentence-call"]["settled_units"] == 0
    assert "skipped" in rounds["4b-sentence-guard"]  # the guard has no call either
    assert any(n.startswith("stage 1 skipped") for n in trace.as_dict()["notes"])
    assert {"v::u1", "v::u2", "v::u3"} <= set(qs)     # every unit judged the full way
    assert len(response.claims) == 4


async def test_guard_without_two_stage_is_skipped_for_a_source_too_large(wal_proposal):
    request = VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1 + " " + filler(300)], generated_answer=ANSWER)
    decider = scripted_decider()
    _, trace = await run_verification(request, deps_with(decider, FakeLLM(wal_proposal), source_tree="off"))
    assert not [q for q in asked(decider) if q[:1] in "ve" and q[1:].isdigit()]
    assert any(n.startswith("sentence guard skipped") for n in trace.as_dict()["notes"])
