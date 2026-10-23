"""The optimized method's own whole-sentence call (stage 1 and the sentence guard): byte for
byte the request the single call (now minimal_jev_seg) sends, in every case, a long source
narrowed the same way included, so recorded calls replay."""

from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM
from verifier.harness.config import Settings
from verifier.harness.pipeline import Dependencies
from verifier.methods.minimal_jev.verifier import _ask_sentences
from verifier.methods.minimal_jev.verifier import verdict_question as single_call_verdict_question
from verifier.methods.optimized.judge.sentence_call import whole_sentence_calls
from verifier.models.decider import Answer, FakeDecider
from verifier.rag.source import TextStore


def any_answer(q) -> Answer:
    label = "SUPPORTED" if q.question_id.startswith("v") else "none"
    return Answer(q.question_id, choice=label, probabilities={label: 0.9}, confidence=0.9)


def plain_deps(decider) -> Dependencies:
    settings = Settings(_env_file=None, typesafe_api_key="t")
    return Dependencies(settings=settings, llm=FakeLLM(None), decider=decider, conn=None)


async def requests_of(store: TextStore) -> tuple[list, list, dict]:
    """(what the single call sends, what the optimized method's own call sends, its info)."""
    old, new = FakeDecider(default_factory=any_answer), FakeDecider(default_factory=any_answer)
    _, old_info = await _ask_sentences(store, plain_deps(old), single_call_verdict_question)
    _, info = await whole_sentence_calls(store, plain_deps(new))
    assert info == old_info
    key = lambda seen: sorted(seen, key=lambda r: r[1][0].question_id)  # noqa: E731
    return key(old.seen), key(new.seen), info


async def test_the_same_request_when_the_source_fits():
    old, new, info = await requests_of(TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1]))
    assert len(new) == 1 and new == old
    assert info["source_sentences_kept"] == len(TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1]).all_source_sentences())


async def test_a_long_answer_is_batched_the_same_way():
    answer = " ".join(f"The cafe sells item number {i} every day." for i in range(26))
    passage = "The cafe sells items. It opens daily. Prices vary."
    old, new, _ = await requests_of(TextStore.from_request(answer, [passage]))
    assert len(new) == 2 and new == old  # 52 questions, at most 48 per request


async def test_a_source_over_the_option_limit_is_narrowed_the_same_way():
    passage = " ".join(f"Unrelated sentence number {i} about gardening." for i in range(300))
    passage += " WAL does not work over a network filesystem."
    old, new, info = await requests_of(TextStore.from_request("WAL works over a network filesystem.", [passage]))
    assert new == old and info["source_sentences_kept"] <= 254
    assert "network filesystem" in new[0][0]  # the overlapping sentence is kept


async def test_a_source_over_the_token_budget_is_narrowed_the_same_way():
    long_sentence = "The committee reviewed the annual budget report in great detail " * 24
    passage = " ".join(f"{long_sentence.strip()} number {i}." for i in range(200))
    store = TextStore.from_request("The committee reviewed the budget.", [passage])
    old, new, info = await requests_of(store)
    assert new == old and info["source_sentences_kept"] < len(store.all_source_sentences())
