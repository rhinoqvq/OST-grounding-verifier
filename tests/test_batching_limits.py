"""Request limits: a question too large for Jev on its own is not sent."""

from verifier.harness.batching import MAX_QUESTION_TOKENS, ask_batched, question_tokens
from verifier.models.decider import Answer, ChoiceQuestion, FakeDecider


def choice(qid: str, text: str) -> ChoiceQuestion:
    return ChoiceQuestion(question_id=qid, instructions=text, options={"a": "first", "b": "second"})


async def test_an_oversized_question_is_not_asked_and_gets_no_answer():
    """TRIVIA+ valid, 2026-10-23: a cut question repeating a table-sized source 'sentence' in
    every option was refused with max_tokens_exceeded and failed the whole answer."""
    small = choice("cut::A.S0::1", "Classify this marker.")
    huge = choice("src0.0|cut::A.S5::1626", "Source sentence: " + "word " * 60_000)
    assert question_tokens(huge) > MAX_QUESTION_TOKENS >= question_tokens(small)
    decider = FakeDecider(default_factory=lambda q: Answer(q.question_id, choice="a", probabilities={"a": 1.0}))
    answers = await ask_batched(decider, "state", [small, huge], max_per_request=48, max_parallel=4)
    assert set(answers) == {"cut::A.S0::1"}
    assert [q.question_id for _, qs in decider.seen for q in qs] == ["cut::A.S0::1"]


async def test_only_oversized_questions_means_no_request():
    huge = choice("src0.0|cut::A.S5::1748", "word " * 60_000)
    decider = FakeDecider(default_factory=lambda q: Answer(q.question_id, choice="a", probabilities={"a": 1.0}))
    assert await ask_batched(decider, "state", [huge], max_per_request=48, max_parallel=4) == {}
    assert decider.seen == []
