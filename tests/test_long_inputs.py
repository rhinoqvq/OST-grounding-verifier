from verifier.harness.batching import plan_chunks
from verifier.harness.config import Settings
from verifier.harness.context import check_input_limits
from verifier.harness.errors import InputTooLarge
from verifier.methods.optimized.judge.entail import (
    aux_premise_question,
    bridge_find_questions,
    discharge_question,
    relation_link_question,
    sentence_options,
)
from verifier.models.decider import MAX_CHOICE_OPTIONS, ChoiceQuestion, YesNoQuestion
from verifier.rag.source import Span, TextStore
from verifier.rag.tree import Node, NodeKind, RelationType

import pytest


def _store(n_sentences: int, needle: str | None = None) -> TextStore:
    sentences = [f"Filler fact number {i} concerns the eastern cluster." for i in range(n_sentences)]
    if needle:
        sentences[n_sentences // 2] = needle
    return TextStore.from_request("The archive keeps exported reports for 400 days.", [" ".join(sentences)])


def _unit(store: TextStore) -> Node:
    return Node(
        node_id="u1", kind=NodeKind.RELATION, origin="answer", sentence_id="A.S0",
        anchor_spans=[Span("answer", 0, len(store.answer))],
        display_text=store.answer, relation_type=RelationType.CAUSE,
    )


def test_small_source_keeps_every_sentence_as_an_option():
    store = _store(30)
    options = sentence_options(store, "anything")
    assert list(options) == [s.sentence_id for s in store.all_source_sentences()]


def test_many_sentences_stay_under_the_option_cap_and_keep_the_relevant_one():
    needle = "The archive keeps exported reports for 400 days."
    store = _store(400, needle)
    needle_id = next(s.sentence_id for s in store.all_source_sentences() if s.text == needle)
    unit = _unit(store)
    link = store.all_source_sentences()[0]
    questions = [
        relation_link_question(store, unit),
        *bridge_find_questions(store, unit),
        discharge_question(store, unit, link),
        aux_premise_question(store, unit, link),
    ]
    for q in questions:
        assert isinstance(q, ChoiceQuestion)
        assert len(q.options) <= MAX_CHOICE_OPTIONS, q.question_id
    for q in (questions[0], questions[1], questions[2]):
        assert needle_id in q.options, q.question_id


def test_short_questions_chunk_by_count_as_before():
    qs = [YesNoQuestion(f"q{i}", "Is it so?") for i in range(100)]
    chunks = plan_chunks("short state", qs, max_per_request=48, max_tokens=28_000)
    assert [len(c) for c in chunks] == [48, 48, 4]


def test_chunks_split_when_the_token_budget_would_be_exceeded():
    long_options = {f"o{i}": "word " * 200 for i in range(40)}
    qs = [ChoiceQuestion(f"c{i}", "pick one", long_options) for i in range(6)]
    chunks = plan_chunks("state " * 2_000, qs, max_per_request=48, max_tokens=28_000)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) == 6


def test_context_up_to_100k_chars_is_accepted_and_beyond_is_refused():
    settings = Settings()
    check_input_limits("answer", ["x" * 50_000, "y" * 50_000], settings)
    check_input_limits("answer", ["p"] * 100, settings)
    with pytest.raises(InputTooLarge):
        check_input_limits("answer", ["x" * 50_000, "y" * 50_001], settings)
    with pytest.raises(InputTooLarge):
        check_input_limits("answer", ["p"] * 101, settings)
