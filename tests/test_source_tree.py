"""Two-tree check: source tree built with the answer's decomposition, answer
nodes mapped onto source nodes, whole-source judge as the fallback."""

import pytest

from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM, scripted_decider, verdict_answer
from verifier.harness.config import Settings
from verifier.harness.context import estimate_tokens
from verifier.harness.pipeline import Dependencies
from verifier.methods.optimized.judge.treemap import decide_fact_mapping, decide_relation_mapping, fact_mapping_plan
from verifier.methods.optimized.pipeline import run_verification
from verifier.methods.optimized.semantics.propose_jev import _LINK_RELATION, cut_question, find_candidates
from verifier.methods.optimized.semantics.source_tree import SourceTree, SourceUnit, build_source_tree
from verifier.models.decider import MAX_CHOICE_OPTIONS, Answer, ChoiceQuestion
from verifier.output.schemas import Verdict, VerifyRequest
from verifier.rag.source import Span, TextStore
from verifier.rag.tree import Node, NodeKind, RelationType


def make_deps(llm, decider, **overrides) -> Dependencies:
    settings = Settings(_env_file=None, openai_api_key="test", typesafe_api_key="test", **overrides)
    return Dependencies(settings=settings, llm=llm, decider=decider, conn=None)


def asked(decider) -> set[str]:
    return {q.question_id for _, qs in decider.seen for q in qs}


@pytest.fixture
def wal_request() -> VerifyRequest:
    return VerifyRequest(context_passages=[PASSAGE_0, PASSAGE_1], generated_answer=ANSWER)


async def test_a_mapped_fact_is_supported_without_the_whole_source_question(wal_request, wal_proposal):
    decider = scripted_decider({
        "v::u1": verdict_answer("v::u1", "contradicted"),
        "r::u1::P0.S2": Answer("r::u1::P0.S2", yes_prob=0.95),
        "m::u3::P1.S1": verdict_answer("m::u3::P1.S1", "supported"),
    })
    response, trace = await run_verification(wal_request, make_deps(FakeLLM(wal_proposal), decider))
    checkpoint = response.claims[2]
    assert checkpoint.verdict is Verdict.SUPPORTED
    assert checkpoint.citations[0].index == 1
    assert "Automatic checkpointing" in checkpoint.citations[0].quote
    assert "v::u3" not in asked(decider)
    assert any("source tree:" in n for n in trace.as_dict()["notes"])


async def test_a_mapped_fact_failing_the_presence_check_goes_to_the_whole_source_judge(wal_request, wal_proposal):
    decider = scripted_decider({
        "m::u3::P1.S1": verdict_answer("m::u3::P1.S1", "supported"),
        "g::u3::whole": Answer("g::u3::whole", yes_prob=0.1),
    })
    response, _ = await run_verification(wal_request, make_deps(FakeLLM(wal_proposal), decider))
    assert "v::u3" in asked(decider)
    assert response.claims[2].verdict is Verdict.UNSUPPORTED  # the scripted whole-source default


def test_claims_about_the_passages_are_never_mapped():
    store = TextStore.from_request("Passage 1 says the sky is blue. The sky is blue.", ["The sky is blue."])
    leaf = SourceUnit("P0.S0", NodeKind.SENTENCE, 0, "P0.S0", store.passage_sentences[0][0].span, "The sky is blue.")
    tree = SourceTree(units=[leaf], notes=[])
    about = Node(node_id="a", kind=NodeKind.FACT, origin="answer", sentence_id="A.S0",
                 anchor_spans=[store.answer_sentences[0].span], display_text=store.answer_sentences[0].text)
    plain = Node(node_id="b", kind=NodeKind.FACT, origin="answer", sentence_id="A.S1",
                 anchor_spans=[store.answer_sentences[1].span], display_text=store.answer_sentences[1].text)
    plan = fact_mapping_plan(store, tree, [about, plain], leaves_per_unit=4)
    ids = [q.question_id for items in plan.values() for q, _ in items]
    assert ids and all(i.startswith("m::b::") for i in ids)


async def test_units_mapped_into_one_source_sentence_skip_the_recheck(wal_request, wal_proposal):
    decider = scripted_decider({
        "m::u2::P1.S0": verdict_answer("m::u2::P1.S0", "supported"),
        "m::u3::P1.S0": verdict_answer("m::u3::P1.S0", "supported"),
    })
    await run_verification(wal_request, make_deps(FakeLLM(wal_proposal), decider))
    assert "v::sent::A.S1" not in asked(decider)


async def test_units_mapped_into_different_sentences_are_rechecked(wal_request, wal_proposal):
    decider = scripted_decider({
        "m::u2::P1.S0": verdict_answer("m::u2::P1.S0", "supported"),
        "m::u3::P1.S1": verdict_answer("m::u3::P1.S1", "supported"),
    })
    await run_verification(wal_request, make_deps(FakeLLM(wal_proposal), decider))
    assert "v::sent::A.S1" in asked(decider)


async def test_source_sentence_splits_into_fragments_and_a_cause_relation():
    passage = "The service restarts every night because memory is exhausted by the batch jobs."
    window = TextStore.from_request(passage, [])
    candidate = next(c for c in find_candidates(window) if c.connective == "because")
    question = cut_question(window, candidate)
    cause_label = next(k for k, v in _LINK_RELATION.items() if v is RelationType.CAUSE and k in question.options)
    probs = dict.fromkeys(question.options, 0.01)
    probs[cause_label] = 0.95
    decider = scripted_decider({
        f"src0.0|{candidate.question_id}": Answer(candidate.question_id, choice=cause_label, probabilities=probs),
    })
    store = TextStore.from_request("x", [passage])
    tree = await build_source_tree(decider, store, Settings(_env_file=None))
    kinds = {u.unit_id: u.kind for u in tree.units}
    assert kinds == {"P0.S0.f0": NodeKind.FACT, "P0.S0.f1": NodeKind.FACT, "P0.S0.r0": NodeKind.RELATION}
    relation = tree.relations()[0]
    assert relation.relation_type is RelationType.CAUSE
    assert relation.member_ids == ["P0.S0.f0", "P0.S0.f1"]
    fragment = store.sentence_by_id("P0.S0.f1")
    assert fragment.text in passage and fragment.text.startswith("memory is exhausted")


def _unit(node_id: str, kind=NodeKind.FACT, relation_type=None, members=None) -> Node:
    return Node(node_id=node_id, kind=kind, origin="answer", sentence_id="A.S0",
                anchor_spans=[Span("answer", 0, 5)], display_text="claim",
                relation_type=relation_type, member_ids=members or [])


def test_fragment_evidence_is_preferred_over_its_sentence():
    unit = _unit("u1")
    answers = {
        "m::u1::P0.S3": verdict_answer("m::u1::P0.S3", "supported", 0.95),
        "m::u1::P0.S3.f1": verdict_answer("m::u1::P0.S3.f1", "supported", 0.80),
    }
    judgment = decide_fact_mapping(unit, answers, min_prob=0.55)
    assert judgment.evidence_sentence_ids == ["P0.S3.f1"]
    assert decide_fact_mapping(unit, {"m::u1::P0.S3": verdict_answer("m::u1::P0.S3", "not_in_context")},
                               min_prob=0.55) is None


def test_a_relation_maps_only_when_both_sides_match():
    unit = _unit("r1", NodeKind.RELATION, RelationType.CAUSE, ["a", "b"])
    rel = SourceUnit("P0.S0.r0", NodeKind.RELATION, 0, "P0.S0", Span("context:0", 0, 10), "x",
                     relation_type=RelationType.CAUSE)
    side = {"matches": 0.9, "opposite": 0.05, "different": 0.05}
    answers = {
        "rel2::r1": Answer("rel2::r1", choice="matches", probabilities=side),
        "rel3::r1": Answer("rel3::r1", choice="matches", probabilities=side),
        "rel5::r1": Answer("rel5::r1", yes_prob=0.9),
    }
    judgment = decide_relation_mapping(unit, rel, answers, min_prob=0.55)
    assert judgment.verdict is Verdict.SUPPORTED and judgment.evidence_sentence_ids == ["P0.S0"]
    answers["rel3::r1"] = Answer("rel3::r1", choice="different",
                                 probabilities={"matches": 0.05, "opposite": 0.05, "different": 0.9})
    assert decide_relation_mapping(unit, rel, answers, min_prob=0.55) is None


async def test_long_sources_keep_states_and_option_lists_within_limits():
    passage = " ".join(f"Cluster {i} restarts its scheduler at {i % 24}:00 every day." for i in range(400))
    request = VerifyRequest(context_passages=[passage],
                            generated_answer="Cluster 7 restarts its scheduler at 7:00 every day.")
    decider = scripted_decider()
    await run_verification(request, make_deps(FakeLLM(), decider, proposer="jev", max_state_tokens=2_000))
    for state, questions in decider.seen:
        for q in questions:
            if isinstance(q, ChoiceQuestion):
                assert len(q.options) <= MAX_CHOICE_OPTIONS, q.question_id
        if state.startswith("Source passages"):
            assert estimate_tokens(state) <= 2_000
