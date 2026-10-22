"""Whole-sentence guard: structural alarms stand only when the whole-sentence call is not S."""

from verifier.methods.minimal_jev.verifier import SentenceCall
from verifier.methods.optimized.judge.entail import UnitJudgment
from verifier.methods.optimized.judge.guard import apply_sentence_guard, guarded_layer
from verifier.output.schemas import Verdict
from verifier.rag.source import Span
from verifier.rag.tree import Node, NodeKind

S, C, U = Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.UNSUPPORTED


def node(node_id: str, kind: NodeKind, sentence_id: str = "A.S0") -> Node:
    return Node(node_id=node_id, origin="answer", sentence_id=sentence_id, kind=kind,
                anchor_spans=[Span("answer", 0, 10)])


def judged(n: Node, verdict: Verdict, *flags: str) -> UnitJudgment:
    return UnitJudgment(node_id=n.node_id, node_version=n.version, verdict=verdict, flags=list(flags))


def test_layer_of_each_alarm():
    fact, rel, sent = node("f", NodeKind.FACT), node("r", NodeKind.RELATION), node("s", NodeKind.SENTENCE)
    assert guarded_layer(fact, judged(fact, U)) == "fragment"
    assert guarded_layer(rel, judged(rel, U, "staged_flat_link")) == "relation"
    assert guarded_layer(sent, judged(sent, U, "ungrounded_part_0(p=0.06)")) == "gate"
    assert guarded_layer(sent, judged(sent, U, "low_confidence(top=not_in_context,p=0.49)")) == "low_confidence"
    assert guarded_layer(fact, judged(fact, U, "c_filter_demoted(matches,p=0.98)")) == "c_filter"
    # the judge's own whole-sentence call is not a structural layer
    assert guarded_layer(sent, judged(sent, U)) is None
    assert guarded_layer(sent, judged(sent, C)) is None


def test_structural_alarm_is_released_when_the_whole_sentence_is_supported():
    fact, gate = node("f", NodeKind.FACT), node("s", NodeKind.SENTENCE, "A.S1")
    judgments = {"f": judged(fact, U), "s": judged(gate, U, "ungrounded_part_0(p=0.06)")}
    calls = {"A.S0": SentenceCall(S, "P0.S2", 0.9), "A.S1": SentenceCall(S, "P1.S0", 0.8)}
    assert apply_sentence_guard([fact, gate], judgments, calls) == ["f", "s"]
    assert judgments["f"].verdict is S and judgments["f"].evidence_sentence_ids == ["P0.S2"]
    assert "sentence_guard(fragment)" in judgments["f"].flags
    assert judgments["s"].verdict is S and judgments["s"].evidence_sentence_ids == ["P1.S0"]
    assert "sentence_guard(gate)" in judgments["s"].flags


def test_alarm_stands_when_the_whole_sentence_is_not_supported_or_has_no_evidence():
    rel, fact = node("r", NodeKind.RELATION), node("f", NodeKind.FACT, "A.S1")
    judgments = {"r": judged(rel, U, "staged_no_link"), "f": judged(fact, C)}
    calls = {"A.S0": SentenceCall(U, None, 0.7), "A.S1": SentenceCall(S, None, 0.6)}
    assert apply_sentence_guard([rel, fact], judgments, calls) == []
    assert judgments["r"].verdict is U and judgments["f"].verdict is C


def test_whole_sentence_call_and_supported_units_are_left_alone():
    sent, ok = node("s", NodeKind.SENTENCE), node("k", NodeKind.FACT)
    judgments = {"s": judged(sent, U), "k": judged(ok, S)}
    judgments["k"].evidence_sentence_ids = ["P0.S0"]
    calls = {"A.S0": SentenceCall(S, "P0.S3", 0.9)}
    assert apply_sentence_guard([sent, ok], judgments, calls) == []
    assert judgments["s"].verdict is U
    assert judgments["k"].evidence_sentence_ids == ["P0.S0"]
