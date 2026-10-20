"""Shared fixtures: the assignment's SQLite/WAL example plus scripted fakes.

No test in this suite touches a real API. The FakeDecider/FakeLLM pair scripts
exactly the behavior each test needs, so failures point at our logic.
"""

from __future__ import annotations

import pytest

from verifier.models.decider import Answer, ChoiceQuestion, FakeDecider, YesNoQuestion
from verifier.rag.source import TextStore
from verifier.semantics.propose import Proposal, ProposedUnit, TokenRange

PASSAGE_0 = (
    "Beginning with version 3.7.0, Write-Ahead Log (WAL) mode is available. "
    "Advantages include significantly faster performance in most scenarios and increased "
    "concurrency: readers do not block writers and writers do not block readers. However, "
    "WAL does not work over a network filesystem because it requires all processes to share "
    "a small amount of memory via shared memory files (-shm)."
)
PASSAGE_1 = (
    "When in active use, the state of a WAL mode database is described by three separate "
    "files: the main database file 'X', the write-ahead log file 'X-wal', and the wal-index "
    "shared-memory file 'X-shm'. Automatic checkpointing occurs by default when the WAL "
    "passes approximately 1000 pages."
)
ANSWER = (
    "SQLite WAL mode allows multiple processes across different host machines on an NFS "
    "network share to write concurrently. Alongside the main database file, it uses two "
    "auxiliary files (-wal and -shm) and automatically checkpoints when the log reaches "
    "roughly 1,000 pages. By default, WAL files are encrypted with AES-256 at rest."
)


@pytest.fixture
def wal_store() -> TextStore:
    return TextStore.from_request(ANSWER, [PASSAGE_0, PASSAGE_1])


def token_range_for(store: TextStore, fragment: str) -> TokenRange:
    """Inclusive token range covering a verbatim answer fragment (test helper)."""
    start = store.answer.index(fragment)
    end = start + len(fragment)
    covered = [t.index for t in store.answer_tokens if t.span.start >= start and t.span.end <= end]
    assert covered, f"no tokens inside {fragment!r}"
    return TokenRange(first=covered[0], last=covered[-1])


@pytest.fixture
def wal_proposal(wal_store: TextStore) -> Proposal:
    s = wal_store
    return Proposal(
        units=[
            ProposedUnit(
                unit_id="u1",
                kind="fact",
                anchor_token_ranges=[token_range_for(s, "SQLite WAL mode allows multiple processes across different host machines on an NFS network share to write concurrently.")],
                display_text="SQLite WAL mode allows multiple processes across different host machines on an NFS network share to write concurrently.",
            ),
            ProposedUnit(
                unit_id="u2",
                kind="fact",
                anchor_token_ranges=[token_range_for(s, "Alongside the main database file, it uses two auxiliary files (-wal and -shm)")],
                display_text="Alongside the main database file, SQLite WAL mode uses two auxiliary files (-wal and -shm).",
            ),
            ProposedUnit(
                unit_id="u3",
                kind="fact",
                anchor_token_ranges=[token_range_for(s, "automatically checkpoints when the log reaches roughly 1,000 pages.")],
                inherited_token_ranges=[token_range_for(s, "it uses")],
                display_text="SQLite WAL mode automatically checkpoints when the log reaches roughly 1,000 pages.",
            ),
            ProposedUnit(
                unit_id="u4",
                kind="fact",
                anchor_token_ranges=[token_range_for(s, "By default, WAL files are encrypted with AES-256 at rest.")],
                display_text="By default, WAL files are encrypted with AES-256 at rest.",
            ),
        ],
    )


class FakeLLM:
    """Scripted stand-in for OpenAILLM.generate_structured."""

    def __init__(self, proposal: Proposal | None = None, error: Exception | None = None):
        self.proposal = proposal
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    async def generate_structured(self, messages, schema):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        assert schema is Proposal
        return self.proposal


def scripted_decider(script: dict[str, Answer] | None = None) -> FakeDecider:
    """FakeDecider with sensible defaults: units pass fidelity, coverage is
    fine, verdicts default to not_in_context, relevance defaults to no."""

    def default(question) -> Answer:
        qid = question.question_id
        # vote copies of a cut question (cut::...::vN) reuse the scripted
        # answer for the base question, so scripted tests are vote-agnostic
        if qid.startswith("cut::") and "::v" in qid:
            base = qid.rsplit("::v", 1)[0]
            if base in (script or {}):
                import dataclasses

                return dataclasses.replace(script[base], question_id=qid)
        if qid.startswith("g::"):
            return Answer(qid, yes_prob=0.9)
        if qid.startswith("recomb::"):
            return Answer(qid, yes_prob=0.9)
        if qid.startswith("rel1::") and isinstance(question, ChoiceQuestion):
            probs = dict.fromkeys(question.options, 0.02)
            probs["no_such_link"] = 0.9
            return Answer(qid, choice="no_such_link", probabilities=probs, confidence=0.9)
        if qid.startswith(("rel2::", "rel3::")) and isinstance(question, ChoiceQuestion):
            probs = {"matches": 0.9, "opposite": 0.05, "different": 0.05}
            return Answer(qid, choice="matches", probabilities=probs, confidence=0.9)
        if qid.startswith("rel4::") and isinstance(question, ChoiceQuestion):
            probs = {"left_first": 0.9, "right_first": 0.05, "no_order": 0.05}
            return Answer(qid, choice="left_first", probabilities=probs, confidence=0.9)
        if qid.startswith("rel5::"):
            # default: the link sentence asserts the link (P-assert holds)
            return Answer(qid, yes_prob=0.9)
        if qid.startswith("rel6::") and isinstance(question, ChoiceQuestion):
            probs = dict.fromkeys(question.options, 0.02)
            probs["no_discharge"] = 0.9
            return Answer(qid, choice="no_discharge", probabilities=probs, confidence=0.9)
        if qid.startswith("rel7::") and isinstance(question, ChoiceQuestion):
            probs = dict.fromkeys(question.options, 0.02)
            probs["none_needed"] = 0.9
            return Answer(qid, choice="none_needed", probabilities=probs, confidence=0.9)
        if qid.startswith("br1::") and isinstance(question, ChoiceQuestion):
            probs = dict.fromkeys(question.options, 0.02)
            probs["no_connection"] = 0.9
            return Answer(qid, choice="no_connection", probabilities=probs, confidence=0.9)
        if qid.startswith("br3::") and isinstance(question, ChoiceQuestion):
            probs = dict.fromkeys(question.options, 0.02)
            probs["no_property"] = 0.9
            return Answer(qid, choice="no_property", probabilities=probs, confidence=0.9)
        if qid.startswith(("br2::", "br5::")):
            return Answer(qid, yes_prob=0.9)
        if qid.startswith("br4::") and isinstance(question, ChoiceQuestion):
            probs = {"matches": 0.9, "opposite": 0.05, "different": 0.05}
            return Answer(qid, choice="matches", probabilities=probs, confidence=0.9)
        if qid.startswith("c1::") and isinstance(question, ChoiceQuestion):
            # default: the opposition is explicit, so scripted C verdicts stand
            probs = {"opposite": 0.9, "different": 0.05, "matches": 0.05}
            return Answer(qid, choice="opposite", probabilities=probs, confidence=0.9)
        if qid.startswith("cut::") and isinstance(question, ChoiceQuestion):
            probs = dict.fromkeys(question.options, 0.02)
            probs["keep_whole"] = 0.9
            return Answer(qid, choice="keep_whole", probabilities=probs, confidence=0.9)
        if qid.startswith("fid::") and isinstance(question, ChoiceQuestion):
            return Answer(qid, choice="faithful",
                          probabilities={"faithful": 0.92, "changed_meaning": 0.03,
                                         "added_information": 0.03, "lost_qualifier": 0.02},
                          confidence=0.9)
        if qid.startswith("cov::") and isinstance(question, YesNoQuestion):
            return Answer(qid, yes_prob=0.05)
        if qid.startswith("v::") and isinstance(question, ChoiceQuestion):
            return Answer(qid, choice="not_in_context",
                          probabilities={"supported": 0.05, "contradicted": 0.05,
                                         "not_in_context": 0.90},
                          confidence=0.9)
        if qid.startswith("r::"):
            return Answer(qid, yes_prob=0.08)
        raise KeyError(f"unexpected question {qid}")

    return FakeDecider(script=script or {}, default_factory=default)


def verdict_answer(qid: str, label: str, p: float = 0.9) -> Answer:
    rest = (1 - p) / 2
    probabilities = {"supported": rest, "contradicted": rest, "not_in_context": rest}
    probabilities[label] = p
    return Answer(qid, choice=label, probabilities=probabilities, confidence=p)


@pytest.fixture
def wal_decider() -> FakeDecider:
    """Scripted judgments matching the assignment's expected output."""
    script = {
        "v::u1": verdict_answer("v::u1", "contradicted"),
        "r::u1::P0.S2": Answer("r::u1::P0.S2", yes_prob=0.95),
        "v::u2": verdict_answer("v::u2", "supported"),
        "r::u2::P1.S0": Answer("r::u2::P1.S0", yes_prob=0.93),
        "v::u3": verdict_answer("v::u3", "supported"),
        "r::u3::P1.S1": Answer("r::u3::P1.S1", yes_prob=0.94),
        "v::u4": verdict_answer("v::u4", "not_in_context"),
        # round-4 recheck of the split sentence (u2+u3 both supported)
        "v::sent::A.S1": verdict_answer("v::sent::A.S1", "supported"),
        "r::sent::A.S1::P1.S0": Answer("r::sent::A.S1::P1.S0", yes_prob=0.9),
        "r::sent::A.S1::P1.S1": Answer("r::sent::A.S1::P1.S1", yes_prob=0.9),
    }
    return scripted_decider(script)
