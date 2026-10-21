"""The "make a choice" interface.

Every model-made decision in the pipeline goes through this interface: a batch
of bounded questions against one state. Implementations: Jev (primary), a
generative LLM (comparison arm, later), and a scripted fake for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# Jev rejects a choice question with more options than this (422).
MAX_CHOICE_OPTIONS = 255


@dataclass(frozen=True)
class ChoiceQuestion:
    question_id: str
    instructions: str
    options: dict[str, str]  # label -> what the label means


@dataclass(frozen=True)
class YesNoQuestion:
    question_id: str
    instructions: str


Question = ChoiceQuestion | YesNoQuestion


@dataclass
class Answer:
    question_id: str
    choice: str | None = None                       # chosen label (choice questions)
    probabilities: dict[str, float] = field(default_factory=dict)
    yes_prob: float | None = None                   # yes/no questions
    confidence: float | None = None
    raw: Any = None


@dataclass
class DeciderUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "DeciderUsage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class Decider(Protocol):
    name: str

    async def ask(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """Answer every question against the same state. Questions are
        independent of each other; implementations may batch them freely."""
        ...


class FakeDecider:
    """Scripted decider for tests and offline runs.

    `script` maps question_id -> Answer. Unknown questions fall back to
    `default_factory`, or raise if none is given (so tests notice gaps).
    """

    name = "fake"

    def __init__(self, script: dict[str, Answer] | None = None, default_factory=None):
        self.script = script or {}
        self.default_factory = default_factory
        self.seen: list[tuple[str, list[Question]]] = []
        self.usage = DeciderUsage()

    async def ask(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        self.seen.append((state, list(questions)))
        self.usage.calls += 1
        out: dict[str, Answer] = {}
        for q in questions:
            if q.question_id in self.script:
                out[q.question_id] = self.script[q.question_id]
            elif self.default_factory is not None:
                out[q.question_id] = self.default_factory(q)
            else:
                raise KeyError(f"FakeDecider has no scripted answer for {q.question_id!r}")
        return out
