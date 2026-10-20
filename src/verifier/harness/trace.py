"""Per-request trace: every round, call and decision, serializable to SQLite.

When a verdict is wrong, the trace answers "which step went wrong" without
re-running anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundTrace:
    name: str
    started: float
    duration_ms: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trace:
    rounds: list[RoundTrace] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def start_round(self, name: str) -> RoundTrace:
        r = RoundTrace(name=name, started=time.perf_counter())
        self.rounds.append(r)
        return r

    def end_round(self, r: RoundTrace, **detail: Any) -> None:
        r.duration_ms = (time.perf_counter() - r.started) * 1000
        r.detail.update(detail)

    def fail_round(self, r: RoundTrace, error: Exception) -> None:
        r.duration_ms = (time.perf_counter() - r.started) * 1000
        r.error = f"{type(error).__name__}: {error}"

    def note(self, message: str) -> None:
        self.notes.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rounds": [
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "detail": r.detail,
                    "error": r.error,
                }
                for r in self.rounds
            ],
            "notes": self.notes,
            "total_ms": sum(r.duration_ms or 0 for r in self.rounds),
        }
