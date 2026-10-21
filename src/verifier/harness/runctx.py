"""The current run's id, visible to everything the run touches.

`run_request` assigns one id per request and sets it here; the model-call
cache stamps it onto every `model_calls` row and the run store reuses it for
the `runs` row, so a stored call joins to the run that made it (traceability
audit 2026-10-22: attribution used to rely on timestamps).
"""

from __future__ import annotations

from contextvars import ContextVar

CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("verifier_current_run_id", default=None)
