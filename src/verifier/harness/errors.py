"""Internal error types and how they surface.

"Could not verify from the context" is a verdict (UNSUPPORTED). These
exceptions are different: the *system* failed to produce a trustworthy result,
and that must never be dressed up as a successful response.
"""

from __future__ import annotations


class VerifierError(Exception):
    """Base for all internal failures."""


class InputTooLarge(VerifierError):
    """Request exceeds configured limits -> HTTP 413."""

    def __init__(self, what: str, size: int, limit: int):
        self.what, self.size, self.limit = what, size, limit
        super().__init__(f"{what} too large: {size} > limit {limit}")


class UpstreamError(VerifierError):
    """A model API failed (after retries, if retryable) -> HTTP 503/502."""

    def __init__(self, provider: str, kind: str, retryable: bool, cause: Exception | None = None):
        self.provider, self.kind, self.retryable, self.cause = provider, kind, retryable, cause
        super().__init__(f"upstream {provider} error ({kind}): {cause!r}")


class MalformedModelOutput(VerifierError):
    """Model returned output that failed validation even after one repair
    attempt -> HTTP 502 (the upstream produced garbage we cannot use)."""

    def __init__(self, provider: str, detail: str):
        self.provider, self.detail = provider, detail
        super().__init__(f"{provider} returned malformed output: {detail}")


class ExtractionFailed(VerifierError):
    """Decomposition could not cover the answer and the fallback also failed
    -> HTTP 500. Distinct from "the answer has no factual claims" (a 200)."""


class ReplayMiss(VerifierError):
    """Cache is in replay mode but a call was not recorded -> HTTP 500."""
