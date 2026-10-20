"""Retry policy for upstream calls: transient failures (timeout, rate limit,
5xx) are retried with exponential backoff; everything else fails fast."""

from __future__ import annotations

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from verifier.harness.errors import UpstreamError


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, UpstreamError):
        return exc.retryable
    return False


def classify_upstream_error(exc: Exception, provider: str) -> UpstreamError:
    """Map SDK-specific exceptions onto our single UpstreamError type."""
    name = type(exc).__name__.lower()
    retryable_markers = ("timeout", "ratelimit", "connection", "internalserver", "apistatus")
    fatal_markers = ("authentication", "permission", "badrequest", "notfound", "unprocessable")
    kind = "other"
    retryable = False
    if any(m in name for m in fatal_markers):
        kind = "fatal"
    elif any(m in name for m in retryable_markers):
        kind, retryable = "transient", True
    else:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if isinstance(status, int):
            retryable = status == 429 or status >= 500
            kind = "transient" if retryable else "fatal"
    return UpstreamError(provider=provider, kind=kind, retryable=retryable, cause=exc)


def retrying(attempts: int) -> AsyncRetrying:
    return AsyncRetrying(
        retry=retry_if_exception(is_retryable),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
