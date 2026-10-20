"""Public request/response types, exactly as the assignment specifies.

The hard invariants live here as Pydantic validators, so an invalid response
cannot even be constructed:
- SUPPORTED / CONTRADICTED claims must have at least one citation.
- UNSUPPORTED claims must have an empty citation list.
Quote verbatim-ness and index bounds need the original passages, so they are
enforced separately in `verifier.output.contract`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class VerifyRequest(BaseModel):
    context_passages: list[str]
    generated_answer: str


class Citation(BaseModel):
    index: int = Field(ge=0, description="0-based index into context_passages")
    quote: str = Field(min_length=1, description="verbatim substring of context_passages[index]")
    start_char: int | None = None
    end_char: int | None = None


class Claim(BaseModel):
    claim_text: str
    verdict: Verdict
    reasoning: str
    citations: list[Citation]

    @model_validator(mode="after")
    def _citations_match_verdict(self) -> "Claim":
        if self.verdict is Verdict.UNSUPPORTED and self.citations:
            raise ValueError("UNSUPPORTED claims must have an empty citations list")
        if self.verdict is not Verdict.UNSUPPORTED and not self.citations:
            raise ValueError(f"{self.verdict.value} claims require at least one citation")
        return self


class VerifyResponse(BaseModel):
    claims: list[Claim]
