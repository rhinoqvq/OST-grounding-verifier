import pytest
from pydantic import ValidationError

from verifier.output.schemas import Citation, Claim, Verdict


def _citation() -> Citation:
    return Citation(index=0, quote="some quote")


def test_supported_requires_citations():
    with pytest.raises(ValidationError):
        Claim(claim_text="x", verdict=Verdict.SUPPORTED, reasoning="r", citations=[])


def test_contradicted_requires_citations():
    with pytest.raises(ValidationError):
        Claim(claim_text="x", verdict=Verdict.CONTRADICTED, reasoning="r", citations=[])


def test_unsupported_must_have_empty_citations():
    with pytest.raises(ValidationError):
        Claim(claim_text="x", verdict=Verdict.UNSUPPORTED, reasoning="r", citations=[_citation()])


def test_valid_claims_construct():
    Claim(claim_text="x", verdict=Verdict.SUPPORTED, reasoning="r", citations=[_citation()])
    Claim(claim_text="x", verdict=Verdict.UNSUPPORTED, reasoning="r", citations=[])


def test_citation_index_non_negative():
    with pytest.raises(ValidationError):
        Citation(index=-1, quote="q")


def test_citation_quote_non_empty():
    with pytest.raises(ValidationError):
        Citation(index=0, quote="")
