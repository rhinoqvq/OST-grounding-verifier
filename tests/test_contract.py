from verifier.output.contract import DraftClaim, enforce_contract
from verifier.output.schemas import Citation, Verdict

PASSAGES = ["The cache is persistent only when storage is enabled.", "Second passage text."]


def _draft(verdict: Verdict, citations: list[Citation]) -> DraftClaim:
    return DraftClaim(claim_text="c", verdict=verdict, reasoning="r", citations=citations)


def test_valid_citation_passes():
    quote = "only when storage is enabled"
    claims, report = enforce_contract(
        [_draft(Verdict.SUPPORTED, [Citation(index=0, quote=quote)])], PASSAGES
    )
    assert claims[0].verdict is Verdict.SUPPORTED
    assert claims[0].citations[0].start_char == PASSAGES[0].index(quote)
    assert report.downgraded_claims == 0


def test_out_of_bounds_index_drops_citation_and_downgrades():
    claims, report = enforce_contract(
        [_draft(Verdict.SUPPORTED, [Citation(index=5, quote="whatever")])], PASSAGES
    )
    assert claims[0].verdict is Verdict.UNSUPPORTED
    assert claims[0].citations == []
    assert report.dropped_citations == 1
    assert report.downgraded_claims == 1
    assert "downgraded" in claims[0].reasoning


def test_non_verbatim_quote_is_dropped():
    claims, report = enforce_contract(
        [_draft(Verdict.CONTRADICTED, [Citation(index=0, quote="cache is always persistent")])],
        PASSAGES,
    )
    assert claims[0].verdict is Verdict.UNSUPPORTED
    assert report.dropped_citations == 1


def test_unsupported_citations_are_cleared():
    claims, report = enforce_contract(
        [_draft(Verdict.UNSUPPORTED, [Citation(index=0, quote="The cache")])], PASSAGES
    )
    assert claims[0].citations == []
    assert any("cleared" in v for v in report.violations)


def test_wrong_offsets_are_reanchored():
    quote = "Second passage"
    claims, report = enforce_contract(
        [_draft(Verdict.SUPPORTED, [Citation(index=1, quote=quote, start_char=3, end_char=9)])],
        PASSAGES,
    )
    c = claims[0].citations[0]
    assert PASSAGES[1][c.start_char : c.end_char] == quote
    assert any("re-anchored" in v for v in report.violations)


def test_one_bad_citation_does_not_sink_a_good_one():
    good = Citation(index=0, quote="The cache is persistent")
    bad = Citation(index=0, quote="not in passage at all")
    claims, report = enforce_contract([_draft(Verdict.SUPPORTED, [bad, good])], PASSAGES)
    assert claims[0].verdict is Verdict.SUPPORTED
    assert len(claims[0].citations) == 1
    assert report.dropped_citations == 1
