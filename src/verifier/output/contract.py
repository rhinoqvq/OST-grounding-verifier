"""The exit contract: the last line of defense before a response leaves.

Every response — whatever produced it — passes through here:
- every citation index must be in bounds;
- every quote must be a verbatim substring of context_passages[index]
  (offsets are re-derived; a quote that exists elsewhere is re-anchored);
- SUPPORTED/CONTRADICTED with no surviving citation is downgraded to
  UNSUPPORTED, with the downgrade recorded;
- UNSUPPORTED keeps an empty citation list.

Violations are returned alongside the fixed claims; the pipeline stores them in
the trace. "How many raw model citations the contract had to remove" is itself
an evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verifier.output.schemas import Citation, Claim, Verdict


@dataclass
class ContractReport:
    violations: list[str] = field(default_factory=list)
    dropped_citations: int = 0
    downgraded_claims: int = 0


@dataclass
class DraftClaim:
    """A claim before contract enforcement (plain data, no invariants yet)."""

    claim_text: str
    verdict: Verdict
    reasoning: str
    citations: list[Citation]


def _check_citation(citation: Citation, passages: list[str], where: str, report: ContractReport) -> Citation | None:
    if not 0 <= citation.index < len(passages):
        report.violations.append(f"{where}: index {citation.index} out of bounds")
        return None
    passage = passages[citation.index]
    start, end = citation.start_char, citation.end_char
    if (
        start is not None
        and end is not None
        and 0 <= start <= end <= len(passage)
        and passage[start:end] == citation.quote
    ):
        return citation
    # offsets wrong or missing: re-anchor by searching for the quote verbatim
    pos = passage.find(citation.quote)
    if pos < 0:
        report.violations.append(f"{where}: quote is not a verbatim substring of passage {citation.index}")
        return None
    report.violations.append(f"{where}: offsets re-anchored to {pos}")
    return Citation(index=citation.index, quote=citation.quote, start_char=pos, end_char=pos + len(citation.quote))


def enforce_contract(drafts: list[DraftClaim], passages: list[str]) -> tuple[list[Claim], ContractReport]:
    report = ContractReport()
    claims: list[Claim] = []
    for i, draft in enumerate(drafts):
        where = f"claim[{i}]"
        verdict = draft.verdict
        reasoning = draft.reasoning
        kept: list[Citation] = []
        for citation in draft.citations:
            checked = _check_citation(citation, passages, where, report)
            if checked is not None:
                kept.append(checked)
            else:
                report.dropped_citations += 1
        if verdict is Verdict.UNSUPPORTED and kept:
            report.violations.append(f"{where}: UNSUPPORTED carried citations; cleared")
            kept = []
        if verdict is not Verdict.UNSUPPORTED and not kept:
            report.violations.append(
                f"{where}: {verdict.value} had no valid citation; downgraded to UNSUPPORTED"
            )
            report.downgraded_claims += 1
            verdict = Verdict.UNSUPPORTED
            reasoning = (
                f"{reasoning} [contract: original verdict {draft.verdict.value} had no verifiable "
                "citation, so it was downgraded to UNSUPPORTED]"
            )
            kept = []
        claims.append(Claim(claim_text=draft.claim_text, verdict=verdict, reasoning=reasoning, citations=kept))
    return claims, report
