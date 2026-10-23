"""Locate flagged claims in a RAGTruth response, for character-level scoring.

RAGTruth labels hallucinated spans (character offsets into the response); it has no
claim-level gold. A response is predicted hallucinated when any final claim is CONTRADICTED
or UNSUPPORTED. For character-level F1, each flagged claim is located in the response
(exact, then piecewise verbatim pieces); claims that cannot be located are counted, never
given a guessed span. Used by paired.py.
"""

from __future__ import annotations

import re


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def flagged_claims(record: dict, strict: bool) -> list[dict]:
    """Final API claims with a flagged verdict — the one scoring object every
    method produces (units exist only on the optimized path, so unit-based scoring would
    count the single-call methods as never flagging)."""
    bad = {"CONTRADICTED", "UNSUPPORTED"} if strict else {"CONTRADICTED"}
    return [c for c in record.get("claims", []) if c["verdict"] in bad]


def locate(claim_text: str, answer: str) -> tuple[int, int] | None:
    """Char range of a claim in its answer: exact, then stripped, then
    case-insensitive. None = not locatable (assembled or paraphrased claims);
    the caller falls back to piecewise_locate, then reports the remainder
    instead of guessing a span."""
    for needle in (claim_text, claim_text.strip()):
        i = answer.find(needle)
        if i >= 0:
            return (i, i + len(needle))
    i = answer.lower().find(claim_text.strip().lower())
    if i >= 0:
        return (i, i + len(claim_text.strip()))
    return None


_MIN_PIECE_TOKENS = 3


def piecewise_locate(claim: str, answer: str) -> list[tuple[int, int]] | None:
    """Recomposed claims (coordination cuts rebuild display text from a shared
    lead + one branch) are verbatim in the answer except for a gap. Split the
    claim into contiguous verbatim pieces of >= 3 tokens, in order; return the
    pieces' spans (union, never the gap). None if any token is left over, so
    reworded claims stay unlocatable rather than getting a guessed span."""
    tokens = claim.split()
    if len(tokens) < _MIN_PIECE_TOKENS:
        return None
    spans: list[tuple[int, int]] = []
    i = 0
    pos = 0
    while i < len(tokens):
        best = None
        for n in range(len(tokens) - i, _MIN_PIECE_TOKENS - 1, -1):
            pattern = r"\s+".join(re.escape(t) for t in tokens[i : i + n])
            m = re.compile(pattern, re.IGNORECASE).search(answer, pos)
            if m:
                best = (n, m.start(), m.end())
                break
        if best is None:
            return None
        n, start, end = best
        spans.append((start, end))
        pos = end
        i += n
    return spans


def flagged_claim_spans(record: dict, answer: str, strict: bool) -> tuple[list[list[tuple[int, int]]], int]:
    """(per-claim span groups of flagged claims, unlocatable flagged claims).
    A located claim contributes one group: a single span, or the verbatim
    pieces from piecewise_locate for recomposed display text."""
    groups: list[list[tuple[int, int]]] = []
    unlocatable = 0
    for c in flagged_claims(record, strict):
        span = locate(c["claim_text"], answer)
        if span is not None:
            groups.append([span])
            continue
        pieces = piecewise_locate(c["claim_text"], answer)
        if pieces:
            groups.append(pieces)
        else:
            unlocatable += 1
    return groups, unlocatable


def char_union(spans: list[tuple[int, int]]) -> set[int]:
    chars: set[int] = set()
    for s, e in spans:
        chars.update(range(s, e))
    return chars
