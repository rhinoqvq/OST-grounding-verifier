"""Candidate evidence selection.

The MVP keeps this deliberately simple: context passages in this task are
short, so by default *every* source sentence is a candidate and Jev answers a
relevance question per (unit, sentence) pair. A lexical prefilter only kicks in
when the context is large, to bound the number of questions. The prefilter
selects, it never rewrites — original sentences and offsets are untouched.
"""

from __future__ import annotations

import re
from collections import Counter

from verifier.rag.source import Sentence, TextStore
from verifier.rag.tree import Node

_WORD = re.compile(r"[a-z0-9][a-z0-9_.-]*")

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "by", "with", "as", "it", "its", "this", "that",
    "from", "at", "not", "no", "can", "will", "when", "all",
}


def _terms(text: str) -> Counter[str]:
    return Counter(w for w in _WORD.findall(text.lower()) if w not in _STOP)


def lexical_scores(unit_text: str, sentences: list[Sentence]) -> list[float]:
    """Simple overlap score, only used to *bound* the candidate set."""
    unit_terms = _terms(unit_text)
    scores = []
    for s in sentences:
        sent_terms = _terms(s.text)
        overlap = sum(min(unit_terms[t], sent_terms[t]) for t in unit_terms)
        scores.append(overlap / (1 + sum(unit_terms.values())))
    return scores


def top_by_overlap(text: str, sentences: list[Sentence], k: int) -> list[Sentence]:
    """The k sentences sharing the most terms with `text`, in source order."""
    scored = sorted(
        zip(lexical_scores(text, sentences), range(len(sentences)), sentences),
        key=lambda t: (-t[0], t[1]),
    )
    keep = sorted(scored[:k], key=lambda t: t[1])
    return [s for _, _, s in keep]


def candidate_sentences(
    store: TextStore,
    unit: Node,
    unit_text: str,
    *,
    max_candidates: int,
) -> list[Sentence]:
    """All source sentences when few; lexical top-k when the context is large."""
    sentences = store.all_source_sentences()
    if len(sentences) <= max_candidates:
        return sentences
    return top_by_overlap(unit_text, sentences, max_candidates)
