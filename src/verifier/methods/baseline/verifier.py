"""Method 1: the non-LLM baseline — sentence split, BM25, two lexical detectors.

Zero model calls. Each answer sentence is a claim; a hand-written BM25 ranks
the source sentences and the best one becomes the citation. An IDF-weighted
coverage gate sends low-overlap claims to UNSUPPORTED; on high overlap, two
conflict detectors (number mismatch, negation parity) send the claim to
CONTRADICTED; everything else is SUPPORTED.

Deliberately dumb and deliberately untuned (round default thresholds): this
method measures how far retrieval alone goes, so the gap to methods 2 and 3
isolates what actually requires semantic judgment. Citations are whole source
sentences sliced from the store, so they are verbatim by construction — the
exit contract still runs, as it does for every method.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from verifier.harness.context import check_input_limits
from verifier.harness.trace import Trace
from verifier.output.contract import DraftClaim, enforce_contract
from verifier.output.schemas import Citation, Verdict, VerifyRequest, VerifyResponse
from verifier.rag.source import Sentence, TextStore

if TYPE_CHECKING:
    from verifier.harness.pipeline import Dependencies

_WORD_RE = re.compile(r"[a-z0-9']+")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Function words excluded from coverage (they match everywhere and mean nothing).
_STOPWORDS = frozenset(
    "a an the and or but if then else when while of in on at to from by for with "
    "as is are was were be been being it its this that these those there here "
    "he she they them his her their we you your i us our".split()
)

_NEGATIONS = frozenset(
    "no not never none nothing cannot can't won't don't doesn't didn't isn't "
    "aren't wasn't weren't shouldn't couldn't wouldn't without nor neither".split()
)

_K1 = 1.5
_B = 0.75


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _numbers(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER_RE.findall(text)}


def _negation_parity(words: list[str]) -> int:
    return sum(1 for w in words if w in _NEGATIONS) % 2


@dataclass
class _Index:
    """BM25 index over the source sentences (pure code, no dependencies)."""

    sentences: list[Sentence]
    docs: list[list[str]]
    idf: dict[str, float]
    avg_len: float

    @classmethod
    def build(cls, sentences: list[Sentence]) -> "_Index":
        docs = [_words(s.text) for s in sentences]
        n = len(docs)
        df: dict[str, int] = {}
        for doc in docs:
            for w in set(doc):
                df[w] = df.get(w, 0) + 1
        idf = {w: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for w, d in df.items()}
        avg_len = sum(len(d) for d in docs) / n if n else 0.0
        return cls(sentences=sentences, docs=docs, idf=idf, avg_len=avg_len)

    def score(self, query: list[str], doc_i: int) -> float:
        doc = self.docs[doc_i]
        if not doc:
            return 0.0
        score = 0.0
        norm = _K1 * (1 - _B + _B * len(doc) / self.avg_len) if self.avg_len else _K1
        for w in set(query):
            tf = doc.count(w)
            if tf == 0:
                continue
            score += self.idf.get(w, 0.0) * tf * (_K1 + 1) / (tf + norm)
        return score

    def best(self, query: list[str]) -> int | None:
        if not self.sentences:
            return None
        scores = [self.score(query, i) for i in range(len(self.sentences))]
        top = max(range(len(scores)), key=lambda i: scores[i])
        return top if scores[top] > 0 else None

    def coverage(self, claim_words: list[str], doc_i: int) -> float:
        """IDF-weighted share of the claim's content words present in the sentence."""
        content = [w for w in claim_words if w not in _STOPWORDS]
        if not content:
            return 0.0
        present = set(self.docs[doc_i])
        # an unseen word gets the highest idf in the index: it matches nowhere
        ceiling = max(self.idf.values(), default=1.0)
        total = sum(self.idf.get(w, ceiling) for w in content)
        hit = sum(self.idf.get(w, ceiling) for w in content if w in present)
        return hit / total if total else 0.0


def _citation_for(store: TextStore, sentence: Sentence) -> Citation:
    return Citation(
        index=store.passage_index_of(sentence.span.doc),
        quote=sentence.text,
        start_char=sentence.span.start,
        end_char=sentence.span.end,
    )


def _judge_claim(index: _Index, store: TextStore, claim: Sentence, coverage_min: float) -> DraftClaim:
    claim_words = _words(claim.text)
    best_i = index.best(claim_words)
    if best_i is None:
        return DraftClaim(
            claim_text=claim.text,
            verdict=Verdict.UNSUPPORTED,
            reasoning="Lexical baseline: no source sentence shares content words with this claim.",
            citations=[],
        )
    evidence = index.sentences[best_i]
    coverage = index.coverage(claim_words, best_i)
    if coverage < coverage_min:
        return DraftClaim(
            claim_text=claim.text,
            verdict=Verdict.UNSUPPORTED,
            reasoning=(
                f"Lexical baseline: best source sentence {evidence.sentence_id} covers only "
                f"{coverage:.2f} of the claim's content words (threshold {coverage_min})."
            ),
            citations=[],
        )
    evidence_words = _words(evidence.text)
    claim_nums, evidence_nums = _numbers(claim.text), _numbers(evidence.text)
    if claim_nums and evidence_nums and not claim_nums <= evidence_nums:
        return DraftClaim(
            claim_text=claim.text,
            verdict=Verdict.CONTRADICTED,
            reasoning=(
                f"Lexical baseline: {evidence.sentence_id} covers {coverage:.2f} of the claim "
                f"but the numbers disagree ({sorted(claim_nums)} vs {sorted(evidence_nums)})."
            ),
            citations=[_citation_for(store, evidence)],
        )
    if _negation_parity(claim_words) != _negation_parity(evidence_words):
        return DraftClaim(
            claim_text=claim.text,
            verdict=Verdict.CONTRADICTED,
            reasoning=(
                f"Lexical baseline: {evidence.sentence_id} covers {coverage:.2f} of the claim "
                "but one side is negated and the other is not."
            ),
            citations=[_citation_for(store, evidence)],
        )
    return DraftClaim(
        claim_text=claim.text,
        verdict=Verdict.SUPPORTED,
        reasoning=(
            f"Lexical baseline: {evidence.sentence_id} covers {coverage:.2f} of the claim's "
            "content words with no number or negation mismatch."
        ),
        citations=[_citation_for(store, evidence)],
    )


async def run_baseline(
    request: VerifyRequest, deps: "Dependencies"
) -> tuple[VerifyResponse, Trace]:
    from verifier.harness.pipeline import _save  # same persistence as the other methods

    trace = Trace()
    try:
        response = _run(request, deps, trace)
        _save(deps, request, response, "ok", trace)
        return response, trace
    except Exception as exc:
        trace.note(f"failed: {type(exc).__name__}: {exc}")
        _save(deps, request, None, f"error:{type(exc).__name__}", trace)
        raise


def _run(request: VerifyRequest, deps: "Dependencies", trace: Trace) -> VerifyResponse:
    r = trace.start_round("0-prepare")
    check_input_limits(request.generated_answer, request.context_passages, deps.settings)
    store = TextStore.from_request(request.generated_answer, request.context_passages)
    trace.end_round(
        r,
        answer_sentences=len(store.answer_sentences),
        source_sentences=len(store.all_source_sentences()),
    )
    if not store.answer_sentences:
        trace.note("answer empty or whitespace: no claims")
        return VerifyResponse(claims=[])

    r = trace.start_round("1-baseline")
    index = _Index.build(store.all_source_sentences())
    coverage_min = deps.settings.baseline_coverage_min
    drafts = [_judge_claim(index, store, claim, coverage_min) for claim in store.answer_sentences]
    trace.end_round(r, verdicts={d.claim_text[:40]: d.verdict.value for d in drafts})

    r = trace.start_round("2-render")
    claims, report = enforce_contract(drafts, request.context_passages)
    trace.end_round(
        r,
        claims=len(claims),
        contract_violations=report.violations,
        dropped_citations=report.dropped_citations,
        downgraded=report.downgraded_claims,
    )
    return VerifyResponse(claims=claims)
