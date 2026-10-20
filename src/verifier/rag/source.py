"""Immutable source-of-truth text store.

Everything downstream works with document ids, sentence ids and character
offsets into the *original* strings. All quoted text is sliced from here, so a
citation can never drift from the source.

Document ids: "answer", "context:0", "context:1", ...
Sentence ids: "A.S0" for the answer, "P0.S1" for context passage 0 sentence 1.
Offsets are Python character indexes, half-open [start, end).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Span:
    doc: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"bad span [{self.start}, {self.end})")


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    span: Span
    text: str


@dataclass(frozen=True)
class Token:
    index: int
    span: Span
    text: str


# Spans that must never contain a sentence boundary: version numbers,
# dotted file names / paths, inline code, formulas with '=', ellipses.
_PROTECT_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)+\b"),                      # 3.7.0, 1.75
    re.compile(r"`[^`\n]+`"),                              # inline code
    re.compile(r"\[[^\]\n]{1,20}\]:?[a-z]*"),              # [h]:mm style formats
    re.compile(r"=\s*[A-Z]{2,}\([^)\n]*\)"),               # =WORKDAY(...)
    re.compile(r"\b[\w-]+\.(?:py|md|txt|json|jsonl|wal|shm|db|exe|dll)\b", re.I),
    re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|Dr|Mr|Ms|Mrs)\.", re.I),
    re.compile(r"\.\.+"),                                  # ".." / "..." artifacts
]

# A numbered-step marker: "1. " / "2) " at a line start or after spaces.
# Trade-off: a real sentence ending in a bare small number ("version 2. It...")
# gets merged with the next one; in technical how-to text, inline step lists
# are far more common than that, so we protect the marker.
_STEP_MARKER = re.compile(r"(?<![\w.,])\d{1,2}[.)]\s")


def protect_special_spans(text: str) -> list[tuple[int, int]]:
    """Return [start, end) ranges inside which sentence boundaries are forbidden."""
    ranges: list[tuple[int, int]] = []
    for pat in _PROTECT_PATTERNS:
        ranges.extend(m.span() for m in pat.finditer(text))
    # A step marker protects itself plus the following char, so "1." never
    # becomes a sentence of its own.
    ranges.extend((m.start(), m.end() + 1) for m in _STEP_MARKER.finditer(text))
    return sorted(ranges)


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def _in_protected(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start < pos < end for start, end in ranges)


def segment(doc_id: str, text: str) -> list[Span]:
    """Split into sentence spans (char offsets into the original text).

    spaCy's rule-based sentencizer proposes boundaries; boundaries that fall
    inside a protected span are merged away. Whitespace-only pieces are
    dropped; leading/trailing whitespace is trimmed from each span.
    """
    if not text.strip():
        return []
    protected = protect_special_spans(text)
    doc = _nlp()(text)
    # Candidate boundaries: sentence start offsets except position 0.
    bounds = [s.start_char for s in doc.sents][1:]
    bounds = [b for b in bounds if not _in_protected(b, protected)]
    starts = [0, *bounds]
    ends = [*bounds, len(text)]
    spans: list[Span] = []
    for raw_start, raw_end in zip(starts, ends):
        chunk = text[raw_start:raw_end]
        if not chunk.strip():
            continue
        start = raw_start + (len(chunk) - len(chunk.lstrip()))
        end = raw_end - (len(chunk) - len(chunk.rstrip()))
        spans.append(Span(doc_id, start, end))
    return spans


_TOKEN_RE = re.compile(r"\S+")


def tokenize(doc_id: str, text: str) -> list[Token]:
    """Whitespace tokens with char offsets, numbered from 0. Used to let the
    LLM reference positions by token index instead of counting characters."""
    return [
        Token(i, Span(doc_id, m.start(), m.end()), m.group())
        for i, m in enumerate(_TOKEN_RE.finditer(text))
    ]


@dataclass
class TextStore:
    """All original texts for one request. Never mutated after construction."""

    answer: str
    passages: list[str]
    answer_sentences: list[Sentence] = field(init=False)
    passage_sentences: list[list[Sentence]] = field(init=False)
    answer_tokens: list[Token] = field(init=False)

    ANSWER_DOC = "answer"

    def __post_init__(self) -> None:
        self.answer_sentences = [
            Sentence(f"A.S{i}", span, self._text_of(span))
            for i, span in enumerate(segment(self.ANSWER_DOC, self.answer))
        ]
        self.passage_sentences = []
        for p_idx, passage in enumerate(self.passages):
            doc = f"context:{p_idx}"
            self.passage_sentences.append(
                [
                    Sentence(f"P{p_idx}.S{s_idx}", span, self._text_of(span))
                    for s_idx, span in enumerate(segment(doc, passage))
                ]
            )
        self.answer_tokens = tokenize(self.ANSWER_DOC, self.answer)

    @classmethod
    def from_request(cls, answer: str, passages: list[str]) -> "TextStore":
        return cls(answer=answer, passages=list(passages))

    # -- raw text access -------------------------------------------------
    def _doc_text(self, doc: str) -> str:
        if doc == self.ANSWER_DOC:
            return self.answer
        if doc.startswith("context:"):
            idx = int(doc.split(":", 1)[1])
            if not 0 <= idx < len(self.passages):
                raise IndexError(f"unknown passage {doc}")
            return self.passages[idx]
        raise KeyError(f"unknown document {doc!r}")

    def _text_of(self, span: Span) -> str:
        return self._doc_text(span.doc)[span.start : span.end]

    def slice(self, span: Span) -> str:
        """The only way any module obtains quoted text: verbatim by offsets."""
        text = self._doc_text(span.doc)
        if span.end > len(text):
            raise ValueError(f"span {span} exceeds document length {len(text)}")
        return text[span.start : span.end]

    # -- lookups ----------------------------------------------------------
    def all_source_sentences(self) -> list[Sentence]:
        return [s for per_passage in self.passage_sentences for s in per_passage]

    def sentence_by_id(self, sentence_id: str) -> Sentence:
        pool = (
            self.answer_sentences
            if sentence_id.startswith("A.")
            else self.all_source_sentences()
        )
        for s in pool:
            if s.sentence_id == sentence_id:
                return s
        raise KeyError(f"unknown sentence {sentence_id!r}")

    def answer_sentence_containing(self, span: Span) -> Sentence | None:
        for s in self.answer_sentences:
            if span.start >= s.span.start and span.end <= s.span.end:
                return s
        return None

    def token_range_to_span(self, first: int, last: int) -> Span:
        """Inclusive token range -> char span in the answer."""
        if not 0 <= first <= last < len(self.answer_tokens):
            raise IndexError(f"token range [{first}, {last}] out of bounds")
        return Span(
            self.ANSWER_DOC,
            self.answer_tokens[first].span.start,
            self.answer_tokens[last].span.end,
        )

    def passage_index_of(self, doc: str) -> int:
        if not doc.startswith("context:"):
            raise ValueError(f"{doc!r} is not a context document")
        return int(doc.split(":", 1)[1])
