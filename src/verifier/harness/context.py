"""Context assembly: exactly what each model call gets to see.

Two hard rules:
- Decomposition and fidelity checking see ONLY the answer. A model that can see
  the sources tends to quietly "fix" the answer while restating it (1001 in the
  answer must stay 1001 even if the source says 1000).
- Source text is selected, never rewritten. If it does not fit, the request is
  rejected (InputTooLarge) rather than silently truncated — a truncated context
  would turn "not shown to the judge" into a false "not in the context".
"""

from __future__ import annotations

from functools import lru_cache

from verifier.harness.config import Settings
from verifier.harness.errors import InputTooLarge
from verifier.rag.source import Sentence, TextStore
from verifier.rag.tree import Node, NodeKind


def check_input_limits(answer: str, passages: list[str], settings: Settings) -> None:
    if len(answer) > settings.max_answer_chars:
        raise InputTooLarge("generated_answer", len(answer), settings.max_answer_chars)
    if len(passages) > settings.max_passages:
        raise InputTooLarge("context_passages", len(passages), settings.max_passages)
    for i, p in enumerate(passages):
        if len(p) > settings.max_passage_chars:
            raise InputTooLarge(f"context_passages[{i}]", len(p), settings.max_passage_chars)
    total = sum(len(p) for p in passages)
    if total > settings.max_total_context_chars:
        raise InputTooLarge("context_passages (total)", total, settings.max_total_context_chars)


def build_answer_token_view(store: TextStore) -> str:
    """The answer with every token numbered, for the proposal LLM:
    `0:SQLite 1:WAL 2:mode ...` — the model cites token indexes, code computes
    character offsets."""
    return " ".join(f"{t.index}:{t.text}" for t in store.answer_tokens)


ANSWER_STATE_TITLE = "The answer being decomposed (sources are deliberately not shown):"


def build_answer_state(store: TextStore, title: str = ANSWER_STATE_TITLE) -> str:
    """State for decomposition and fidelity checks: the text being decomposed
    alone, with sentence ids. `title` names that text (a source passage when
    the source tree runs the same decomposition)."""
    lines = [title]
    lines += [f"[{s.sentence_id}] {s.text}" for s in store.answer_sentences]
    return "\n".join(lines)


def build_source_state(store: TextStore, sentences: list[Sentence] | None = None) -> str:
    """State for verdict/relevance questions: source sentences, tagged with
    ids, verbatim. `sentences` narrows the set (selection, never rewriting)."""
    chosen = sentences if sentences is not None else store.all_source_sentences()
    if not chosen:
        return "The provided context is empty."
    lines = ["Source passages (the only ground truth; sentence ids in brackets):"]
    last_doc = None
    for s in chosen:
        if s.span.doc != last_doc:
            # Labels count from 1 because answers do ("passage 3" = third passage);
            # with 0-based labels the judge read correct citations as wrong and
            # off-by-one citations as right, and a translation note did not help.
            # Sentence ids and API citation indexes stay 0-based.
            lines.append(f"--- passage {store.passage_index_of(s.span.doc) + 1} ---")
            last_doc = s.span.doc
        lines.append(f"[{s.sentence_id}] {s.text}")
    return "\n".join(lines)


def describe_unit(store: TextStore, unit: Node) -> str:
    """How one claim unit appears inside a question: the self-contained text,
    plus the verbatim fragments it stands on."""
    anchor_text = " ... ".join(store.slice(sp) for sp in unit.anchor_spans)
    parts = [f'Claim: "{unit.display_text or anchor_text}"']
    if unit.display_text and unit.display_text != anchor_text:
        parts.append(f'Verbatim answer fragment: "{anchor_text}"')
    if unit.inherited_spans:
        inherited = " ... ".join(store.slice(sp) for sp in unit.inherited_spans)
        parts.append(f'Carried-over context from its sentence: "{inherited}"')
        # Scoping note for FACT fragments only: without it, the sentence's frame
        # leaks into the fragment's verdict (a contradicted sibling conjunct or a
        # contested causal link drags a verbatim-supported fragment down — the
        # 2026-10-21 ablation measured 1.00 -> 0.55). RELATION units are the
        # opposite case: their claim IS the sentence's structure, so they keep
        # the frame.
        if unit.kind is NodeKind.FACT:
            parts.append(
                "The context is only there to resolve references (pronouns, shared "
                "subjects). Judge ONLY the Claim itself."
            )
    return "\n".join(parts)


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget checks (Jev's tokenizer is not public;
    tiktoken plus headroom is close enough for a limit check)."""
    try:
        return len(_encoder().encode(text))
    except Exception:
        return max(1, len(text) // 4)
