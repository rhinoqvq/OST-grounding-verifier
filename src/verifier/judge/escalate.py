"""Second-opinion rescue for flat UNSUPPORTED facts (the flat-U battle).

The calibrated judge leaves ~6% of units flat; measured, most flat errors sit
on content the source actually supports. Same-judge stacking is correlated
(n02: four guards sharing one judge failed together), so the rescue asks a
DIFFERENT judge — the generative model — and accepts its opinion only with a
constructive witness: the model must name the supporting sentence and copy a
verbatim quote from it, and code verifies the quote inside that sentence.
No witness, no upgrade; only flat U -> S, never C; failure leaves the flat
verdict untouched. K2 discipline throughout (see the proof document, 7.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from verifier.harness.context import describe_unit
from verifier.judge.entail import UnitJudgment, _normalized_tokens
from verifier.output.schemas import Verdict
from verifier.rag.source import TextStore
from verifier.rag.tree import Node

if TYPE_CHECKING:
    from verifier.models.openai_llm import StructuredLLM

_MIN_QUOTE_TOKENS = 3

_SYSTEM = """You are a second-opinion fact verifier. A cautious primary judge
could not decide whether the claim below is supported by the sources.

Decide ONLY between:
- supported: some source sentence states the claim's content (paraphrase counts).
- not_established: the sources do not state it, or you are not sure.

If supported, you MUST name the single most decisive source sentence by its id
and copy a quote from it VERBATIM, character for character. If you cannot quote
a supporting sentence verbatim, answer not_established."""


class EscalationAnswer(BaseModel):
    verdict: Literal["supported", "not_established"]
    sentence_id: str | None = None
    quote: str | None = None


def escalation_messages(store: TextStore, unit: Node) -> list[dict[str, str]]:
    sources = "\n".join(f"[{s.sentence_id}] {s.text}" for s in store.all_source_sentences())
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Sources:\n{sources}\n\n{describe_unit(store, unit)}"},
    ]


async def escalate_flat(
    llm: "StructuredLLM", store: TextStore, unit: Node, judgment: UnitJudgment
) -> bool:
    """Returns True iff the flat U was upgraded to S with a verified witness."""
    try:
        answer = await llm.generate_structured(escalation_messages(store, unit), EscalationAnswer)
    except Exception:  # noqa: BLE001 — a rescue must never break the verdict
        judgment.flags.append("llm_rescue_error")
        return False
    if answer.verdict != "supported" or not answer.sentence_id or not answer.quote:
        judgment.flags.append("llm_rescue_declined")
        return False
    try:
        sentence = store.sentence_by_id(answer.sentence_id)
    except KeyError:
        judgment.flags.append("llm_rescue_bad_sid")
        return False
    quote_norm = _normalized_tokens(answer.quote)
    if len(quote_norm.split()) < _MIN_QUOTE_TOKENS or quote_norm not in _normalized_tokens(sentence.text):
        # the witness is the whole point: no verbatim quote, no upgrade
        judgment.flags.append(f"llm_rescue_no_witness({answer.sentence_id})")
        return False
    judgment.verdict = Verdict.SUPPORTED
    judgment.evidence_sentence_ids = [answer.sentence_id]
    judgment.flags.append(f"llm_rescue({answer.sentence_id})")
    return True
