"""Template-based reasoning: assembled from verified pieces (verdict,
calibrated probability, cited sentence ids) so the explanation step cannot
introduce new facts of its own. A generative rationale is a later comparison
arm, not the default."""

from __future__ import annotations

from verifier.judge.entail import UnitJudgment
from verifier.output.schemas import Verdict
from verifier.rag.tree import Node, NodeKind


def render_reasoning(unit: Node, judgment: UnitJudgment, *, context_empty: bool = False) -> str:
    if context_empty:
        return "The provided context is empty, so the claim cannot be verified against it."

    flags = set(judgment.flags)
    evidence = ", ".join(judgment.evidence_sentence_ids) or "no passage sentence"
    p = max(judgment.probabilities.values(), default=None)
    p_text = f" (calibrated p={p:.2f})" if p is not None else ""

    if any(f.startswith("ungrounded_part") for f in flags):
        return (
            "The unit as a whole matched the source, but part of its content (a condition, "
            "alternative or added detail) has no grounding in the provided context, so the claim "
            "is reported as UNSUPPORTED rather than letting the grounded part carry the rest."
        )
    # staged relation check: templates keyed to which stage decided, unless the
    # member aggregation overrode the staged verdict afterwards
    if not any(f.startswith("relation_merged_from_") for f in flags):
        if "contrast_truth_functional" in flags and judgment.verdict is Verdict.SUPPORTED:
            return (
                "A contrast ('but') asserts no mechanism of its own — the claim holds exactly "
                f"when both sides hold, and both do ({evidence})."
            )
        if any(f.startswith("staged_matched") for f in flags):
            return (
                f"The source itself states this connection ({evidence}), and both sides of the "
                "answer's link match the stated link's sides (verbatim, or a specific instance "
                f"or subset){p_text}."
            )
        if any(f.startswith("staged_order_inverted") for f in flags):
            return (
                f"The source states these two events in the OPPOSITE order ({evidence}), "
                f"so the claimed sequence is contradicted{p_text}."
            )
        if any(f.startswith("staged_order_unstated") for f in flags):
            return (
                f"A source sentence links these events ({evidence}), but it does not "
                "establish the order the answer asserts, so the claimed sequence cannot "
                "be verified from the context."
            )
        if any(f.startswith("staged_incompatible") for f in flags):
            return (
                f"The source states an incompatible link ({evidence}): the same mechanism with "
                f"the opposite direction or outcome{p_text}."
            )
        if any(f.startswith("staged_side_mismatch") for f in flags):
            return (
                "A source sentence states a link of this kind, but the answer transfers it to a "
                "different trigger or agent, or widens its scope, so the claimed connection "
                "cannot be verified from the context."
            )
        if any(f.startswith("staged_no_link") for f in flags):
            return (
                "No source sentence states the connection the answer asserts between these "
                "parts; the parts may each be true, but the link itself is unverifiable from "
                "the provided context."
            )
        if any(f.startswith("staged_flat_link") for f in flags):
            return (
                "The judge could not settle on any source sentence as stating this connection"
                f"{p_text}; per policy the uncertain link is reported as UNSUPPORTED."
            )
    if any(f.startswith("bridge_rescue") for f in flags):
        if judgment.verdict is Verdict.CONTRADICTED:
            return (
                f"The claim needs an identity hop: one cited sentence connects the claim's "
                f"subject in full to another entity, and the other states the OPPOSITE of the "
                f"asserted property for that entity ({evidence}) — combined, the source "
                f"contradicts the claim."
            )
        return (
            f"The claim needs an identity hop: one cited sentence connects the claim's subject "
            f"in full to another entity, and the other states the asserted property for that "
            f"entity ({evidence}) — combined, the source supports the claim."
        )
    if any(f.startswith("member_rescue") for f in flags):
        return (
            f"This claim is one side of a connection the source itself states ({evidence}): "
            "the stated link covers both sides, so the side is supported with the link "
            "sentence as evidence."
        )
    if any(f.startswith("c_filter_demoted") for f in flags):
        return (
            "The judge leaned CONTRADICTED, but the cited text states something different "
            "rather than the explicit opposite of this claim, so per the strict rule it is "
            "reported as UNSUPPORTED."
        )
    if any(f.startswith("low_confidence") for f in flags):
        return (
            f"The judge's probability distribution over verdicts was too flat to trust{p_text}; "
            "per policy an uncertain claim is reported as UNSUPPORTED rather than guessed."
        )
    if any(f.startswith("no_evidence_for_") for f in flags):
        return (
            "The judge leaned toward a definite verdict but no source sentence qualified as "
            "relevant evidence, so the claim is reported as UNSUPPORTED (no citable basis)."
        )
    if "parent_override" in flags:
        return (
            f"Each split part of this sentence looked supported, but the sentence as a whole "
            f"did not hold up against the source ({evidence}){p_text} — the split likely dropped "
            "a connection (same entity, cause, or scope), so the whole sentence is judged instead."
        )

    if judgment.verdict is Verdict.SUPPORTED:
        lead = "The cited source text entails the claim"
        if unit.kind is NodeKind.RELATION:
            lead = "The source supports both linked parts and the stated connection between them"
        return f"{lead} ({evidence}){p_text}."
    if judgment.verdict is Verdict.CONTRADICTED:
        lead = "The cited source text explicitly states the opposite of the claim"
        if unit.kind is NodeKind.RELATION:
            lead = "The source contradicts the claim or the stated connection between its parts"
        return f"{lead} ({evidence}){p_text}."
    return (
        f"The source passages neither establish nor contradict this claim{p_text}; "
        "it may be true elsewhere, but it cannot be verified from the provided context."
    )
