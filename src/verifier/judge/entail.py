"""Round 3: verdicts and evidence, in one batched Jev round.

Per unit: one 3-way choice (supported / contradicted / not_in_context), per the
agreed rule R1: CONTRADICTED only when the source explicitly states something
that cannot be true together with the claim; weaker-than-claimed or merely
implied counts as not_in_context.

Per (unit, candidate sentence): one yes/no "is this sentence relevant evidence"
— relevance, not support, so evidence that only works combined across passages
is still collected. Both question sets are independent given the same state, so
they ship in the same round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from verifier.harness.context import describe_unit
from verifier.models.decider import Answer, ChoiceQuestion, Question, YesNoQuestion
from verifier.output.schemas import Verdict
from verifier.rag.source import Sentence, TextStore
from verifier.rag.tree import Node, RelationType

VERDICT_OPTIONS = {
    "supported": (
        "Every part of the claim (subject, numbers, conditions, negation) follows from the "
        "source sentences alone. Combining several source sentences is fine. Paraphrase is fine."
    ),
    "contradicted": (
        "The source explicitly states something that cannot be true together with the claim "
        "(opposite assertion, different number for the same thing, inverted condition)."
    ),
    "not_in_context": (
        "The source neither establishes nor contradicts the claim: it is not mentioned, is only "
        "implied, or the source says something weaker (e.g. 'most' vs the claim's 'all'). "
        "True-in-the-real-world but absent from the source belongs here."
    ),
}


def verdict_question(store: TextStore, unit: Node) -> ChoiceQuestion:
    return ChoiceQuestion(
        question_id=f"v::{unit.node_id}",
        instructions=(
            "Judge this claim strictly against the source passages in the state — "
            f"never against outside knowledge.\n{describe_unit(store, unit)}"
        ),
        options=VERDICT_OPTIONS,
    )


def relevance_question(store: TextStore, unit: Node, sentence: Sentence) -> YesNoQuestion:
    return YesNoQuestion(
        question_id=f"r::{unit.node_id}::{sentence.sentence_id}",
        instructions=(
            f"Source sentence [{sentence.sentence_id}] reads: \"{sentence.text}\"\n"
            "Would a careful fact-checker quote this sentence as evidence when explaining why the "
            "claim below is supported or contradicted? Merely being about the same topic is not "
            f"evidence; the sentence must bear on whether the claim holds.\n{describe_unit(store, unit)}"
        ),
    )


def grounding_question(store: TextStore, unit: Node, part_index: int) -> YesNoQuestion:
    """Diagnostic part of a kept-whole unit (conditional/disjunction/reported):
    asks for *presence in the source*, not truth. Grounding composes across any
    structure, truth does not — this is what catches a fabricated disjunct or
    condition inside a unit whose overall verdict looks SUPPORTED."""
    part_text = store.slice(unit.diagnostic_spans[part_index])
    return YesNoQuestion(
        question_id=f"g::{unit.node_id}::{part_index}",
        instructions=(
            f"The answer contains this unit (kept whole because splitting would change its meaning):\n"
            f"\"{unit.display_text}\"\n"
            f"Question: does the source contain the following piece of that unit's content — stated "
            f"anywhere, including inside a conditional, an alternative/option, or reported speech?\n"
            f"Content piece: \"{part_text}\""
        ),
    )


def judge_questions(
    store: TextStore,
    units: list[Node],
    candidates: dict[str, list[Sentence]],
) -> list[Question]:
    questions: list[Question] = []
    for unit in units:
        questions.append(verdict_question(store, unit))
        for sentence in candidates.get(unit.node_id, []):
            questions.append(relevance_question(store, unit, sentence))
        for i in range(len(unit.diagnostic_spans)):
            questions.append(grounding_question(store, unit, i))
    return questions


# --- staged relation check -------------------------------------------------
# A relation unit ("A —so→ B") packs a multi-hop judgment into one question:
# find the stated link, match the antecedent, match the consequent. One bounded
# question cannot carry that chain (measured: instantiation judged C, pure
# narrowing judged U), so the check is decomposed into single-hop questions
# that code composes under the settled rule:
#   SUPPORTED     verbatim link, general-rule instantiation, pure narrowing
#   UNSUPPORTED   sibling-trigger transfer, novel link, agent shift, widening
#   CONTRADICTED  only when the source states an incompatible link

_LINK_KIND = {
    "cause": "causal link (one side causes or explains the other)",
    "contrast": "contrast (the sides are set against each other)",
    "sequence": "temporal order (one side happens before the other)",
    "condition": "conditional link (one side holds under the other)",
}


def relation_link_question(store: TextStore, unit: Node) -> ChoiceQuestion:
    """Stage 1: which source sentence states a link of this kind at all —
    in ANY direction or polarity (an opposite link still counts here; stage 2
    sorts compatibility out)."""
    kind = _LINK_KIND.get(unit.relation_type.value, "link")
    options = {
        s.sentence_id: f'"{s.text}"' for s in store.all_source_sentences()
    }
    options["no_such_link"] = (
        "No source sentence states a link of this kind between contents like these; "
        "sentences that merely mention the parts separately do not count."
    )
    return ChoiceQuestion(
        question_id=f"rel1::{unit.node_id}",
        instructions=(
            f"{describe_unit(store, unit)}\n"
            f"Which source sentence ITSELF states a {kind} between contents of this kind — "
            "in any direction or polarity? Pick the sentence that states the connection "
            "(because / so that / but / then ...), not one that merely mentions the parts."
        ),
        options=options,
    )


SIDE_OPTIONS = {
    "matches": (
        "In the claimed link this side plays the same role and refers to the same thing as the "
        "source link's corresponding side — or to a specific instance/subset of it (a generic "
        "statement covers its instances)."
    ),
    "opposite": (
        "Same role, same aspect, but with the OPPOSITE or negated value (fast vs slow, enabled "
        "vs disabled, inverted direction)."
    ),
    "different": (
        "It refers to a DIFFERENT thing: another trigger, agent or mechanism that the source "
        "keeps separate, a generalization, or a wider scope than the source side covers."
    ),
}


# A sequence is an ASYMMETRIC relation: r(x,y) and r(y,x) differ while the
# side checks alone cannot tell them apart (rel1 finds the link in any
# direction, rel2/rel3 match content per side — both are invariant under
# swapping the sides; measured: an inverted order composed S at this layer,
# mathprobe m09). The order itself gets its own single-hop question, and the
# claim's own order is fixed by the cut option's wording (sequence_link means
# LEFT happens first), so code can compare the two.
ORDER_OPTIONS = {
    "left_first": (
        "The source states that the FIRST side happens before the THEN side — "
        "the same order the claim asserts."
    ),
    "right_first": (
        "The source states that the THEN side happens before the FIRST side — "
        "the opposite of the order the claim asserts."
    ),
    "no_order": "The source sentence does not establish which of the two happens first.",
}


def sequence_order_question(
    unit: Node, link_sentence: Sentence, left_text: str, right_text: str
) -> ChoiceQuestion:
    return ChoiceQuestion(
        question_id=f"rel4::{unit.node_id}",
        instructions=(
            f'Source sentence [{link_sentence.sentence_id}] states a temporal order: '
            f'"{link_sentence.text}"\n'
            f'The claim asserts this order — FIRST: "{left_text}", THEN: "{right_text}".\n'
            "According to the source sentence alone, which of the claim's two sides "
            "happens first?"
        ),
        options=ORDER_OPTIONS,
    )


def relation_stage2_questions(
    store: TextStore, unit: Node, link_sentence: Sentence, left_text: str, right_text: str
) -> list[Question]:
    """Stage 2 for a found link sentence: classify each side against the
    stated link's corresponding side. Polarity is folded into the options —
    'opposite' (same aspect, inverted value) is what licenses CONTRADICTED,
    while 'different' (sibling trigger, widening) can only mean UNSUPPORTED.
    A SEQUENCE relation gets one extra question for the stated order, because
    the side checks are blind to direction."""
    header = (
        f'Source sentence [{link_sentence.sentence_id}] states a link: "{link_sentence.text}"\n'
        f"Claimed link from the answer: {unit.display_text}\n"
    )
    tail = "Compare it with the corresponding side of the source link and classify."
    questions: list[Question] = [
        ChoiceQuestion(
            question_id=f"rel2::{unit.node_id}",
            instructions=f'{header}Side under check: "{left_text}"\n{tail}',
            options=SIDE_OPTIONS,
        ),
        ChoiceQuestion(
            question_id=f"rel3::{unit.node_id}",
            instructions=f'{header}Side under check: "{right_text}"\n{tail}',
            options=SIDE_OPTIONS,
        ),
    ]
    if unit.relation_type is RelationType.SEQUENCE:
        questions.append(sequence_order_question(unit, link_sentence, left_text, right_text))
    return questions


# --- opposition filter on confident CONTRADICTED ------------------------------
# R1 (Def. 2's C clause): CONTRADICTED only when the source explicitly states
# the opposite of the claim itself. Measured: C outputs were 25-29% correct,
# the dominant error being "a DIFFERENT entity/aspect read as exclusive"
# (D does-not-establish phi conflated with D establishes not-phi). The filter
# is a pure K1 operator (C -> U, nothing else): the S set and the flagged set
# are untouched by construction, so hallucination coverage cannot drop; the
# only cost dimension is severity labeling of true C's, guarded by reusing the
# staged check's proven opposite/different distinction at the fact level.

OPPOSITION_OPTIONS = {
    "opposite": (
        "At least one cited sentence RULES THE CLAIM OUT: it states the same aspect "
        "with an inverted or incompatible value, or it restricts the claim's subject "
        "in a way that excludes the claim ('only does X', 'never', 'nothing but') — "
        "the sentence and the claim cannot be true together."
    ),
    "different": (
        "The cited sentences state something about a DIFFERENT entity, aspect or scope "
        "(for example, another thing did or has it) WITHOUT ruling the claim out: both "
        "could still be true together."
    ),
    "matches": "The cited sentences actually agree with or support the claim.",
}


def opposition_question(store: TextStore, unit: Node, judgment: "UnitJudgment") -> ChoiceQuestion:
    quoted = "\n".join(
        f'[{sid}] "{store.sentence_by_id(sid).text}"'
        for sid in judgment.evidence_sentence_ids[:4]
    )
    return ChoiceQuestion(
        question_id=f"c1::{unit.node_id}",
        instructions=(
            "This claim was judged CONTRADICTED, citing the source sentence(s):\n"
            f"{quoted}\n{describe_unit(store, unit)}\n"
            "A claim counts as contradicted only if a cited sentence states the opposite "
            "of the claim ITSELF. Classify what the cited sentence(s) state relative to "
            "the claim."
        ),
        options=OPPOSITION_OPTIONS,
    )


def decide_opposition_filter(
    judgment: "UnitJudgment", answers: dict[str, Answer], *, min_prob: float
) -> bool:
    """Demote a CONTRADICTED verdict to UNSUPPORTED unless the opposition is
    confirmed as explicit ('opposite' at or above the bar). Returns True when
    the verdict was demoted. Evidence is kept for traceability."""
    answer = answers.get(f"c1::{judgment.node_id}")
    probs = answer.probabilities if answer and answer.probabilities else {}
    top = max(probs, key=probs.__getitem__) if probs else "different"
    p = probs.get(top, 0.0)
    judgment.probabilities[f"opposition_{top}"] = p
    if top == "opposite" and p >= min_prob:
        return False
    judgment.verdict = Verdict.UNSUPPORTED
    judgment.flags.append(f"c_filter_demoted({top},p={p:.2f})")
    return True


# --- staged bridge check (two-hop rescue) ------------------------------------
# A claim like "Production accepts V2" against "Production routes exclusively
# to Azure" + "Azure accepts V2" needs an identity hop that one bounded verdict
# question cannot carry (measured: flat ~0.47). When the ordinary verdict comes
# back flat, the hop is decomposed: find the bridge sentence, check it licenses
# a full transfer, find the property sentence, compare the asserted property.
# It can only upgrade a flat UNSUPPORTED to a composed S or C — a confident
# ordinary verdict is never touched.


def bridge_find_questions(store: TextStore, unit: Node) -> list[Question]:
    """Round A: locate the bridge sentence and the property sentence."""
    sentence_options = {s.sentence_id: f'"{s.text}"' for s in store.all_source_sentences()}
    return [
        ChoiceQuestion(
            question_id=f"br1::{unit.node_id}",
            instructions=(
                f"{describe_unit(store, unit)}\n"
                "Which source sentence states that the claim's SUBJECT is connected to or "
                "covered by some other entity in the source (routed to it, an alias of it, "
                "part of it, 'all X go through Y')? Pick the connecting sentence itself; if "
                "no sentence states such a connection, choose no_connection."
            ),
            options={
                **sentence_options,
                "no_connection": "No source sentence connects the claim's subject to another entity.",
            },
        ),
        ChoiceQuestion(
            question_id=f"br3::{unit.node_id}",
            instructions=(
                f"{describe_unit(store, unit)}\n"
                "Which source sentence states the PROPERTY or behaviour that the claim "
                "asserts — in ANY polarity (a sentence stating its opposite or negation "
                "also counts), possibly about a different entity than the claim's subject? "
                "If none does, choose no_property."
            ),
            options={
                **sentence_options,
                "no_property": (
                    "No source sentence states the asserted property or its opposite."
                ),
            },
        ),
    ]


def bridge_check_questions(
    store: TextStore, unit: Node, bridge: Sentence, prop: Sentence
) -> list[Question]:
    """Round B: the premises of the substitution inference, one question each.
    The rescue concludes phi(a) from: (P1) the bridge covers a fully / a IS b
    (br2); (P2a) the stated property matches the asserted one (br4); (P2b) the
    property sentence predicates of the BRIDGED entity, not some third one
    (br5) — without P2b a sentence about an unrelated entity passes br4, which
    deliberately ignores subjects (measured false S, conflict 189 shape).
    Reuses the relation check's 3-way side vocabulary for br4."""
    return [
        # deliberately claim-free: with the claim in view, a contradiction
        # scenario contaminates the coverage judgment (measured: 0.76 on a
        # supported case vs 0.35 on the same bridge shape with a contradicted
        # property). The bridge's completeness is a fact about one sentence.
        YesNoQuestion(
            question_id=f"br2::{unit.node_id}",
            instructions=(
                f'Sentence: "{bridge.text}"\n'
                "Does this sentence state a COMPLETE connection — ALL of its subject goes "
                "to, passes through, or simply is the other entity (exclusive routing, a "
                "plain alias, total coverage)? Partial coverage ('some', 'about half') or "
                "qualified coverage means no."
            ),
        ),
        ChoiceQuestion(
            question_id=f"br4::{unit.node_id}",
            instructions=(
                f'Property sentence [{prop.sentence_id}]: "{prop.text}"\n'
                f"{describe_unit(store, unit)}\n"
                "The SUBJECTS may differ — the connection between the claim's subject and "
                "this sentence's entity is checked separately, so ignore who or what the "
                "subject is. Compare ONLY the asserted property or behaviour itself with "
                "what this sentence states (it may state several things; the one matching "
                "the claim's property is what counts) and classify."
            ),
            options=SIDE_OPTIONS,
        ),
        # claim-free like br2: a fact about the two sentences alone. This is the
        # binding premise br4 deliberately looks away from.
        YesNoQuestion(
            question_id=f"br5::{unit.node_id}",
            instructions=(
                f'Bridge sentence: "{bridge.text}"\n'
                f'Property sentence: "{prop.text}"\n'
                "The bridge sentence connects something to a TARGET entity (the thing "
                "routed to, the alias, the whole it belongs to). Is the property "
                "sentence's statement made about that same target entity? If the "
                "property sentence is about some other entity that the bridge sentence "
                "does not mention, answer no."
            ),
        ),
    ]


def decide_bridge_rescue(
    unit: Node,
    answers: dict[str, Answer],
    *,
    min_prob: float,
) -> UnitJudgment | None:
    """Compose the bridge answers; None means the rescue does not apply and
    the original flat verdict stands."""
    br1 = answers.get(f"br1::{unit.node_id}")
    br3 = answers.get(f"br3::{unit.node_id}")
    bid, _s1 = staged_link_choice(br1, min_prob=min_prob, none_label="no_connection")
    pid, _s3 = staged_link_choice(br3, min_prob=min_prob, none_label="no_property")
    if bid is None or pid is None or bid == pid:
        return None
    full_p = (answers.get(f"br2::{unit.node_id}") or Answer("")).yes_prob or 0.0
    bind_p = (answers.get(f"br5::{unit.node_id}") or Answer("")).yes_prob or 0.0
    br4 = answers.get(f"br4::{unit.node_id}")
    probs = br4.probabilities if br4 and br4.probabilities else {}
    prop_match = max(probs, key=probs.__getitem__) if probs else "different"
    prop_p = probs.get(prop_match, 0.0)
    if full_p < min_prob or bind_p < min_prob or prop_p < min_prob:
        return None
    if prop_match == "different":
        return None
    judgment = UnitJudgment(
        node_id=unit.node_id,
        node_version=unit.version,
        verdict=Verdict.SUPPORTED if prop_match == "matches" else Verdict.CONTRADICTED,
        probabilities={"bridge_full": full_p, "bridge_binding": bind_p, f"property_{prop_match}": prop_p},
        evidence_sentence_ids=[bid, pid],
        flags=[f"bridge_rescue({bid}->{pid},{prop_match})"],
    )
    return judgment


def staged_link_choice(
    answer: Answer | None, *, min_prob: float, none_label: str = "no_such_link"
) -> tuple[str | None, str]:
    """Resolve a pick-a-sentence stage: (sentence_id, 'chosen') |
    (None, 'no_link'|'flat'). `none_label` is that question's none-option."""
    if answer is None or not answer.probabilities:
        return None, "flat"
    top = max(answer.probabilities, key=answer.probabilities.__getitem__)
    p = answer.probabilities[top]
    if p < min_prob:
        return None, "flat"
    if top == none_label:
        return None, "no_link"
    return top, "chosen"


def decide_relation_verdict(
    unit: Node,
    answers: dict[str, Answer],
    *,
    min_prob: float,
) -> UnitJudgment:
    judgment = UnitJudgment(node_id=unit.node_id, node_version=unit.version, verdict=Verdict.UNSUPPORTED)
    link_answer = answers.get(f"rel1::{unit.node_id}")
    if link_answer is not None:
        judgment.probabilities = dict(link_answer.probabilities)
        judgment.confidence = link_answer.confidence
    sid, status = staged_link_choice(link_answer, min_prob=min_prob)
    if sid is None:
        judgment.flags.append("staged_no_link" if status == "no_link" else "staged_flat_link")
        return judgment

    sides: list[tuple[str, str, float]] = []  # (which, top option, its prob)
    for which, qid in (("left", f"rel2::{unit.node_id}"), ("right", f"rel3::{unit.node_id}")):
        answer = answers.get(qid)
        probs = answer.probabilities if answer and answer.probabilities else {}
        top = max(probs, key=probs.__getitem__) if probs else "different"
        p = probs.get(top, 0.0)
        judgment.probabilities[f"{which}_{top}"] = p
        sides.append((which, top if p >= min_prob else "flat", p))

    detail = ",".join(f"{w}={t}({p:.2f})" for w, t, p in sides)
    # 'different' (sibling trigger, agent switch, widening) can only mean the
    # link is unstated; 'opposite' (same aspect, inverted value) is what
    # licenses CONTRADICTED; both sides matching makes the stated link cover
    # the claimed one.
    if any(t in ("different", "flat") for _, t, _ in sides):
        judgment.flags.append(f"staged_side_mismatch({sid},{detail})")
        return judgment
    # sequence only: the sides matching says nothing about direction, so the
    # stated order must agree with the claimed one (LEFT first, per the cut
    # option's definition). An explicitly inverted order is a stated
    # incompatibility (C); an unstated order leaves the claimed one unverified.
    if unit.relation_type is RelationType.SEQUENCE:
        order = answers.get(f"rel4::{unit.node_id}")
        order_probs = order.probabilities if order and order.probabilities else {}
        order_top = max(order_probs, key=order_probs.__getitem__) if order_probs else "no_order"
        order_p = order_probs.get(order_top, 0.0)
        judgment.probabilities[f"order_{order_top}"] = order_p
        if order_top == "right_first" and order_p >= min_prob:
            judgment.verdict = Verdict.CONTRADICTED
            judgment.evidence_sentence_ids = [sid]
            judgment.flags.append(f"staged_order_inverted({sid},p={order_p:.2f})")
            return judgment
        if order_top != "left_first" or order_p < min_prob:
            judgment.flags.append(f"staged_order_unstated({sid},{order_top},p={order_p:.2f})")
            return judgment
    if any(t == "opposite" for _, t, _ in sides):
        judgment.verdict = Verdict.CONTRADICTED
        judgment.evidence_sentence_ids = [sid]
        judgment.flags.append(f"staged_incompatible({sid},{detail})")
        return judgment
    judgment.verdict = Verdict.SUPPORTED
    judgment.evidence_sentence_ids = [sid]
    judgment.flags.append(f"staged_matched({sid})")
    return judgment


# Leading framing words a diagnostic part may carry; the part's CONTENT minus
# such a word being present verbatim is still a syntactic presence witness.
_WITNESS_LEAD_WORDS = {
    "if", "when", "unless", "because", "although", "while", "whereas",
    "and", "or", "but", "then",
}


def _normalized_tokens(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def verbatim_in_source(store: TextStore, part_text: str) -> bool:
    """Syntactic presence witness: the part's words appear verbatim (modulo
    case/punctuation/whitespace) inside one source passage. Content presence
    is a string property, so this is decidable with certainty — a probabilistic
    grounding answer must not override it (Def. 3's verbatim clause; measured
    false caps hit verbatim-supported if/when content at probe 0.02-0.47).
    Fails closed: a paraphrased part gets no witness and the cap stands."""
    norm = _normalized_tokens(part_text)
    tokens = norm.split()
    alt = ""
    if tokens and tokens[0] in _WITNESS_LEAD_WORDS and len(tokens) > 1:
        alt = " " + " ".join(tokens[1:]) + " "
    for passage in store.passages:
        hay = _normalized_tokens(passage)
        if norm in hay or (alt and alt in hay):
            return True
    return False


def apply_grounding_gate(
    unit: Node,
    judgment: "UnitJudgment",
    answers: dict[str, Answer],
    *,
    grounding_min_prob: float,
    store: TextStore | None = None,
) -> None:
    """An ungrounded diagnostic part caps a SUPPORTED verdict at UNSUPPORTED.
    CONTRADICTED is left alone (the block already flags a conflict), and
    UNSUPPORTED cannot get worse. A part whose text is verbatim present in a
    source passage is grounded by that fact alone: the syntactic witness
    overrides a low probe score (the gate stays a one-directional S->U
    operator; the witness only narrows where it fires, never adds S or C)."""
    if judgment.verdict is not Verdict.SUPPORTED or not unit.diagnostic_spans:
        return
    for i in range(len(unit.diagnostic_spans)):
        answer = answers.get(f"g::{unit.node_id}::{i}")
        if answer is None or answer.yes_prob is None:
            continue
        if answer.yes_prob < grounding_min_prob:
            if store is not None and verbatim_in_source(store, store.slice(unit.diagnostic_spans[i])):
                judgment.flags.append(f"grounding_witness_{i}(p={answer.yes_prob:.2f})")
                continue
            judgment.verdict = Verdict.UNSUPPORTED
            judgment.evidence_sentence_ids = []
            judgment.flags.append(f"ungrounded_part_{i}(p={answer.yes_prob:.2f})")
            return


@dataclass
class UnitJudgment:
    node_id: str
    node_version: int
    verdict: Verdict
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    evidence_sentence_ids: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_version": self.node_version,
            "verdict": self.verdict.value,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "evidence_sentence_ids": self.evidence_sentence_ids,
            "flags": self.flags,
        }


_LABEL_TO_VERDICT = {
    "supported": Verdict.SUPPORTED,
    "contradicted": Verdict.CONTRADICTED,
    "not_in_context": Verdict.UNSUPPORTED,
}


def decide_verdict(
    unit: Node,
    answers: dict[str, Answer],
    candidates: list[Sentence],
    *,
    verdict_min_prob: float,
    relevance_min_prob: float,
    evidence_floor_prob: float,
    max_evidence: int,
    embed_ranked: list[str] | None = None,
    embed_top_k: int = 2,
    embed_floor_prob: float = 0.25,
) -> UnitJudgment:
    judgment = UnitJudgment(node_id=unit.node_id, node_version=unit.version, verdict=Verdict.UNSUPPORTED)
    answer = answers.get(f"v::{unit.node_id}")
    if answer is None:
        judgment.flags.append("no_verdict_answer")
        return judgment
    judgment.probabilities = dict(answer.probabilities)
    judgment.confidence = answer.confidence

    ranked = sorted(
        (
            ((answers.get(f"r::{unit.node_id}::{s.sentence_id}") or Answer("")).yes_prob or 0.0, s.sentence_id)
            for s in candidates
        ),
        key=lambda t: -t[0],
    )
    judgment.evidence_sentence_ids = [
        sid for p, sid in ranked[:max_evidence] if p >= relevance_min_prob
    ]
    # Embedding prior: the top embedding-ranked sentences join the evidence if
    # the probe gave them at least a floor score. This rescues linking premises
    # ("X uses policy Y") that look unevidential alone and score 0.3-0.5 on the
    # probe, while the floor keeps out pure topic noise.
    if embed_ranked:
        probe = {sid: p for p, sid in ranked}
        for sid in embed_ranked[:embed_top_k]:
            if sid in judgment.evidence_sentence_ids:
                continue
            if probe.get(sid, 0.0) >= embed_floor_prob:
                judgment.evidence_sentence_ids.append(sid)
                judgment.flags.append(f"embed_prior({sid},probe={probe.get(sid, 0.0):.2f})")

    top_label = max(answer.probabilities, key=answer.probabilities.get, default=None)
    top_prob = answer.probabilities.get(top_label, 0.0) if top_label else 0.0
    if top_label is None or top_prob < verdict_min_prob:
        # the calibrated distribution is too flat: say "cannot verify" rather than guess
        judgment.verdict = Verdict.UNSUPPORTED
        judgment.flags.append(f"low_confidence(top={top_label},p={top_prob:.2f})")
        return judgment

    verdict = _LABEL_TO_VERDICT[top_label]
    if verdict is not Verdict.UNSUPPORTED and not judgment.evidence_sentence_ids:
        # Definite verdict but no sentence cleared the relevance bar. The
        # relevance probe is a separate, weaker signal than the verdict itself,
        # so before throwing a confident verdict away, accept the best-ranked
        # sentence if it is at least plausible (above the floor); flag it so
        # evaluation can count how often this path fires and how it performs.
        if ranked and ranked[0][0] >= evidence_floor_prob:
            judgment.evidence_sentence_ids = [ranked[0][1]]
            judgment.flags.append(f"evidence_rank_fallback(p={ranked[0][0]:.2f})")
        else:
            # nothing even plausibly citable: no basis for a definite verdict
            judgment.verdict = Verdict.UNSUPPORTED
            judgment.flags.append(f"no_evidence_for_{verdict.value.lower()}")
            return judgment
    judgment.verdict = verdict
    return judgment
