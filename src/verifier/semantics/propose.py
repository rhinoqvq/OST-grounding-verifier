"""Round 1: the generative LLM proposes the semantic structure of the answer.

The model sees ONLY the answer (never the sources) as numbered tokens, and
returns token ranges — character offsets are always computed by code. Units it
proposes are *candidates*: Jev checks them in round 2, deterministic rules
validate them here, and anything broken falls back to the whole sentence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from verifier.harness.context import build_answer_token_view
from verifier.models.openai_llm import OpenAILLM
from verifier.rag.source import Span, TextStore
from verifier.rag.tree import Node, NodeKind, NodeStatus, RelationType, SemanticTree


class TokenRange(BaseModel):
    first: int = Field(ge=0, description="first token index, inclusive")
    last: int = Field(ge=0, description="last token index, inclusive")


class ProposedUnit(BaseModel):
    unit_id: str
    kind: Literal["fact", "relation", "conditional", "disjunction", "reported"]
    anchor_token_ranges: list[TokenRange] = Field(default_factory=list)
    inherited_token_ranges: list[TokenRange] = Field(default_factory=list)
    relation_type: Literal["cause", "condition", "contrast", "sequence", "none"] = "none"
    member_ids: list[str] = Field(default_factory=list)
    display_text: str


class Proposal(BaseModel):
    units: list[ProposedUnit]
    non_claim_token_ranges: list[TokenRange] = Field(default_factory=list)


_SYSTEM_PROMPT = """You decompose an answer into minimal verifiable claim units. You see ONLY the answer (numbered tokens); you never see any source documents, and you must not correct, complete or doubt anything the answer says — even if you believe it is wrong, reproduce its meaning exactly.

Rules:
1. Split a sentence only where it conjoins independent assertions ("and", lists, two clauses sharing a subject). Each part becomes a `fact` unit.
2. Never split across: conditionals (if/unless/only when -> one `conditional` unit), disjunctions (or -> one `disjunction` unit), reported or hedged statements (reportedly/may/some say -> one `reported` unit), or a negation and what it negates.
3. A named connection (because/so/although/then) between two asserted parts: output the parts as `fact` units AND one `relation` unit whose member_ids are those parts, with relation_type cause/contrast/sequence (use `condition` only inside a conditional). The relation's anchor is the connective region (or the whole sentence).
4. anchor_token_ranges: the tokens that carry the unit's content (inclusive ranges, may be several). inherited_token_ranges: tokens elsewhere in the SAME sentence that the unit depends on (shared subject, shared condition, the word a pronoun refers to).
5. display_text: the unit restated so it stands alone (fill in the subject a pronoun refers to), changing nothing else — keep every number, unit, qualifier (approximately, by default, most) exactly as written.
6. non_claim_token_ranges: greetings, meta-talk about the passages themselves ("Based on the provided passages", "Passage 3 provides the answer"), offers to help. If meta-talk makes a verifiable statement ("the steps in passage 1 turn Wi-Fi on"), that statement IS a claim unit.
7. Every content-bearing token of the answer must end up in some unit's anchor or in non_claim_token_ranges. When in doubt, keep a whole sentence as one `fact` unit.

Example answer: "0:Because 1:the 2:disk 3:was 4:full, 5:the 6:backup 7:failed 8:and 9:the 10:service 11:stopped. 12:Hope 13:this 14:helps!"
Correct output (abbreviated): fact u1 "The disk was full" (anchor 1-4), fact u2 "The backup failed" (anchor 5-7), fact u3 "The service stopped" (anchor 9-11), relation u4 cause members [u1, u2] (anchor 0, display "The disk being full is why the backup failed"), non_claim 12-14."""


def build_decompose_messages(store: TextStore) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Decompose this answer:\n" + build_answer_token_view(store),
        },
    ]


async def propose_structure(llm: OpenAILLM, store: TextStore) -> Proposal:
    return await llm.generate_structured(build_decompose_messages(store), Proposal)


def _ranges_to_spans(store: TextStore, ranges: list[TokenRange]) -> list[Span]:
    return [store.token_range_to_span(r.first, min(r.last, len(store.answer_tokens) - 1)) for r in ranges]


def proposal_to_tree(store: TextStore, proposal: Proposal) -> tuple[SemanticTree, list[Span], list[str]]:
    """Build the semantic tree from a proposal.

    Returns (tree, non_claim_spans, problems). Units that violate deterministic
    rules are dropped and reported in `problems`; coverage fallback in
    `semantics.select` will catch anything they were supposed to cover.
    """
    tree = SemanticTree(store=store)
    problems: list[str] = []

    sentence_node_by_id: dict[str, Node] = {}
    for s in store.answer_sentences:
        node = tree.add(
            Node(
                node_id=f"sent::{s.sentence_id}",
                origin="answer",
                sentence_id=s.sentence_id,
                kind=NodeKind.SENTENCE,
                anchor_spans=[s.span],
                display_text=s.text,
                status=NodeStatus.CHECKED,  # original text is ground truth, not an interpretation
            )
        )
        sentence_node_by_id[s.sentence_id] = node

    kind_map = {
        "fact": NodeKind.FACT,
        "relation": NodeKind.RELATION,
        "conditional": NodeKind.CONDITIONAL,
        "disjunction": NodeKind.DISJUNCTION,
        "reported": NodeKind.REPORTED,
    }

    proposed_ids = {u.unit_id for u in proposal.units}
    for unit in proposal.units:
        try:
            anchors = _ranges_to_spans(store, unit.anchor_token_ranges)
            inherited = _ranges_to_spans(store, unit.inherited_token_ranges)
        except IndexError as exc:
            problems.append(f"{unit.unit_id}: {exc}")
            continue
        if not anchors:
            problems.append(f"{unit.unit_id}: no anchors")
            continue
        sentence = store.answer_sentence_containing(
            Span(store.ANSWER_DOC, min(a.start for a in anchors), max(a.end for a in anchors))
        )
        if sentence is None:
            problems.append(f"{unit.unit_id}: anchors cross sentence boundaries")
            continue
        if unit.kind == "relation" and (
            len(unit.member_ids) < 2 or any(m not in proposed_ids for m in unit.member_ids)
        ):
            problems.append(f"{unit.unit_id}: relation with unknown/insufficient members")
            continue
        tree.add(
            Node(
                node_id=unit.unit_id,
                origin="answer",
                sentence_id=sentence.sentence_id,
                kind=kind_map[unit.kind],
                anchor_spans=anchors,
                inherited_spans=inherited,
                relation_type=RelationType(unit.relation_type),
                member_ids=list(unit.member_ids),
                parent_id=f"sent::{sentence.sentence_id}",
                display_text=unit.display_text.strip(),
            )
        )

    # drop relations whose members got dropped
    for node in list(tree.nodes.values()):
        if node.kind is NodeKind.RELATION and any(m not in tree.nodes for m in node.member_ids):
            problems.append(f"{node.node_id}: member was dropped; dropping relation")
            del tree.nodes[node.node_id]

    try:
        non_claim = _ranges_to_spans(store, proposal.non_claim_token_ranges)
    except IndexError as exc:
        problems.append(f"non_claim ranges: {exc}")
        non_claim = []

    problems.extend(tree.validate_structure())
    # remove nodes flagged by structural validation (keep sentence nodes)
    bad_ids = {p.split(":", 1)[0] for p in problems if ":" in p}
    for node_id in list(tree.nodes):
        if node_id in bad_ids and tree.nodes[node_id].kind is not NodeKind.SENTENCE:
            del tree.nodes[node_id]
    return tree, non_claim, problems
