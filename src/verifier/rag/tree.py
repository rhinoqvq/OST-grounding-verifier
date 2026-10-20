"""Semantic tree: the data structure the whole pipeline operates on.

Nodes are *interpretations* of immutable source text. A node never owns text:
it points at spans in the TextStore. Invalidation bumps the version and marks
derived nodes invalid; original text is always the recovery point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from verifier.rag.source import Span, TextStore


class NodeKind(str, Enum):
    FACT = "fact"                  # atomic verifiable assertion
    RELATION = "relation"          # asserted link between members (cause, contrast, sequence)
    CONDITIONAL = "conditional"    # if/unless: neither side asserted; kept whole
    DISJUNCTION = "disjunction"    # A or B: kept whole
    REPORTED = "reported"          # "some say ...", "may ...": not asserted; kept whole
    SENTENCE = "sentence"          # a whole original sentence (fallback unit / tree root level)


class NodeStatus(str, Enum):
    CANDIDATE = "candidate"
    CHECKED = "checked"
    INVALID = "invalid"


class RelationType(str, Enum):
    CAUSE = "cause"
    CONDITION = "condition"
    CONTRAST = "contrast"
    SEQUENCE = "sequence"
    NONE = "none"


@dataclass
class Node:
    node_id: str
    origin: str                      # "answer" | "source"
    sentence_id: str                 # sentence this node belongs to
    kind: NodeKind
    anchor_spans: list[Span]         # the text that carries this unit (may be discontiguous)
    inherited_spans: list[Span] = field(default_factory=list)  # shared subject/condition/referent
    relation_type: RelationType = RelationType.NONE
    member_ids: list[str] = field(default_factory=list)        # for RELATION nodes
    parent_id: str | None = None
    display_text: str = ""           # self-contained rendering (subject filled in); never a source of truth
    status: NodeStatus = NodeStatus.CANDIDATE
    version: int = 1
    notes: list[str] = field(default_factory=list)


@dataclass
class Edge:
    """Non-tree relation between nodes (shared subject, coreference)."""

    kind: str
    from_id: str
    to_id: str


@dataclass
class SemanticTree:
    store: TextStore
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    def add(self, node: Node) -> Node:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id {node.node_id}")
        self.nodes[node.node_id] = node
        return node

    # -- navigation --------------------------------------------------------
    def children(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def ancestors(self, node_id: str) -> list[Node]:
        out: list[Node] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            node = self.nodes[current]
            out.append(node)
            current = node.parent_id
        return out

    def subtree(self, node_id: str) -> list[Node]:
        out = [self.nodes[node_id]]
        stack = [node_id]
        while stack:
            for child in self.children(stack.pop()):
                out.append(child)
                stack.append(child.node_id)
        return out

    def roots(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent_id is None]

    # -- integrity ----------------------------------------------------------
    def validate_structure(self) -> list[str]:
        """Deterministic checks. Returns a list of problems (empty = ok)."""
        problems: list[str] = []
        for node in self.nodes.values():
            if not node.anchor_spans and node.kind is not NodeKind.RELATION:
                problems.append(f"{node.node_id}: no anchor spans")
            for span in [*node.anchor_spans, *node.inherited_spans]:
                try:
                    self.store.slice(span)
                except (ValueError, KeyError, IndexError) as exc:
                    problems.append(f"{node.node_id}: bad span {span}: {exc}")
            if node.parent_id is not None and node.parent_id not in self.nodes:
                problems.append(f"{node.node_id}: missing parent {node.parent_id}")
            for member in node.member_ids:
                if member not in self.nodes:
                    problems.append(f"{node.node_id}: missing member {member}")
            if node.kind is NodeKind.RELATION and len(node.member_ids) < 2:
                problems.append(f"{node.node_id}: relation needs >= 2 members")
            # anchors of an answer node must stay inside its sentence
            if node.origin == "answer" and node.kind is not NodeKind.RELATION:
                try:
                    sentence = self.store.sentence_by_id(node.sentence_id)
                except KeyError:
                    problems.append(f"{node.node_id}: unknown sentence {node.sentence_id}")
                else:
                    for span in node.anchor_spans:
                        if span.doc != sentence.span.doc or not (
                            sentence.span.start <= span.start and span.end <= sentence.span.end
                        ):
                            problems.append(
                                f"{node.node_id}: anchor {span} outside sentence {node.sentence_id}"
                            )
        problems.extend(self._find_cycles())
        return problems

    def _find_cycles(self) -> list[str]:
        problems = []
        for node_id in self.nodes:
            seen = set()
            current: str | None = node_id
            while current is not None:
                if current in seen:
                    problems.append(f"cycle through {current}")
                    break
                seen.add(current)
                current = self.nodes[current].parent_id if current in self.nodes else None
        return problems

    # -- invalidation (rebuild support) --------------------------------------
    def invalidate_sentence(self, sentence_id: str) -> list[Node]:
        """Mark every node derived from a sentence invalid and bump versions.
        The sentence itself (original text) stays valid — it is the fallback."""
        touched = []
        for node in self.nodes.values():
            if node.sentence_id == sentence_id and node.kind is not NodeKind.SENTENCE:
                node.status = NodeStatus.INVALID
                node.version += 1
                touched.append(node)
        return touched
