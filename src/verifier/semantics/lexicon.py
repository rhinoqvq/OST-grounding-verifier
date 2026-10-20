"""The one connective lexicon: every connective-driven behavior is a projection.

Redundancy audit (2026-10-22, revised after rebuttal): what must die is the
independent hand-maintenance of overlapping word lists — the scope guard's
effective domain is `conditional ∩ candidate-generating roles`, so adding a
word to one list without the other is a silent no-op (it happened twice).
Derived runtime views are fine; this table is the single place with the
authority to change membership.

Roles:
  cut          interior cut-point candidate (the _CUT_RE alternation)
  lead         sentence-initial subordinator ("Because A, B")
  inversion    verbless conditional inversion ("Should the watchdog fire, ...")
  conditional  a keep_whole verdict on this marker triggers the scope guard
  share        the cut question also offers shared-part options
  strippable   a witness may strip this as a leading framing word

Unit of record is a surface pattern, not a concept: 'should' earns its roles
only in the inversion position (the regex anchors it sentence-initially);
an interior 'should' has no role here.
"""

from __future__ import annotations

_LEXICON: dict[str, frozenset[str]] = {
    "because":     frozenset({"cut", "lead", "strippable"}),
    "so that":     frozenset({"cut"}),
    "so":          frozenset({"cut"}),
    "therefore":   frozenset({"cut"}),
    "thus":        frozenset({"cut"}),
    "but":         frozenset({"cut", "strippable"}),
    "however":     frozenset({"cut"}),
    "whereas":     frozenset({"cut", "lead", "strippable"}),
    "while":       frozenset({"cut", "lead", "strippable"}),
    "although":    frozenset({"cut", "lead", "strippable"}),
    "which":       frozenset({"cut"}),
    "who":         frozenset({"cut"}),
    "whose":       frozenset({"cut"}),
    "and":         frozenset({"cut", "share", "strippable"}),
    "or":          frozenset({"cut", "strippable"}),
    "unless":      frozenset({"cut", "lead", "conditional", "strippable"}),
    "if":          frozenset({"cut", "lead", "conditional", "strippable"}),
    "when":        frozenset({"cut", "lead", "conditional", "strippable"}),
    "after which": frozenset({"cut"}),
    "then":        frozenset({"cut", "strippable"}),
    "should":      frozenset({"inversion", "conditional"}),
    "had":         frozenset({"inversion", "conditional"}),
    "were":        frozenset({"inversion", "conditional"}),
}


def words_with(role: str) -> tuple[str, ...]:
    """All lexicon words carrying a role, in table order (order is cosmetic:
    the cut regex sorts by length, the lead regexes have no prefix overlaps)."""
    return tuple(w for w, roles in _LEXICON.items() if role in roles)
