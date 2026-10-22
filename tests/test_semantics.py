from verifier.rag.source import TextStore
from verifier.rag.tree import NodeKind
from verifier.methods.optimized.semantics.check import apply_check_results
from verifier.methods.optimized.semantics.propose import Proposal, ProposedUnit, TokenRange, proposal_to_tree
from verifier.methods.optimized.semantics.select import coverage_gaps, is_entirely_non_claim, select_units

from conftest import scripted_decider, token_range_for
from verifier.models.decider import Answer


def test_proposal_to_tree_builds_units(wal_store, wal_proposal):
    tree, non_claim, problems = proposal_to_tree(wal_store, wal_proposal)
    unit_ids = {n.node_id for n in tree.nodes.values() if n.kind is not NodeKind.SENTENCE}
    assert unit_ids == {"u1", "u2", "u3", "u4"}
    assert problems == []
    u3 = tree.nodes["u3"]
    assert u3.sentence_id == "A.S1"
    assert wal_store.slice(u3.anchor_spans[0]).startswith("automatically checkpoints")


def test_out_of_range_tokens_drop_the_unit(wal_store):
    proposal = Proposal(
        units=[
            ProposedUnit(
                unit_id="bad",
                kind="fact",
                anchor_token_ranges=[TokenRange(first=9_000, last=9_001)],
                display_text="x",
            )
        ]
    )
    tree, _, problems = proposal_to_tree(wal_store, proposal)
    assert "bad" not in tree.nodes
    assert problems


def test_cross_sentence_anchor_is_rejected(wal_store):
    n_tokens = len(wal_store.answer_tokens)
    proposal = Proposal(
        units=[
            ProposedUnit(
                unit_id="wide",
                kind="fact",
                anchor_token_ranges=[TokenRange(first=0, last=n_tokens - 1)],
                display_text="everything",
            )
        ]
    )
    tree, _, problems = proposal_to_tree(wal_store, proposal)
    assert "wide" not in tree.nodes
    assert any("cross" in p for p in problems)


def test_relation_with_missing_member_is_dropped(wal_store):
    proposal = Proposal(
        units=[
            ProposedUnit(
                unit_id="rel",
                kind="relation",
                anchor_token_ranges=[TokenRange(first=0, last=1)],
                relation_type="cause",
                member_ids=["ghost1", "ghost2"],
                display_text="x because y",
            )
        ]
    )
    tree, _, problems = proposal_to_tree(wal_store, proposal)
    assert "rel" not in tree.nodes


async def test_failed_fidelity_falls_back_to_sentence(wal_store, wal_proposal):
    tree, non_claim, _ = proposal_to_tree(wal_store, wal_proposal)
    decider = scripted_decider(
        {
            "fid::u2": Answer(
                "fid::u2",
                choice="lost_qualifier",
                probabilities={"faithful": 0.2, "changed_meaning": 0.1,
                               "added_information": 0.1, "lost_qualifier": 0.6},
            )
        }
    )
    from verifier.harness.batching import ask_batched
    from verifier.methods.optimized.semantics.check import check_questions

    answers = await ask_batched(
        decider, "state", check_questions(wal_store, tree), max_per_request=50, max_parallel=2
    )
    outcome = apply_check_results(tree, answers, fidelity_min_prob=0.6)
    assert "A.S1" in outcome.fallback_sentence_ids
    selection = select_units(wal_store, tree, outcome.accepted_units, outcome.fallback_sentence_ids, non_claim)
    by_id = {u.node_id for u in selection.units}
    # u2/u3's sentence collapsed to one whole-sentence unit; u1/u4 survive
    assert "sent::A.S1" in by_id
    assert "u2" not in by_id and "u3" not in by_id
    assert {"u1", "u4"} <= by_id


def test_coverage_gap_detected(wal_store):
    sentence = wal_store.answer_sentences[1]
    # cover only the first half of the sentence
    partial = [wal_store.token_range_to_span(*_range_of(wal_store, "Alongside the main database file, it uses two auxiliary files"))]
    gaps = coverage_gaps(wal_store, sentence, partial, [])
    assert gaps, "the checkpoint half should be reported as uncovered"


def _range_of(store: TextStore, fragment: str) -> tuple[int, int]:
    r = token_range_for(store, fragment)
    return r.first, r.last


def test_entirely_non_claim_sentence(wal_store):
    store = TextStore.from_request("The log file grew. Hope this helps!", [])
    span = store.answer_sentences[1].span
    assert is_entirely_non_claim(store.answer_sentences[1], [span])
    assert not is_entirely_non_claim(store.answer_sentences[0], [span])


def test_short_answers_are_claims_by_content_not_length():
    """sa1: a bare value or name is a claim however short; a sentence with no
    content word (punctuation, a dangling "is:", a lone list enumerator next to
    other sentences) is not."""
    from verifier.rag.tree import SemanticTree

    for text in ("3", "80%", "Paris", "No.", "1964", "3."):
        store = TextStore.from_request(text, [])
        (sentence,) = store.answer_sentences
        assert not is_entirely_non_claim(sentence, [], whole_answer=True), text
        units = select_units(store, SemanticTree(store=store), [], set(), []).units
        assert [u.node_id for u in units] == ["sent::A.S0"], text

    for text in ("is:", '"', "--"):
        (sentence,) = TextStore.from_request(text, []).answer_sentences
        assert is_entirely_non_claim(sentence, [], whole_answer=True), text
    for text in ("4.)", "3.", "3. -"):
        (sentence,) = TextStore.from_request(text, []).answer_sentences
        assert is_entirely_non_claim(sentence, [], whole_answer=False), text
    # meta-discourse coverage still decides first
    (sentence,) = TextStore.from_request("63%", []).answer_sentences
    assert is_entirely_non_claim(sentence, [sentence.span], whole_answer=True)
