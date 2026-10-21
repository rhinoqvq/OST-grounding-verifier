from verifier.harness.context import build_source_state
from verifier.output.citations import make_citation
from verifier.rag.source import TextStore


def test_source_state_labels_passages_the_way_answers_count_them():
    store = TextStore.from_request(
        "Passage 2 says the sky is blue.",
        ["Grass is green.", "The sky is blue."],
    )
    state = build_source_state(store)
    assert "--- passage 1 ---\n[P0.S0] Grass is green." in state
    assert "--- passage 2 ---\n[P1.S0] The sky is blue." in state
    assert "--- passage 0 ---" not in state


def test_citation_index_stays_zero_based():
    store = TextStore.from_request("x", ["Grass is green.", "The sky is blue."])
    assert make_citation(store, "P1.S0").index == 1
