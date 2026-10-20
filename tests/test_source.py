from verifier.rag.source import Span, TextStore, segment


def test_version_number_does_not_split_sentence():
    text = "Beginning with version 3.7.0, WAL mode is available. It works."
    spans = segment("answer", text)
    assert len(spans) == 2
    assert text[spans[0].start : spans[0].end].startswith("Beginning")
    assert "3.7.0," in text[spans[0].start : spans[0].end]


def test_numbered_steps_stay_attached():
    text = "1. Open the Start menu. 2. Click Settings."
    spans = segment("answer", text)
    pieces = [text[s.start : s.end] for s in spans]
    assert all(len(p.split()) > 1 for p in pieces), pieces


def test_abbreviations_do_not_split():
    text = "Use a tool, e.g. ripgrep, to search. Then review."
    spans = segment("answer", text)
    assert len(spans) == 2


def test_formula_and_filenames_survive():
    text = "Enter =WORKDAY(DATE(2022,1,1),DATE(2022,7,1)) into a cell. See data.json for details."
    spans = segment("answer", text)
    assert len(spans) == 2
    assert "=WORKDAY" in text[spans[0].start : spans[0].end]


def test_slice_is_verbatim(wal_store: TextStore):
    for sentence in [*wal_store.answer_sentences, *wal_store.all_source_sentences()]:
        assert wal_store.slice(sentence.span) == sentence.text
        doc_text = wal_store.answer if sentence.span.doc == "answer" else wal_store.passages[
            wal_store.passage_index_of(sentence.span.doc)
        ]
        assert sentence.text in doc_text


def test_wal_example_segmentation(wal_store: TextStore):
    assert len(wal_store.answer_sentences) == 3
    assert len(wal_store.passage_sentences[0]) == 3
    assert len(wal_store.passage_sentences[1]) == 2
    assert wal_store.passage_sentences[0][2].text.startswith("However, WAL does not work")
    assert wal_store.passage_sentences[1][1].text.startswith("Automatic checkpointing")


def test_token_range_to_span_roundtrip(wal_store: TextStore):
    span = wal_store.token_range_to_span(0, 2)
    assert wal_store.slice(span) == "SQLite WAL mode"


def test_empty_and_whitespace_segments():
    assert segment("answer", "") == []
    assert segment("answer", "   \n  ") == []


def test_slice_rejects_out_of_range(wal_store: TextStore):
    import pytest

    with pytest.raises(ValueError):
        wal_store.slice(Span("answer", 0, 10_000))
