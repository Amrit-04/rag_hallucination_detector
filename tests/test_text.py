from app.utils.text import chunk_text, split_sentences, stable_id


def test_stable_id_is_deterministic():
    assert stable_id("hello") == stable_id("hello")


def test_split_sentences():
    assert split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_chunk_text_overlaps_long_text():
    chunks = chunk_text("a" * 1200, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert all(chunks)

