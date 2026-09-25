from src.preprocess import clean_text, normalise_label


def test_lowercase_and_spaces():
    assert clean_text("  Bohat   ACHA  ") == "bohat acha"


def test_removes_urls_and_mentions():
    assert clean_text("@ali dekho https://x.com zabardast") == "dekho zabardast"


def test_squashes_repeated_letters():
    assert clean_text("bohtttttt acha") == "bohtt acha"


def test_keeps_exclamation():
    assert "!" in clean_text("zabardast!")


def test_non_string_returns_empty():
    assert clean_text(None) == ""


def test_label_typo_fixed():
    assert normalise_label("Neative") == "negative"
    assert normalise_label(" Positive ") == "positive"
    assert normalise_label("random") is None
