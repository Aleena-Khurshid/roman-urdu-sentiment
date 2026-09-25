"""
Text cleaning for Roman Urdu.

IMPORTANT: The exact same cleaning must run during training (in the Colab
notebook) and during serving (in the API). If they differ, the model sees
text at prediction time that looks different from what it learned on, and
accuracy silently drops. This mismatch is called "training-serving skew".
"""

import re

LABELS = ["negative", "neutral", "positive"]          # index = label id
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")                  # 3+ same chars in a row
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise one Roman Urdu sentence.

    Steps and reasons:
    1. lowercase          -> "Acha" and "acha" become the same word
    2. remove URLs        -> links carry no sentiment, only noise
    3. remove @mentions   -> usernames carry no sentiment
    4. squash repeats     -> "bohttttt" -> "bohtt", so spelling stretches
                             map to fewer variants
    5. collapse spaces    -> tidy whitespace

    We do NOT remove punctuation like "!" or "?" because it can carry
    emotion ("zabardast!!").
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def normalise_label(raw) -> str | None:
    """Map messy raw labels to one of LABELS, or None if unknown.

    The original dataset has a few typos (for example "Neative"), so we
    fix known ones and drop anything we cannot trust.
    """
    if raw is None:
        return None
    label = str(raw).strip().lower()
    fixes = {"neative": "negative"}
    label = fixes.get(label, label)
    return label if label in LABEL2ID else None
