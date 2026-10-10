"""Frozen, GT-free CCDA head validity under the documented lexicon assumption."""

import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).with_name("visual_nouns.txt")
TOKENS = re.compile(r"[^\W_]+", re.UNICODE)
IRREGULAR = {"men": "man", "women": "woman", "people": "person", "children": "child",
             "feet": "foot", "mice": "mouse", "geese": "goose"}


def normalize(text):
    return " ".join(TOKENS.findall(text.casefold()))


@lru_cache(maxsize=1)
def lexicon():
    return frozenset(normalize(line) for line in LEXICON_PATH.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#"))


def singular_form(phrase):
    words = phrase.split()
    if not words:
        return phrase
    word = words[-1]
    if word in IRREGULAR:
        words[-1] = IRREGULAR[word]
    elif word.endswith("ies") and len(word) > 4:
        words[-1] = word[:-3] + "y"
    elif word.endswith(("ches", "shes", "xes", "zes")):
        words[-1] = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        words[-1] = word[:-1]
    return " ".join(words)


def head_validity(head, expression):
    """Return visual-noun, full-token-phrase appearance, and conjunction."""
    head_phrase = normalize(head or "")
    sentence = normalize(expression or "")
    visual = bool(head_phrase) and (head_phrase in lexicon() or singular_form(head_phrase) in lexicon())
    appears = bool(head_phrase) and f" {head_phrase} " in f" {sentence} "
    return visual, appears, visual and appears
