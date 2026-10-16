"""
SurSaathi — shared NLP preprocessing
------------------------------------
Single source of truth for text cleaning. Training, evaluation,
similarity building and the Flask backend all import from here, so
preprocessing can never drift between training and prediction.

Two scripts are handled:
  * Roman Hindi (transliterated)  -> clean_roman()
  * Devanagari Hindi              -> clean_devanagari()

Roman-Hindi spelling is not standardised (pyaar / pyar / pyaaar), so
clean_roman() applies light rule-based normalisation before stopword
removal. Stemming / lemmatisation is deliberately NOT used: English
stemmers corrupt Roman-Hindi words and no reliable Hindi stemmer
exists for the Roman script.
"""

import re
import unicodedata

# ------------------------------------------------------------------
# Roman Hindi
# ------------------------------------------------------------------

_ROMAN_STOPWORDS_RAW = {
    "hai", "hain", "ho", "hoon", "hun", "the", "thi", "tha", "the",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "main", "mai",
    "hum", "humne", "tum", "tumhe", "tumko", "aur", "na", "nahi",
    "nahin", "toh", "to", "bhi", "ye", "yeh", "wo", "woh", "jo",
    "jaise", "kya", "kyun", "kyu", "koi", "kuch", "sa", "si", "hi",
    "bas", "par", "pe", "kar", "gaye", "gayi", "gaya", "diya",
    "diye", "liya",
}

# Vowel-length spelling variants -> one canonical form
_VOWEL_RULES = [
    (re.compile(r"(.)\1{2,}"), r"\1"),   # kahaaaan -> kahan (3+ repeats)
    (re.compile(r"aa"), "a"),            # pyaar -> pyar
    (re.compile(r"ee"), "i"),            # dheere -> dhire
    (re.compile(r"ii"), "i"),
    (re.compile(r"oo"), "u"),            # tum / toom -> tum
    (re.compile(r"uu"), "u"),
]


def normalize_roman_word(word: str) -> str:
    for pattern, repl in _VOWEL_RULES:
        word = pattern.sub(repl, word)
    return word


ROMAN_STOPWORDS = {normalize_roman_word(w) for w in _ROMAN_STOPWORDS_RAW}


def clean_roman(text, normalize: bool = True) -> str:
    """lowercase -> keep a-z -> (normalise spelling) -> drop stopwords
    -> drop 1-letter tokens.  normalize=False reproduces the original
    baseline preprocessing used in the first version of the project."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    out = []
    for w in text.split():
        if not normalize:
            if w in _ROMAN_STOPWORDS_RAW or len(w) <= 1:
                continue
            out.append(w)
            continue
        w = normalize_roman_word(w)
        if w in ROMAN_STOPWORDS or len(w) <= 1:
            continue
        out.append(w)
    return " ".join(out)


def clean_roman_baseline(text) -> str:
    return clean_roman(text, normalize=False)


# ------------------------------------------------------------------
# Devanagari
# ------------------------------------------------------------------

DEVANAGARI_STOPWORDS = {
    "है", "हैं", "हूं", "हूँ", "हो", "था", "थी", "थे",
    "का", "की", "के", "को", "से", "में", "मैं", "मै",
    "हम", "तुम", "और", "न", "ना", "नहीं", "तो", "भी",
    "ये", "यह", "वो", "वह", "जो", "क्या", "कोई", "कुछ",
    "पर", "पे", "कर", "ही", "इस", "उस", "एक", "बस",
}

_DEV_KEEP = re.compile(r"[^\u0900-\u097F\s]")


def clean_devanagari(text) -> str:
    """Unicode-normalise, strip nukta/chandrabindu variants, remove
    punctuation (incl. danda), drop Hindi stopwords."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u093C", "")      # nukta: ज़ -> ज
    text = text.replace("\u0901", "\u0902")  # chandrabindu -> anusvara
    text = re.sub(r"[\u0964\u0965]", " ", text)   # danda
    text = _DEV_KEEP.sub(" ", text)
    words = [w for w in text.split()
             if w not in DEVANAGARI_STOPWORDS and len(w) > 1]
    return " ".join(words)


# ==================================================================
# v2 additions: spelling variants, suffix stripping, repetition handling
# (used by ablation_preprocessing.py; the deployed model still uses
#  clean_roman_baseline unless the ablation shows a clear gain)
# ==================================================================
import json as _json
from pathlib import Path as _Path

# Hand-written variants (colloquial / vowel-dropped Roman spellings)
MANUAL_VARIANTS = {
    "kia": "kya", "kyaa": "kya", "hy": "hai", "hm": "hum", "ksi": "kisi",
    "agr": "agar", "q": "kyun", "koie": "koi", "phli": "pehli",
    "jaie": "jaye", "yeh": "ye",
}

_vm_path = _Path(__file__).resolve().parent / "variant_map.json"
MINED_VARIANTS = (_json.load(open(_vm_path, encoding="utf-8"))
                  if _vm_path.exists() else {})

_CONS_REPEAT = re.compile(r"([bcdfghjklmnpqrstvwxyz])\1+")   # dill -> dil

# Light Roman-Hindi suffix stripper: longest suffix first, stem >= 3 letters
ROMAN_SUFFIXES = sorted(
    ["ein", "ata", "ati", "ate", "on", "ti", "ta", "te", "ya", "ne", "na"],
    key=len, reverse=True)


def stem_roman(word: str) -> str:
    for suf in ROMAN_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


# Light Devanagari suffix stripper (plural/oblique/gerund/verb endings)
DEV_SUFFIXES = sorted(
    ["\u093e\u090f\u0902", "\u093f\u092f\u093e\u0902", "\u093e\u0913\u0902",
     "\u094b\u0902", "\u0947\u0902", "\u0924\u093e", "\u0924\u0940",
     "\u0924\u0947", "\u092f\u093e", "\u0928\u093e", "\u0928\u0947"],
    key=len, reverse=True)


def stem_devanagari(word: str) -> str:
    for suf in DEV_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def dedupe_lines(text) -> str:
    """Songs repeat their chorus. Lines are separated by 2+ spaces or
    newlines in this dataset; keep only the first occurrence of each."""
    seen, keep = set(), []
    for line in re.split(r"\s{2,}|\n", str(text)):
        key = re.sub(r"\W+", " ", line.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            keep.append(line)
    return "  ".join(keep)


def collapse_runs(tokens):
    """kahaan kahaan kahaan -> kahaan"""
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def clean_roman_v2(text, variants=True, stem=False, repeat=False) -> str:
    if repeat:
        text = dedupe_lines(text)
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    out = []
    for w in text.split():
        if variants:
            w = _CONS_REPEAT.sub(r"\1", normalize_roman_word(w))
            w = MANUAL_VARIANTS.get(w, MINED_VARIANTS.get(w, w))
            w = normalize_roman_word(w)
        if w in ROMAN_STOPWORDS or w in _ROMAN_STOPWORDS_RAW or len(w) <= 1:
            continue
        if stem:
            w = stem_roman(w)
        out.append(w)
    if repeat:
        out = collapse_runs(out)
    return " ".join(out)


def clean_devanagari_v2(text, stem=False, repeat=False) -> str:
    if repeat:
        text = dedupe_lines(text)
    words = clean_devanagari(text).split()
    if stem:
        words = [stem_devanagari(w) for w in words]
    if repeat:
        words = collapse_runs(words)
    return " ".join(words)
