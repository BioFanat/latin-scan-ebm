"""Text normalization: raw Latin strings → canonical form.

Unicode NFC, lowercasing, punctuation stripping, whitespace collapse,
retention of word boundaries. Preserves editorial diacritics as an
auxiliary feature stream rather than required input.
"""

from __future__ import annotations

import re
import unicodedata


# Punctuation to strip: everything that isn't a letter or space.
# We keep letters (including accented ones) and spaces.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")

# Macron / breve combining characters
_COMBINING_MACRON = "\u0304"   # ā = a + combining macron
_COMBINING_BREVE = "\u0306"    # ă = a + combining breve
_COMBINING_DIAERESIS = "\u0308"  # ë = e + combining diaeresis

# Precomposed macron vowels → base vowel
_MACRON_MAP: dict[str, str] = {
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ȳ": "y",
}

# Precomposed breve vowels → base vowel
_BREVE_MAP: dict[str, str] = {
    "ă": "a", "ĕ": "e", "ĭ": "i", "ŏ": "o", "ŭ": "u",
}


def normalize(raw: str) -> str:
    """Normalize a raw Latin string to canonical form.

    Steps:
    1. Unicode NFC normalization
    2. Strip macrons, breves, diaereses (store separately if needed)
    3. Lowercase
    4. Strip punctuation (keep letters and spaces)
    5. Collapse whitespace
    6. Strip leading/trailing whitespace
    """
    # NFC first so precomposed forms are consistent
    text = unicodedata.normalize("NFC", raw)

    # Strip macrons and breves (precomposed)
    for accented, base in _MACRON_MAP.items():
        text = text.replace(accented, base)
        text = text.replace(accented.upper(), base.upper())
    for accented, base in _BREVE_MAP.items():
        text = text.replace(accented, base)
        text = text.replace(accented.upper(), base.upper())

    # Strip diaeresis (precomposed forms after NFC)
    for base in "aeiouy":
        composed = unicodedata.normalize("NFC", base + _COMBINING_DIAERESIS)
        if composed != base + _COMBINING_DIAERESIS:  # NFC actually composed it
            text = text.replace(composed, base)
            text = text.replace(composed.upper(), base.upper())

    # Strip any remaining combining diacritics
    text = text.replace(_COMBINING_MACRON, "")
    text = text.replace(_COMBINING_BREVE, "")
    text = text.replace(_COMBINING_DIAERESIS, "")

    # Lowercase
    text = text.lower()

    # Strip punctuation
    text = _PUNCT_RE.sub("", text)

    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)

    return text.strip()
