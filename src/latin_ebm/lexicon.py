"""Vowel length lexicon: lookup natural vowel lengths for word forms.

Uses the MQDQ frequency-based dictionary from the anceps project,
with optional Morpheus fallback. The MQDQ dictionary provides
empirical, author-specific vowel length data derived from ~259K
scanned verses in the MQDQ/Pedecerto corpus.

MQDQ macron encoding:
    _ after vowel = long
    ^ after vowel = short
    * after vowel = anceps/ambiguous
    [ae], [oe] = diphthong (one vowel position, always long)
    j = consonantal i
    v = consonantal u (replaces u in forms)
    qv = qu
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from latin_ebm.types import PhonWeight


# Vowels in the macron notation (excludes j and v which are consonantal)
_MACRON_VOWELS = set("aeiouy")

# Pattern to extract vowel positions and their marks from a macronized form.
# Vowels can be followed by _ (long), ^ (short), * (anceps), or nothing.
# Diphthongs [ae], [oe], etc. count as one long vowel position.
_DIPHTHONG_RE = re.compile(r"\[([a-z]{2})\]")


def _parse_macron_form(form: str) -> list[PhonWeight | None]:
    """Parse a macronized form into vowel lengths.

    Returns one PhonWeight (or None) per vowel position in the word.
    Diphthongs like [ae] count as one position (always LONG).
    """
    lengths: list[PhonWeight | None] = []
    i = 0
    while i < len(form):
        ch = form[i]

        # Diphthong: [ae], [oe], etc.
        if ch == "[":
            end = form.index("]", i)
            lengths.append(PhonWeight.LONG)  # diphthongs are always long
            i = end + 1
            continue

        # Vowel
        if ch in _MACRON_VOWELS:
            # Check the mark after the vowel
            if i + 1 < len(form):
                mark = form[i + 1]
                if mark == "_":
                    lengths.append(PhonWeight.LONG)
                    i += 2
                    continue
                elif mark == "^":
                    lengths.append(PhonWeight.SHORT)
                    i += 2
                    continue
                elif mark == "*":
                    lengths.append(None)  # anceps
                    i += 2
                    continue

            # No mark after vowel → ambiguous
            lengths.append(None)
            i += 1
            continue

        # Consonant (j, v, qv, etc.) — skip
        i += 1

    return lengths


class VowelLengthLexicon:
    """Lookup natural vowel lengths for word forms.

    Primary source: MQDQ frequency-based dictionary (empirical).
    Fallback: Morpheus lexicon (theory-based).
    """

    def __init__(
        self,
        mqdq_path: Path | None = None,
        morpheus_path: Path | None = None,
    ) -> None:
        self._mqdq: dict[str, dict[str, dict[str, int]]] = {}
        self._morpheus: dict[str, list[PhonWeight | None]] = {}

        if mqdq_path and mqdq_path.exists():
            with open(mqdq_path) as f:
                self._mqdq = json.load(f)

        if morpheus_path and morpheus_path.exists():
            self._morpheus = self._load_morpheus(morpheus_path)

    @staticmethod
    def _load_morpheus(path: Path) -> dict[str, list[PhonWeight | None]]:
        """Load Morpheus macron data.

        Format: word_form  morph_tag  lemma  macronized_form
        Where _ = long, ^ = short after each vowel.
        """
        result: dict[str, list[PhonWeight | None]] = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                word_form = parts[0].lower()
                macro_form = parts[3]
                if word_form not in result:
                    result[word_form] = _parse_macron_form(macro_form.lower())
        return result

    def lookup(
        self,
        word: str,
        author: str = "",
    ) -> list[PhonWeight | None] | None:
        """Return natural vowel lengths for each vowel in the word.

        Returns None if the word is not in any dictionary.
        Returns a list with one entry per vowel: LONG, SHORT, or None (ambiguous).

        Uses MQDQ majority vote across attestations.
        Falls back to Morpheus if MQDQ has no entry.
        """
        word_lower = word.lower()

        # Try MQDQ first
        if word_lower in self._mqdq:
            return self._lookup_mqdq(word_lower, author)

        # Fallback to Morpheus
        if word_lower in self._morpheus:
            return self._morpheus[word_lower]

        return None

    def _lookup_mqdq(
        self,
        word: str,
        author: str,
        threshold: float = 0.85,
    ) -> list[PhonWeight | None]:
        """Lookup in MQDQ dictionary using per-vowel consensus.

        For each vowel position, aggregates across all macronized variants
        weighted by frequency. A vowel is assigned LONG or SHORT only if
        that length accounts for >= threshold of total attestations.
        Otherwise it's left as None (ambiguous).
        """
        variants = self._mqdq[word]

        # Parse all variants and weight by frequency
        parsed_variants: list[tuple[list[PhonWeight | None], int]] = []
        for macro_form, author_counts in variants.items():
            if author:
                freq = author_counts.get(author, 0)
            else:
                freq = sum(author_counts.values())
            if freq > 0:
                lengths = _parse_macron_form(macro_form)
                parsed_variants.append((lengths, freq))

        if not parsed_variants:
            return [None]

        # Find the max vowel count across variants
        max_vowels = max(len(lengths) for lengths, _ in parsed_variants)
        if max_vowels == 0:
            return [None]

        # Per-vowel consensus
        result: list[PhonWeight | None] = []
        for pos in range(max_vowels):
            long_count = 0
            short_count = 0
            total_count = 0

            for lengths, freq in parsed_variants:
                if pos >= len(lengths):
                    continue
                total_count += freq
                v = lengths[pos]
                if v == PhonWeight.LONG:
                    long_count += freq
                elif v == PhonWeight.SHORT:
                    short_count += freq
                # None (anceps) doesn't count toward either

            if total_count == 0:
                result.append(None)
            elif long_count / total_count >= threshold:
                result.append(PhonWeight.LONG)
            elif short_count / total_count >= threshold:
                result.append(PhonWeight.SHORT)
            else:
                result.append(None)  # ambiguous

        return result

    def lookup_aligned(
        self,
        word: str,
        atom_vowels: list[str],
        author: str = "",
    ) -> list[PhonWeight | None]:
        """Lookup vowel lengths aligned to a specific atom vowel sequence.

        Instead of returning lengths by position in the dictionary's
        vowel counting scheme (which may differ from our atomizer's),
        this aligns by walking both the dictionary's macronized form
        and our atom vowels, matching by character identity.

        Returns one PhonWeight per atom_vowel entry.
        """
        word_lower = word.lower()

        # Get the best macronized form
        if word_lower in self._mqdq:
            best_form = self._get_best_mqdq_form(word_lower, author)
            if best_form:
                return self._align_to_atoms(best_form, atom_vowels)

        if word_lower in self._morpheus:
            # Morpheus doesn't give us the raw form, just parsed lengths.
            # Fall back to positional alignment.
            lengths = self._morpheus[word_lower]
            result: list[PhonWeight | None] = []
            for i in range(len(atom_vowels)):
                if i < len(lengths):
                    result.append(lengths[i])
                else:
                    result.append(None)
            return result

        return [None] * len(atom_vowels)

    def _get_best_mqdq_form(self, word: str, author: str) -> str | None:
        """Get the highest-frequency macronized form for a word."""
        variants = self._mqdq[word]
        best_form = None
        best_freq = 0
        for macro_form, author_counts in variants.items():
            freq = author_counts.get(author, 0) if author else sum(author_counts.values())
            if freq > best_freq:
                best_freq = freq
                best_form = macro_form
        return best_form

    @staticmethod
    def _align_to_atoms(
        macro_form: str,
        atom_vowels: list[str],
    ) -> list[PhonWeight | None]:
        """Align a macronized form's vowel lengths to our atom vowels.

        Walks the macronized form character by character. When a vowel
        character matches the next expected atom vowel, assigns its
        length mark to that atom. Diphthongs in brackets [ae] are
        matched against consecutive a + e atoms if present.
        """
        result: list[PhonWeight | None] = [None] * len(atom_vowels)
        atom_idx = 0
        i = 0

        while i < len(macro_form) and atom_idx < len(atom_vowels):
            ch = macro_form[i]

            # Diphthong [ae], [oe], etc.
            if ch == "[":
                end = macro_form.index("]", i)
                diph_chars = macro_form[i + 1:end]  # e.g. "ae"
                # Try to match against atom vowels
                if (atom_idx < len(atom_vowels)
                        and len(diph_chars) >= 2
                        and atom_vowels[atom_idx] == diph_chars[0]):
                    result[atom_idx] = PhonWeight.LONG  # first of diphthong
                    atom_idx += 1
                    if (atom_idx < len(atom_vowels)
                            and atom_vowels[atom_idx] == diph_chars[1]):
                        result[atom_idx] = PhonWeight.LONG  # second of diphthong
                        atom_idx += 1
                i = end + 1
                continue

            # Regular vowel
            if ch in _MACRON_VOWELS:
                # Read the mark
                mark: PhonWeight | None = None
                if i + 1 < len(macro_form) and macro_form[i + 1] in "_^*":
                    m = macro_form[i + 1]
                    if m == "_":
                        mark = PhonWeight.LONG
                    elif m == "^":
                        mark = PhonWeight.SHORT
                    # * = anceps → None
                    i += 2
                else:
                    i += 1

                # Try to match to current atom vowel
                if atom_idx < len(atom_vowels) and atom_vowels[atom_idx] == ch:
                    result[atom_idx] = mark
                    atom_idx += 1
                # If doesn't match (e.g., Morpheus counts a vowel we treat as consonantal),
                # skip this dictionary vowel
                continue

            # Consonant — skip
            i += 1

        return result

    @property
    def size(self) -> int:
        """Number of word forms in the lexicon."""
        return len(self._mqdq) + len(self._morpheus)
