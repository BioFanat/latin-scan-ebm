"""Feature extraction: φ(x, y) for energy computation.

v1 features are all sparse binary/count indicators, concatenated into
a single flat numpy array. A FeatureIndex registry maps feature names
to column indices.

Optionally includes MQDQ lexicon features: for each word in the line,
what scansion pattern the MQDQ corpus most commonly assigns. This is
soft evidence (a training feature), not a hard constraint.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from latin_ebm.types import (
    FootType,
    LatinLine,
    Parse,
    PhonWeight,
    SiteChoice,
    SiteType,
)

if TYPE_CHECKING:
    from latin_ebm.lexicon import VowelLengthLexicon


# Optional allowlist for lemma features (set to None = allow all; set to a frozenset
# of allowed lemmas to gate). Used to reduce feature-index blowup.
_LEMMA_ALLOWLIST: frozenset[str] | None = None


def set_lemma_allowlist(allowlist: frozenset[str] | None) -> None:
    """Set the allowlist used to gate lemma features."""
    global _LEMMA_ALLOWLIST
    _LEMMA_ALLOWLIST = allowlist


def _syllable_index_for_atom(parse: Parse, atom_idx: int) -> int | None:
    """Find which realized syllable contains the given atom_idx (or None)."""
    for syl_idx, syl in enumerate(parse.syllables):
        if atom_idx in syl.atom_indices:
            return syl_idx
    return None


def _foot_for_syllable(parse: Parse, syl_idx: int | None) -> int | None:
    """Which foot contains syllable syl_idx (0..n_feet-1) or None."""
    if syl_idx is None:
        return None
    bnds = parse.foot_boundaries
    n_feet = len(parse.foot_types)
    for foot_idx in range(n_feet):
        start = bnds[foot_idx]
        end = bnds[foot_idx + 1] if foot_idx + 1 < len(bnds) else len(parse.syllables)
        if start <= syl_idx < end:
            return foot_idx
    return None


class FeatureIndex:
    """Registry mapping feature names to column indices.

    Built from training data in a first pass, then frozen for
    feature extraction during training and inference.
    """

    def __init__(self) -> None:
        self._name_to_idx: dict[str, int] = {}
        self._frozen = False

    @property
    def n_features(self) -> int:
        return len(self._name_to_idx)

    def get_or_add(self, name: str) -> int:
        """Get the index for a feature name, adding it if not frozen."""
        if name in self._name_to_idx:
            return self._name_to_idx[name]
        if self._frozen:
            return -1  # unknown feature after freezing
        idx = len(self._name_to_idx)
        self._name_to_idx[name] = idx
        return idx

    def freeze(self) -> None:
        """Freeze the index — no new features can be added."""
        self._frozen = True

    def get(self, name: str) -> int:
        """Get index for a known feature. Returns -1 if unknown."""
        return self._name_to_idx.get(name, -1)

    @property
    def names(self) -> list[str]:
        """All feature names in index order."""
        items = sorted(self._name_to_idx.items(), key=lambda x: x[1])
        return [name for name, _ in items]


def extract_features(
    line: LatinLine,
    parse: Parse,
    index: FeatureIndex,
    lexicon: VowelLengthLexicon | None = None,
) -> np.ndarray:
    """Extract a feature vector φ(x, y) for a (line, parse) pair.

    Features are sparse binary/count indicators grouped into:
    - Site features (per ambiguity site decision)
    - Global features (per parse)
    """
    features: dict[str, float] = defaultdict(float)

    # --- Site-level features ---
    for site in line.sites:
        decision = parse.decisions.get(site.index, site.default)

        # Decision type × site type
        feat = f"site:{site.site_type.name}:{decision.name}"
        features[feat] += 1.0

        # Vowel identity at site
        if site.atom_indices:
            left_atom = line.atoms[site.atom_indices[0]]
            features[f"site_vowel:{left_atom.chars}:{decision.name}"] += 1.0

        # --- Phase B/C/D: contextual site features ---
        n_atoms = len(line.atoms)
        if site.site_type == SiteType.ELISION and len(site.atom_indices) >= 2:
            li, ri = site.atom_indices[0], site.atom_indices[1]
            la = line.atoms[li]
            ra = line.atoms[ri]
            pair = la.chars + ra.chars
            features[f"elision_pair:{pair}:{decision.name}"] += 1.0

            # bridge between the two atoms tells us about coda + nasal-m pattern
            if 0 <= li < len(line.bridges):
                bridge = line.bridges[li]
                # nasal-m elision (final m before vowel) vs pure vowel hiatus
                has_m = "m" in bridge.chars.lower()
                features[f"elision_has_m:{int(has_m)}:{decision.name}"] += 1.0

            # which word follows? prodelision sensitive to e/es/est
            if ra.word_idx < len(line.words):
                right_word = line.words[ra.word_idx].lower()
                if right_word in ("es", "est"):
                    features[f"elision_right_es:{decision.name}"] += 1.0

            # which foot does the elision fall in?
            syl_idx_el = _syllable_index_for_atom(parse, li)
            foot_el = _foot_for_syllable(parse, syl_idx_el)
            if foot_el is not None:
                features[f"elision_in_foot:{foot_el}:{decision.name}"] += 1.0

            # left word ending pattern (last 2 chars of left word)
            if la.word_idx < len(line.words):
                lw = line.words[la.word_idx].lower()
                if len(lw) >= 2:
                    features[f"elision_lw_end:{lw[-2:]}:{decision.name}"] += 1.0

        elif site.site_type == SiteType.SYNIZESIS and len(site.atom_indices) >= 2:
            li, ri = site.atom_indices[0], site.atom_indices[1]
            pair = line.atoms[li].chars + line.atoms[ri].chars
            features[f"synizesis_pair:{pair}:{decision.name}"] += 1.0
            syl_idx_s = _syllable_index_for_atom(parse, li)
            foot_s = _foot_for_syllable(parse, syl_idx_s)
            if foot_s is not None:
                features[f"synizesis_in_foot:{foot_s}:{decision.name}"] += 1.0

        elif site.site_type == SiteType.DIPHTHONG_SPLIT and len(site.atom_indices) >= 2:
            li, ri = site.atom_indices[0], site.atom_indices[1]
            pair = line.atoms[li].chars + line.atoms[ri].chars
            features[f"diphthong_pair:{pair}:{decision.name}"] += 1.0
            syl_idx_d = _syllable_index_for_atom(parse, li)
            foot_d = _foot_for_syllable(parse, syl_idx_d)
            if foot_d is not None:
                features[f"diphthong_in_foot:{foot_d}:{decision.name}"] += 1.0

        elif site.site_type == SiteType.MUTA_CUM_LIQUIDA and site.atom_indices:
            li = site.atom_indices[0]
            if 0 <= li < len(line.bridges):
                bridge = line.bridges[li].chars.lower()
                for stop in "bpdtcgf":
                    for liquid in "lr":
                        if stop + liquid in bridge:
                            features[f"mcl_cluster:{stop+liquid}:{decision.name}"] += 1.0
                            break
            syl_idx_m = _syllable_index_for_atom(parse, li)
            foot_m = _foot_for_syllable(parse, syl_idx_m)
            if foot_m is not None:
                features[f"mcl_in_foot:{foot_m}:{decision.name}"] += 1.0

        elif site.site_type == SiteType.PRODELISION and len(site.atom_indices) >= 2:
            li, ri = site.atom_indices[0], site.atom_indices[1]
            pair = line.atoms[li].chars + line.atoms[ri].chars
            features[f"prodelision_pair:{pair}:{decision.name}"] += 1.0

        # Lemma-conditional site feature (gated via _LEMMA_ALLOWLIST when set)
        if lexicon is not None and site.atom_indices:
            la = line.atoms[site.atom_indices[0]]
            if la.word_idx < len(line.words):
                lem = lexicon.lemma(line.words[la.word_idx])
                if lem is not None and len(lem) >= 2:
                    if _LEMMA_ALLOWLIST is None or lem in _LEMMA_ALLOWLIST:
                        features[f"lemma:{lem}:{site.site_type.name}:{decision.name}"] += 1.0

    # --- Global features ---

    # Fifth foot type
    if len(parse.foot_types) >= 6:
        ft5 = parse.foot_types[4]
        features[f"foot5:{ft5.name}"] = 1.0

    # Caesura type
    features[f"caesura:{parse.caesura.name}"] = 1.0

    # Bucolic diaeresis
    if parse.bucolic_diaeresis:
        features["bucolic_diaeresis"] = 1.0

    # Elision count
    n_elisions = sum(
        1 for site in line.sites
        if site.site_type == SiteType.ELISION
        and parse.decisions.get(site.index, site.default) == SiteChoice.ELIDE
    )
    features[f"elision_count:{min(n_elisions, 3)}"] = 1.0

    # Spondee and dactyl counts among feet 1-5
    if len(parse.foot_types) >= 6:
        n_spondees = sum(
            1 for ft in parse.foot_types[:5] if ft == FootType.SPONDEE
        )
        n_dactyls = sum(
            1 for ft in parse.foot_types[:5] if ft == FootType.DACTYL
        )
        features[f"spondee_count:{n_spondees}"] = 1.0
        features[f"dactyl_count:{n_dactyls}"] = 1.0

    # Syllable count
    features[f"syllable_count:{len(parse.syllables)}"] = 1.0

    # Foot pattern (the full pattern as a single feature)
    pattern = "".join(
        "D" if ft == FootType.DACTYL else "S" if ft == FootType.SPONDEE else "F"
        for ft in parse.foot_types
    )
    features[f"pattern:{pattern}"] = 1.0

    # --- Phase B: per-syllable weight × position ---
    for syl_idx, syl in enumerate(parse.syllables):
        features[f"syl_pos:{syl_idx}:{syl.weight.name}"] += 1.0

    # --- Phase B: foot bigrams ---
    for i in range(len(parse.foot_types) - 1):
        ft1 = parse.foot_types[i].name
        ft2 = parse.foot_types[i + 1].name
        features[f"foot_bigram:{i}:{ft1}_{ft2}"] += 1.0

    # --- Phase B: per-foot foot-type features (feet 0..4) ---
    for i, ft in enumerate(parse.foot_types[:5]):
        features[f"foot:{i}:{ft.name}"] += 1.0

    # --- Phase B: author-conditioning (cheap; helps if multi-author) ---
    if line.author:
        features[f"author:{line.author}"] = 1.0

    # --- Lexicon features (MQDQ word-level scansion priors) ---
    if lexicon is not None:
        # For each word, check if the parse's weight assignment
        # agrees with the MQDQ majority scansion for that word.
        # This is soft evidence: a feature, not a constraint.
        n_agree = 0
        n_disagree = 0
        n_checked = 0

        for word_idx, word in enumerate(line.words):
            # Get atoms for this word
            word_atoms = [a for a in line.atoms if a.word_idx == word_idx]
            if not word_atoms:
                continue

            atom_vowels = [a.chars for a in word_atoms]
            lex_lengths = lexicon.lookup_aligned(word, atom_vowels)

            # Find syllables that contain atoms from this word
            for atom in word_atoms:
                # Find the syllable containing this atom
                for syl_idx, syl in enumerate(parse.syllables):
                    if atom.index in syl.atom_indices:
                        lex_len = lex_lengths[atom_vowels.index(atom.chars)] if atom.chars in atom_vowels else None
                        if lex_len is not None:
                            n_checked += 1
                            if syl.weight == lex_len:
                                n_agree += 1
                            else:
                                n_disagree += 1
                        break

        if n_checked > 0:
            features["lex_agree_ratio"] = n_agree / n_checked
            features["lex_disagree_ratio"] = n_disagree / n_checked
            features[f"lex_disagree_count:{min(n_disagree, 4)}"] = 1.0

    # --- Convert to numpy array ---
    vec = np.zeros(index.n_features if index._frozen else max(index.n_features, len(features) + 100))

    for name, value in features.items():
        idx = index.get_or_add(name)
        if idx >= 0:
            if idx >= len(vec):
                vec = np.pad(vec, (0, idx - len(vec) + 1))
            vec[idx] = value

    # Trim to exact size if frozen
    if index._frozen:
        return vec[:index.n_features]
    return vec[:index.n_features]


def build_feature_index(
    lines: list[LatinLine],
    parses_per_line: list[list[Parse]],
    lexicon: VowelLengthLexicon | None = None,
) -> FeatureIndex:
    """Build a FeatureIndex by extracting features from all training candidates.

    First pass: collect all feature names that fire on any training candidate.
    """
    index = FeatureIndex()

    for line, parses in zip(lines, parses_per_line):
        for parse in parses:
            extract_features(line, parse, index, lexicon=lexicon)

    index.freeze()
    return index
