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
    SiteChoice,
    SiteType,
)

if TYPE_CHECKING:
    from latin_ebm.lexicon import VowelLengthLexicon


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
