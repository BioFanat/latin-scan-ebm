"""Energy-based model scoring.

v1: E_θ(x, y) = θᵀφ(x, y) — linear energy, one dot product.
E_foot is baked into enumeration (invalid structures never become candidates).
E_pair deferred to v2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from latin_ebm.features import FeatureIndex, extract_features
from latin_ebm.types import (
    LatinLine,
    Parse,
    ScoredParse,
)

if TYPE_CHECKING:
    from latin_ebm.lexicon import VowelLengthLexicon


class LinearEBM:
    """Linear energy-based model for Latin scansion.

    Energy is a simple dot product: E(x, y) = θᵀφ(x, y).
    Lower energy = more preferred parse.
    """

    def __init__(self, feature_index: FeatureIndex, lexicon: VowelLengthLexicon | None = None) -> None:
        self.feature_index = feature_index
        self.lexicon = lexicon
        self.theta = np.zeros(feature_index.n_features)

    def energy(self, features: np.ndarray) -> float:
        """Compute energy for a feature vector."""
        return float(self.theta @ features)

    def score_candidates(
        self,
        line: LatinLine,
        candidates: list[Parse],
    ) -> list[ScoredParse]:
        """Score all candidates for a line."""
        results: list[ScoredParse] = []
        for parse in candidates:
            feat = extract_features(line, parse, self.feature_index, lexicon=self.lexicon)
            e = self.energy(feat)
            results.append(ScoredParse(
                parse=parse,
                e_total=e,
                e_site=0.0,  # decomposition not tracked in v1
                e_syll=0.0,
                e_pair=0.0,
                e_foot=0.0,
                e_global=0.0,
            ))
        return results

    def predict(
        self,
        line: LatinLine,
        candidates: list[Parse],
    ) -> Parse:
        """Predict the best parse (lowest energy) for a line."""
        scored = self.score_candidates(line, candidates)
        return min(scored, key=lambda sp: sp.e_total).parse
