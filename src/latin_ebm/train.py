"""Training objectives for the scansion EBM.

v1: Exact conditional log-likelihood (NLL) with tractable partition function.
The partition function Z(x) is a finite sum over the enumerated candidate set,
which is the key computational advantage of this architecture.

Supports partial supervision: when some gold decisions are unobserved,
marginalize over all compatible candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from latin_ebm.energy import LinearEBM
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import FeatureIndex, extract_features, build_feature_index
from latin_ebm.meters import Hexameter
from latin_ebm.types import (
    LatinLine,
    Parse,
    TrainingExample,
)

if TYPE_CHECKING:
    from latin_ebm.lexicon import VowelLengthLexicon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precomputed training data
# ---------------------------------------------------------------------------


@dataclass
class PrecomputedLine:
    """Precomputed candidates and features for one training line."""
    line: LatinLine
    gold_parse: Parse
    candidates: list[Parse]
    features: np.ndarray        # (n_candidates, n_features)
    gold_indices: list[int]     # indices of gold-compatible candidates


def precompute_training_data(
    examples: list[TrainingExample],
    feature_index: FeatureIndex,
    meter: Hexameter | None = None,
    lexicon: VowelLengthLexicon | None = None,
) -> list[PrecomputedLine]:
    """Precompute candidates and features for all training lines.

    This is the expensive step — done once before optimization.
    """
    if meter is None:
        meter = Hexameter()

    result: list[PrecomputedLine] = []
    skipped = 0

    for i, ex in enumerate(examples):
        candidates = enumerate_parses(ex.line, meter)
        if not candidates:
            skipped += 1
            continue

        # Extract features for all candidates
        feat_list = [
            extract_features(ex.line, c, feature_index, lexicon=lexicon)
            for c in candidates
        ]
        features = np.array(feat_list)

        # Find gold-compatible candidates
        gold = ex.gold_parse
        gold_indices: list[int] = []
        for j, c in enumerate(candidates):
            if c.foot_types == gold.foot_types and c.slots == gold.slots:
                gold_indices.append(j)

        if not gold_indices:
            skipped += 1
            continue

        result.append(PrecomputedLine(
            line=ex.line,
            gold_parse=gold,
            candidates=candidates,
            features=features,
            gold_indices=gold_indices,
        ))

        if (i + 1) % 500 == 0:
            logger.info("Precomputed %d/%d lines (%d skipped)", i + 1, len(examples), skipped)

    logger.info(
        "Precomputed %d lines, skipped %d (no candidates or no gold match)",
        len(result), skipped,
    )
    return result


# ---------------------------------------------------------------------------
# NLL loss and gradient
# ---------------------------------------------------------------------------


def nll_loss_and_grad(
    theta: np.ndarray,
    data: list[PrecomputedLine],
    l2_lambda: float = 0.01,
) -> tuple[float, np.ndarray]:
    """Compute total NLL loss and gradient over the training set.

    For each line:
        L = -log p(gold | x)
          = -log [Σ_{y∈C(gold)} exp(-E(y))] + log Z(x)

    where C(gold) is the set of candidates compatible with gold,
    and Z(x) = Σ_y exp(-E(x,y)).

    With partial supervision, C(gold) may contain multiple candidates.
    """
    total_loss = 0.0
    total_grad = np.zeros_like(theta)

    for item in data:
        # Energies for all candidates
        energies = item.features @ theta  # (n_candidates,)

        # Log partition function: log Z = logsumexp(-energies)
        log_z = logsumexp(-energies)

        # Log numerator: log Σ_{y∈C} exp(-E(y))
        gold_energies = energies[item.gold_indices]
        log_num = logsumexp(-gold_energies)

        # NLL for this line
        loss = -log_num + log_z
        total_loss += loss

        # Gradient: ∇L = E_q[φ] - E_p[φ]
        # where q = uniform over gold-compatible, p = model distribution over all
        all_probs = np.exp(-energies - log_z)  # p(y|x) for all candidates
        gold_probs = np.exp(-gold_energies - log_num)  # normalized within gold set

        # Expected features under model
        expected_all = all_probs @ item.features

        # Expected features under gold-compatible
        gold_features = item.features[item.gold_indices]
        expected_gold = gold_probs @ gold_features

        total_grad += expected_gold - expected_all

    # L2 regularization
    total_loss += l2_lambda * np.sum(theta ** 2)
    total_grad += 2 * l2_lambda * theta

    return total_loss, total_grad


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Result of training."""
    theta: np.ndarray
    final_loss: float
    n_iterations: int
    converged: bool


def train_nll(
    examples: list[TrainingExample],
    feature_index: FeatureIndex | None = None,
    meter: Hexameter | None = None,
    l2_lambda: float = 0.01,
    max_iter: int = 200,
    lexicon: VowelLengthLexicon | None = None,
) -> tuple[LinearEBM, TrainResult]:
    """Train a linear EBM with exact NLL.

    Steps:
    1. Enumerate candidates for all training lines
    2. Build feature index (if not provided)
    3. Extract features (cached)
    4. Optimize θ with L-BFGS

    Returns the trained model and training statistics.
    """
    if meter is None:
        meter = Hexameter()

    # Build feature index from training data
    logger.info("Enumerating candidates for feature index...")
    if feature_index is None:
        lines = [ex.line for ex in examples]
        parses_per_line = [enumerate_parses(ex.line, meter) for ex in examples]
        feature_index = build_feature_index(lines, parses_per_line, lexicon=lexicon)
        logger.info("Feature index built: %d features", feature_index.n_features)

    # Precompute
    logger.info("Precomputing training data...")
    data = precompute_training_data(examples, feature_index, meter, lexicon=lexicon)
    logger.info("Training on %d lines with %d features", len(data), feature_index.n_features)

    # Initial theta
    theta0 = np.zeros(feature_index.n_features)

    # Optimize with L-BFGS
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return nll_loss_and_grad(theta, data, l2_lambda)

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "disp": False},
    )

    # Build model
    model = LinearEBM(feature_index, lexicon=lexicon)
    model.theta = result.x

    train_result = TrainResult(
        theta=result.x,
        final_loss=float(result.fun),
        n_iterations=result.nit,
        converged=result.success,
    )

    logger.info(
        "Training complete: loss=%.4f, iterations=%d, converged=%s",
        train_result.final_loss, train_result.n_iterations, train_result.converged,
    )

    return model, train_result
