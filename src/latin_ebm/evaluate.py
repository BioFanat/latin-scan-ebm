"""Evaluation harness: metrics, splits, and baselines.

Provides book-held-out and random-line splitting, plus metrics for
line-exact-match, foot-pattern accuracy, syllable accuracy, caesura
accuracy, and per-phenomenon F1 on ambiguous sites.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from latin_ebm.types import (
    LatinLine,
    Parse,
    SiteChoice,
    SiteType,
    TrainingExample,
)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------


def book_split(
    examples: list[TrainingExample],
    test_books: tuple[str, ...] = ("11", "12"),
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split by book: hold out specified books for test."""
    train = [ex for ex in examples if ex.line.book not in test_books]
    test = [ex for ex in examples if ex.line.book in test_books]
    return train, test


def random_split(
    examples: list[TrainingExample],
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Random split into train/dev/test."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]
    return train, dev, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Evaluation results."""
    line_exact_match: float = 0.0
    foot_pattern_accuracy: float = 0.0
    syllable_accuracy: float = 0.0
    caesura_accuracy: float = 0.0

    # Per-phenomenon F1 on ambiguous sites
    elision_f1: float = 0.0
    synizesis_f1: float = 0.0
    diphthong_f1: float = 0.0
    mcl_f1: float = 0.0

    # Breakdown
    per_book: dict[str, float] = field(default_factory=dict)
    n_test: int = 0


def _f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from true positives, false positives, false negatives."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    predictions: list[tuple[LatinLine, Parse, Parse]],
) -> EvalResult:
    """Evaluate predictions against gold.

    predictions: list of (line, predicted_parse, gold_parse) triples.
    """
    n = len(predictions)
    if n == 0:
        return EvalResult()

    foot_correct = 0
    line_exact = 0
    total_syllables = 0
    correct_syllables = 0
    caesura_correct = 0

    # Per-phenomenon counts: tp, fp, fn
    phenomenon_counts: dict[str, list[int]] = {
        "elision": [0, 0, 0],
        "synizesis": [0, 0, 0],
        "diphthong": [0, 0, 0],
        "mcl": [0, 0, 0],
    }

    per_book_correct: dict[str, int] = {}
    per_book_total: dict[str, int] = {}

    for line, predicted, gold in predictions:
        # Foot pattern accuracy
        if predicted.foot_types == gold.foot_types:
            foot_correct += 1

        # Line exact match: foot types + slots must match
        if predicted.foot_types == gold.foot_types and predicted.slots == gold.slots:
            line_exact += 1

        # Syllable accuracy (compare slot sequences, which encode quantities)
        min_len = min(len(predicted.slots), len(gold.slots))
        total_syllables += len(gold.slots)
        for i in range(min_len):
            if predicted.slots[i] == gold.slots[i]:
                correct_syllables += 1

        # Caesura accuracy
        if predicted.caesura == gold.caesura:
            caesura_correct += 1

        # Per-phenomenon F1
        _score_phenomenon(
            line, predicted, gold,
            SiteType.ELISION, SiteChoice.ELIDE,
            phenomenon_counts["elision"],
        )
        _score_phenomenon(
            line, predicted, gold,
            SiteType.SYNIZESIS, SiteChoice.MERGE,
            phenomenon_counts["synizesis"],
        )
        _score_phenomenon(
            line, predicted, gold,
            SiteType.DIPHTHONG_SPLIT, SiteChoice.SPLIT,
            phenomenon_counts["diphthong"],
        )
        _score_phenomenon(
            line, predicted, gold,
            SiteType.MUTA_CUM_LIQUIDA, SiteChoice.CLOSE,
            phenomenon_counts["mcl"],
        )

        # Per-book tracking
        book = line.book
        per_book_total[book] = per_book_total.get(book, 0) + 1
        if predicted.foot_types == gold.foot_types:
            per_book_correct[book] = per_book_correct.get(book, 0) + 1

    per_book = {
        book: per_book_correct.get(book, 0) / per_book_total[book]
        for book in per_book_total
    }

    return EvalResult(
        line_exact_match=line_exact / n,
        foot_pattern_accuracy=foot_correct / n,
        syllable_accuracy=correct_syllables / total_syllables if total_syllables > 0 else 0.0,
        caesura_accuracy=caesura_correct / n,
        elision_f1=_f1(*phenomenon_counts["elision"]),
        synizesis_f1=_f1(*phenomenon_counts["synizesis"]),
        diphthong_f1=_f1(*phenomenon_counts["diphthong"]),
        mcl_f1=_f1(*phenomenon_counts["mcl"]),
        per_book=per_book,
        n_test=n,
    )


def _score_phenomenon(
    line: LatinLine,
    predicted: Parse,
    gold: Parse,
    site_type: SiteType,
    active_choice: SiteChoice,
    counts: list[int],
) -> None:
    """Update tp/fp/fn counts for a specific phenomenon.

    A "positive" is when the active_choice is made (e.g., ELIDE for elision).
    """
    for site in line.sites:
        if site.site_type != site_type:
            continue

        pred_decision = predicted.decisions.get(site.index, site.default)
        gold_decision = gold.decisions.get(site.index, site.default)

        pred_active = (pred_decision == active_choice)
        gold_active = (gold_decision == active_choice)

        if pred_active and gold_active:
            counts[0] += 1  # tp
        elif pred_active and not gold_active:
            counts[1] += 1  # fp
        elif not pred_active and gold_active:
            counts[2] += 1  # fn


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def default_baseline(
    line: LatinLine,
    candidates: list[Parse],
) -> Parse:
    """Baseline: take all default decisions, pick first compatible parse."""
    # Find candidate with all-default decisions
    for c in candidates:
        all_default = all(
            c.decisions.get(site.index, site.default) == site.default
            for site in line.sites
        )
        if all_default:
            return c

    # Fallback: first candidate
    return candidates[0] if candidates else None  # type: ignore[return-value]


def random_baseline(
    candidates: list[Parse],
    rng: random.Random | None = None,
) -> Parse:
    """Baseline: pick a random valid parse."""
    if rng is None:
        rng = random.Random(42)
    return rng.choice(candidates)
