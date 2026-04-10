#!/usr/bin/env python3
"""Train v1 linear EBM on Vergil's Aeneid and evaluate.

Usage:
    python scripts/train_v1.py [--xml PATH] [--test-books 11,12] [--l2 0.01]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import book_split, evaluate, default_baseline, random_baseline, EvalResult
from latin_ebm.meters import Hexameter
from latin_ebm.train import train_nll

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def print_eval(name: str, result: EvalResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name} (n={result.n_test})")
    print(f"{'=' * 60}")
    print(f"  Line exact match:     {result.line_exact_match:.1%}")
    print(f"  Foot pattern acc:     {result.foot_pattern_accuracy:.1%}")
    print(f"  Syllable accuracy:    {result.syllable_accuracy:.1%}")
    print(f"  Caesura accuracy:     {result.caesura_accuracy:.1%}")
    print(f"  Elision F1:           {result.elision_f1:.3f}")
    print(f"  Synizesis F1:         {result.synizesis_f1:.3f}")
    print(f"  Diphthong F1:         {result.diphthong_f1:.3f}")
    print(f"  MCL F1:               {result.mcl_f1:.3f}")
    if result.per_book:
        print("\n  Per-book foot accuracy:")
        for book in sorted(result.per_book, key=lambda b: int(b) if b.isdigit() else 0):
            print(f"    Book {book:>2s}: {result.per_book[book]:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v1 linear EBM")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml",
        help="Path to Pedecerto XML file",
    )
    parser.add_argument(
        "--test-books",
        default="11,12",
        help="Comma-separated book numbers to hold out for test",
    )
    parser.add_argument("--l2", type=float, default=0.01, help="L2 regularization strength")
    parser.add_argument("--max-iter", type=int, default=200, help="Max L-BFGS iterations")
    args = parser.parse_args()

    # 1. Load corpus
    logger.info("Loading corpus from %s", args.xml)
    result = parse_xml(args.xml)
    logger.info("Loaded %d examples (%d skipped)", len(result.examples), result.skipped)

    # 2. Split
    test_books = tuple(args.test_books.split(","))
    train_examples, test_examples = book_split(result.examples, test_books)
    logger.info("Train: %d lines, Test: %d lines (books %s)", len(train_examples), len(test_examples), test_books)

    # 3. Train
    t0 = time.time()
    model, train_result = train_nll(
        train_examples,
        l2_lambda=args.l2,
        max_iter=args.max_iter,
    )
    t1 = time.time()
    logger.info("Training took %.1fs", t1 - t0)
    logger.info("Final loss: %.4f, iterations: %d, converged: %s",
                train_result.final_loss, train_result.n_iterations, train_result.converged)

    # 4. Evaluate on test set
    logger.info("Evaluating on test set...")
    meter = Hexameter()

    # Model predictions
    model_predictions = []
    baseline_default_predictions = []
    baseline_random_predictions = []

    import random
    rng = random.Random(42)

    for ex in test_examples:
        candidates = enumerate_parses(ex.line, meter)
        if not candidates:
            continue

        pred = model.predict(ex.line, candidates)
        model_predictions.append((ex.line, pred, ex.gold_parse))

        default_pred = default_baseline(ex.line, candidates)
        if default_pred:
            baseline_default_predictions.append((ex.line, default_pred, ex.gold_parse))

        random_pred = random_baseline(candidates, rng)
        baseline_random_predictions.append((ex.line, random_pred, ex.gold_parse))

    # Compute metrics
    model_eval = evaluate(model_predictions)
    default_eval = evaluate(baseline_default_predictions)
    random_eval = evaluate(baseline_random_predictions)

    print_eval("Linear EBM (v1)", model_eval)
    print_eval("Default Baseline", default_eval)
    print_eval("Random Baseline", random_eval)

    # 5. Show top features
    print(f"\n{'=' * 60}")
    print("  Top 20 features by |weight|")
    print(f"{'=' * 60}")
    top_indices = np.argsort(np.abs(model.theta))[::-1][:20]
    names = model.feature_index.names
    for idx in top_indices:
        if idx < len(names):
            print(f"  {model.theta[idx]:+.4f}  {names[idx]}")


if __name__ == "__main__":
    import numpy as np  # noqa: E402
    main()
