#!/usr/bin/env python3
"""Detailed failure-mode analysis: categorize wrong predictions and examine features."""

from __future__ import annotations

import argparse
import logging
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import book_split
from latin_ebm.meters import Hexameter
from latin_ebm.train import train_nll
from latin_ebm.types import SiteType, SiteChoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def categorize_line_error(line, predicted, gold):
    """Categorize a wrong prediction by what decision(s) differ from gold.
    
    Returns a dict with keys:
    - wrong_foot_pattern: bool
    - wrong_decisions: list of (site_type_name, gold_decision_name, pred_decision_name)
    """
    result = {
        "wrong_foot_pattern": predicted.foot_types != gold.foot_types,
        "wrong_decisions": [],
    }
    
    # Compare decisions at all sites
    for site in line.sites:
        gold_decision = gold.decisions.get(site.index, site.default)
        pred_decision = predicted.decisions.get(site.index, site.default)
        
        if gold_decision != pred_decision:
            result["wrong_decisions"].append({
                "site_type": site.site_type.name,
                "gold_decision": gold_decision.name,
                "pred_decision": pred_decision.name,
            })
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze failure modes in scansion")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml",
        help="Path to Pedecerto XML file",
    )
    parser.add_argument(
        "--test-books",
        default="1,2",
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
    logger.info("Train: %d lines, Test: %d lines", len(train_examples), len(test_examples))

    # 3. Train
    logger.info("Training model...")
    model, train_result = train_nll(
        train_examples,
        l2_lambda=args.l2,
        max_iter=args.max_iter,
    )
    logger.info("Training complete: loss=%.4f, iterations=%d", 
                train_result.final_loss, train_result.n_iterations)

    # 4. Evaluate and categorize errors
    meter = Hexameter()
    
    correct_lines = []
    wrong_reachable_lines = []  # gold in candidate set but model chose wrong
    unreachable_lines = []  # gold not in candidate set
    
    # Categorization counters
    decision_error_counts = Counter()  # e.g., "ELISION:ELIDE->RETAIN"
    foot_pattern_errors = 0
    decision_only_errors = 0  # foot pattern correct but decision wrong
    
    # Candidate set analysis
    candidate_counts = []
    gold_in_candidate = 0
    
    logger.info("Evaluating on test set...")
    for ex in test_examples:
        candidates = enumerate_parses(ex.line, meter)
        if not candidates:
            continue
        
        candidate_counts.append(len(candidates))
        
        # Check if gold is reachable
        gold_reachable = False
        gold_candidate_idx = None
        for j, c in enumerate(candidates):
            if c.foot_types == ex.gold_parse.foot_types and c.slots == ex.gold_parse.slots:
                gold_reachable = True
                gold_candidate_idx = j
                gold_in_candidate += 1
                break
        
        # Get prediction
        pred = model.predict(ex.line, candidates)
        
        # Categorize
        is_correct = (pred.foot_types == ex.gold_parse.foot_types and 
                     pred.slots == ex.gold_parse.slots)
        
        if is_correct:
            correct_lines.append((ex.line, pred, ex.gold_parse))
        elif gold_reachable:
            wrong_reachable_lines.append((ex.line, pred, ex.gold_parse))
            
            # Categorize the error
            error_info = categorize_line_error(ex.line, pred, ex.gold_parse)
            if error_info["wrong_foot_pattern"]:
                foot_pattern_errors += 1
            else:
                decision_only_errors += 1
            
            # Record decision errors
            for dec in error_info["wrong_decisions"]:
                key = f"{dec['site_type']}:{dec['gold_decision']}->{dec['pred_decision']}"
                decision_error_counts[key] += 1
        else:
            unreachable_lines.append(ex.line)
    
    # 5. Report statistics
    n_evaluated = len(correct_lines) + len(wrong_reachable_lines) + len(unreachable_lines)
    
    print(f"\n{'=' * 70}")
    print(f"FAILURE-MODE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total test lines evaluated: {n_evaluated}")
    print(f"  Correct: {len(correct_lines)} ({len(correct_lines)/n_evaluated*100:.1f}%)")
    print(f"  Wrong but reachable: {len(wrong_reachable_lines)} ({len(wrong_reachable_lines)/n_evaluated*100:.1f}%)")
    print(f"  Unreachable: {len(unreachable_lines)} ({len(unreachable_lines)/n_evaluated*100:.1f}%)")
    print()
    
    print(f"Gold reachability: {gold_in_candidate}/{n_evaluated} ({gold_in_candidate/n_evaluated*100:.1f}%)")
    print()
    
    print(f"Wrong reachable breakdown:")
    print(f"  Foot pattern wrong: {foot_pattern_errors} ({foot_pattern_errors/len(wrong_reachable_lines)*100:.1f}%)")
    print(f"  Foot pattern correct, decision(s) wrong: {decision_only_errors} ({decision_only_errors/len(wrong_reachable_lines)*100:.1f}%)")
    print()
    
    print(f"Candidate set sizes:")
    print(f"  Mean: {np.mean(candidate_counts):.1f}")
    print(f"  Median: {np.median(candidate_counts):.1f}")
    print(f"  Min: {np.min(candidate_counts)}, Max: {np.max(candidate_counts)}")
    print()
    
    print(f"Top decision-level errors (wrong_reachable lines):")
    for (dec_err, count) in decision_error_counts.most_common(15):
        pct = 100 * count / len(wrong_reachable_lines)
        print(f"  {dec_err}: {count} ({pct:.1f}%)")
    print()
    
    # 6. Show top features
    print(f"{'=' * 70}")
    print("Top 25 features by |weight|")
    print(f"{'=' * 70}")
    top_indices = np.argsort(np.abs(model.theta))[::-1][:25]
    names = model.feature_index.names
    for idx in top_indices:
        if idx < len(names):
            print(f"  {model.theta[idx]:+.4f}  {names[idx]}")
    print()
    
    # 7. Feature analysis: missing categories
    print(f"{'=' * 70}")
    print("Feature coverage analysis")
    print(f"{'=' * 70}")
    feature_categories = defaultdict(int)
    for name in names:
        # Extract category prefix
        if ":" in name:
            cat = name.split(":")[0]
        else:
            cat = name
        feature_categories[cat] += 1
    
    for cat in sorted(feature_categories.keys()):
        print(f"  {cat}: {feature_categories[cat]} features")
    print()
    
    # Check what's present and what's missing
    present_sets = set(feature_categories.keys())
    expected = {
        "site", "site_vowel", "foot5", "caesura", "elision_count",
        "spondee_count", "dactyl_count", "syllable_count", "pattern",
        "bucolic_diaeresis",  # may not be present if never active in training
    }
    missing = expected - present_sets
    if missing:
        print(f"Expected features not found: {missing}")
    print()


if __name__ == "__main__":
    main()
