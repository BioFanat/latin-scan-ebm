"""Compute EBM accuracy on lines anceps abstained from."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_anceps import _norm_text, pattern_to_feet, load_anceps  # noqa: E402
from train_v3 import build_lemma_allowlist, fit_linear_adamw, fit_mlp_residual, precompute, predict_with  # noqa: E402

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import build_feature_index, set_lemma_allowlist
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.types import FootType


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--anceps", default="../anceps_output.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]

    allow = build_lemma_allowlist(train_ex, lexicon, min_count=3)
    set_lemma_allowlist(allow)

    lines = [ex.line for ex in train_ex]
    parses_per_line = [enumerate_parses(ex.line) for ex in train_ex]
    feature_index = build_feature_index(lines, parses_per_line, lexicon=lexicon)
    train_data = precompute(train_ex, feature_index, lexicon)
    test_data = precompute(test_ex, feature_index, lexicon)

    theta_np = fit_linear_adamw(train_data, feature_index.n_features, lr=5e-3, epochs=30, weight_decay=1e-3)
    theta_np, mlp = fit_mlp_residual(train_data, theta_np, mlp_hidden=64, lr=1e-3, epochs=30, finetune_linear=True, finetune_lr=5e-4)

    anc = load_anceps(Path(args.anceps))

    on_anc_matched = []
    off_anc_lines = []
    ebm_total_correct = 0
    for pre in test_data:
        idx = predict_with(pre, theta_np, mlp)
        ebm_feet = pre.candidates[idx].foot_types
        gold_feet = pre.gold_parse.foot_types
        correct = ebm_feet == gold_feet
        if correct:
            ebm_total_correct += 1
        key = _norm_text(pre.line.raw)
        anc_feet = anc.get(key)
        if anc_feet is not None:
            on_anc_matched.append((correct, ebm_feet == anc_feet))
        else:
            off_anc_lines.append(correct)

    matched_n = len(on_anc_matched)
    matched_ebm_acc = sum(1 for c, _ in on_anc_matched if c) / max(matched_n, 1)
    off_n = len(off_anc_lines)
    off_acc = sum(1 for c in off_anc_lines if c) / max(off_n, 1)

    summary = {
        "n_test_total_enumerable": len(test_data),
        "ebm_overall_acc_test": ebm_total_correct / max(len(test_data), 1),
        "n_anceps_matched": matched_n,
        "ebm_acc_on_anceps_matched": matched_ebm_acc,
        "n_anceps_abstained": off_n,
        "ebm_acc_on_anceps_abstained": off_acc,
        "effective_anceps_coverage": matched_n / max(matched_n + off_n, 1),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
