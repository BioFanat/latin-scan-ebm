"""Anceps + EBM ensemble.

Strategy: for each test line, prefer anceps's prediction when it didn't fail
('automatic' method). Fall back to EBM otherwise.

We evaluate three modes:
  - anceps_only:   anceps where available else MISS (counted as error)
  - ebm_only:      EBM everywhere
  - ensemble:      anceps where 'automatic', EBM elsewhere

All evaluated on EBM-enumerable test lines (lines where the gold parse is in
EBM's candidate set).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_v4 import (  # noqa: E402
    build_lemma_allowlist,
    collect_feature_dicts,
    fit_linear_adamw,
    fit_mlp_residual,
    predict_with,
    vectorize,
)
from compare_anceps import _norm_text, pattern_to_feet  # noqa: E402

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.features import set_lemma_allowlist
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.types import FootType


def load_anceps_full(path: Path):
    """Return {norm_text: (foot_types, method)} including failed methods."""
    d = json.load(open(path))
    out: dict[str, tuple] = {}
    for _k, v in d["text"].items():
        verse = v.get("verse", "")
        pat = v.get("pattern", "")
        method = v.get("method", "")
        if not verse:
            continue
        feet = pattern_to_feet(pat) if pat else tuple()
        out[_norm_text(verse)] = (feet, method)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--anceps", default="../anceps_output.json")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]

    allow = build_lemma_allowlist(train_ex, lexicon, min_count=3)
    set_lemma_allowlist(allow)

    train_data = collect_feature_dicts(train_ex, lexicon)
    test_data = collect_feature_dicts(test_ex, lexicon)
    from collections import Counter
    name_counts = Counter()
    for d in train_data:
        for fd in d.feature_dicts:
            for name in fd:
                name_counts[name] += 1
    feature_names = sorted(name_counts.keys())
    vectorize(train_data, feature_names, include_dense=True)
    vectorize(test_data, feature_names, include_dense=True)

    theta_np = fit_linear_adamw(train_data, len(feature_names), lr=5e-3, epochs=30, weight_decay=1e-3)
    theta_np, mlp = fit_mlp_residual(train_data, theta_np, mlp_hidden=64, lr=1e-3, epochs=30,
                                      finetune_linear=True, finetune_lr=5e-4)

    anc = load_anceps_full(Path(args.anceps))

    n = len(test_data)
    anceps_correct = 0      # anceps automatic + matches gold
    anceps_attempts = 0     # anceps automatic
    ebm_correct = 0
    ensemble_correct = 0
    sources = {"anceps": 0, "ebm_fallback": 0}
    for d in test_data:
        idx = predict_with(d, theta_np, mlp)
        ebm_feet = d.candidates[idx].foot_types
        gold_feet = d.gold_parse.foot_types
        key = _norm_text(d.line.raw)
        entry = anc.get(key)
        anc_feet = entry[0] if entry else tuple()
        anc_method = entry[1] if entry else "missing"
        anc_attempted = (anc_method == "automatic") and len(anc_feet) == 6

        if anc_attempted:
            anceps_attempts += 1
            if anc_feet == gold_feet:
                anceps_correct += 1
        if ebm_feet == gold_feet:
            ebm_correct += 1
        # Ensemble: anceps if it attempted, else EBM
        if anc_attempted:
            ens_feet = anc_feet
            sources["anceps"] += 1
        else:
            ens_feet = ebm_feet
            sources["ebm_fallback"] += 1
        if ens_feet == gold_feet:
            ensemble_correct += 1

    summary = {
        "n_enumerable": n,
        "anceps_attempts": anceps_attempts,
        "anceps_attempt_rate": anceps_attempts / n if n else 0,
        "anceps_accuracy_when_attempts": anceps_correct / max(anceps_attempts, 1),
        "anceps_overall_accuracy_full_set": anceps_correct / n if n else 0,
        "ebm_accuracy": ebm_correct / n if n else 0,
        "ensemble_accuracy": ensemble_correct / n if n else 0,
        "ensemble_sources": sources,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
