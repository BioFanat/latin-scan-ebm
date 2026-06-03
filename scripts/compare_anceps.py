"""Compare EBM model predictions against anceps output and pedecerto gold.

Aligns by line text (using the raw verse string). Computes:
  - per-line accuracy of each system vs gold
  - confusion matrix between EBM and anceps
  - error categories where they differ
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_v3 import precompute, predict_with  # noqa: E402

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import (
    FeatureIndex,
    build_feature_index,
    extract_features,
    set_lemma_allowlist,
)
from latin_ebm.features_v2 import PER_FOOT_DIM
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.mlp import PerFootMLP
from latin_ebm.types import FootType
from train_v3 import build_lemma_allowlist, fit_linear_adamw, fit_mlp_residual


_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    return _WS.sub(" ", s).strip()


def pattern_to_feet(pattern: str) -> tuple[FootType, ...]:
    """anceps pattern like '_ ^ ^ _ ^ ^ _ _ _ _ _ ^ ^ _ *' → feet."""
    syms = [c for c in pattern if c in "_^*"]
    feet: list[FootType] = []
    i = 0
    # Feet 1..5: dactyl '_^^' or spondee '__'
    while len(feet) < 5 and i + 1 < len(syms):
        if syms[i] == "_" and i + 2 < len(syms) and syms[i + 1] == "^" and syms[i + 2] == "^":
            feet.append(FootType.DACTYL)
            i += 3
        elif syms[i] == "_" and syms[i + 1] == "_":
            feet.append(FootType.SPONDEE)
            i += 2
        else:
            return tuple()  # malformed
    # Foot 6: final, '_*' or '__'
    if i + 1 < len(syms) and syms[i] == "_":
        feet.append(FootType.FINAL)
    return tuple(feet)


def load_anceps(path: Path) -> dict[str, tuple[FootType, ...]]:
    """Return {normalized_verse_text: foot_types}."""
    d = json.load(open(path))
    out: dict[str, tuple[FootType, ...]] = {}
    for _k, v in d["text"].items():
        verse = v.get("verse", "")
        pat = v.get("pattern", "")
        if not verse or not pat:
            continue
        feet = pattern_to_feet(pat)
        if len(feet) == 6:
            out[_norm_text(verse)] = feet
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--anceps", default="../anceps_output.json")
    p.add_argument("--out", required=True)
    p.add_argument("--linear-lr", type=float, default=5e-3)
    p.add_argument("--linear-epochs", type=int, default=30)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
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

    log.info("Building feature index...")
    lines = [ex.line for ex in train_ex]
    parses_per_line = [enumerate_parses(ex.line) for ex in train_ex]
    feature_index = build_feature_index(lines, parses_per_line, lexicon=lexicon)
    log.info("n_features=%d", feature_index.n_features)

    log.info("Precomputing...")
    train_data = precompute(train_ex, feature_index, lexicon)
    test_data = precompute(test_ex, feature_index, lexicon)

    theta_np = fit_linear_adamw(
        train_data, feature_index.n_features,
        lr=args.linear_lr, epochs=args.linear_epochs, weight_decay=1e-3,
    )
    theta_np, mlp = fit_mlp_residual(
        train_data, theta_np,
        mlp_hidden=args.mlp_hidden, lr=args.mlp_lr,
        epochs=args.mlp_epochs, finetune_linear=True, finetune_lr=args.finetune_lr,
    )

    anc = load_anceps(Path(args.anceps))
    log.info("anceps lines parsed: %d", len(anc))

    matched = 0
    ebm_correct = 0
    anc_correct = 0
    both_correct = 0
    only_ebm = 0
    only_anc = 0
    neither = 0
    ebm_vs_anc_agree = 0
    rows = []

    for pre in test_data:
        # find anceps prediction by text match
        key = _norm_text(pre.line.raw)
        anc_feet = anc.get(key)
        gold_feet = pre.gold_parse.foot_types
        idx = predict_with(pre, theta_np, mlp)
        ebm_feet = pre.candidates[idx].foot_types
        if anc_feet is None:
            continue
        matched += 1
        ebm_right = ebm_feet == gold_feet
        anc_right = anc_feet == gold_feet
        agree = ebm_feet == anc_feet
        if ebm_right:
            ebm_correct += 1
        if anc_right:
            anc_correct += 1
        if ebm_right and anc_right:
            both_correct += 1
        elif ebm_right and not anc_right:
            only_ebm += 1
        elif not ebm_right and anc_right:
            only_anc += 1
        else:
            neither += 1
        if agree:
            ebm_vs_anc_agree += 1
        rows.append({
            "line_id": pre.line.corpus_id,
            "text": pre.line.raw,
            "gold": "".join("D" if f == FootType.DACTYL else "S" if f == FootType.SPONDEE else "F" for f in gold_feet),
            "ebm":  "".join("D" if f == FootType.DACTYL else "S" if f == FootType.SPONDEE else "F" for f in ebm_feet),
            "anceps": "".join("D" if f == FootType.DACTYL else "S" if f == FootType.SPONDEE else "F" for f in anc_feet),
            "ebm_right": ebm_right, "anc_right": anc_right, "agree": agree,
        })

    summary = {
        "matched_lines": matched,
        "ebm_accuracy": ebm_correct / matched if matched else 0.0,
        "anceps_accuracy": anc_correct / matched if matched else 0.0,
        "ebm_vs_anceps_agreement": ebm_vs_anc_agree / matched if matched else 0.0,
        "both_correct": both_correct,
        "only_ebm_correct": only_ebm,
        "only_anceps_correct": only_anc,
        "neither_correct": neither,
    }
    log.info("summary: %s", json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
