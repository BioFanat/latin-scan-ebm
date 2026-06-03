"""Joint training of linear + MLP-residual hybrid EBM."""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import evaluate
from latin_ebm.features import FeatureIndex, build_feature_index, extract_features
from latin_ebm.features_v2 import PER_FOOT_DIM, per_foot_dense_features
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.mlp import PerFootMLP
from latin_ebm.types import Parse

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedLineV2:
    line: object
    candidates: list
    gold_parse: Parse
    gold_indices: list[int]
    hand_features: np.ndarray
    dense_features: np.ndarray


def precompute_v2(examples, feature_index: FeatureIndex, lexicon) -> list[PrecomputedLineV2]:
    out: list[PrecomputedLineV2] = []
    for ex in examples:
        cands = enumerate_parses(ex.line)
        if not cands:
            continue
        hand = np.stack(
            [extract_features(ex.line, c, feature_index, lexicon=lexicon) for c in cands]
        ).astype(np.float32)
        dense = np.stack([per_foot_dense_features(ex.line, c) for c in cands]).astype(np.float32)
        gold = ex.gold_parse
        gi = [
            i
            for i, c in enumerate(cands)
            if c.foot_types == gold.foot_types and c.slots == gold.slots
        ]
        if not gi:
            continue
        out.append(
            PrecomputedLineV2(
                line=ex.line,
                candidates=cands,
                gold_parse=gold,
                gold_indices=gi,
                hand_features=hand,
                dense_features=dense,
            )
        )
    return out


def train_joint(
    precomputed: list[PrecomputedLineV2],
    n_hand_features: int,
    mlp_hidden: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    l2_lambda: float = 0.01,
    device: str = "cpu",
    use_mlp: bool = True,
):
    theta = nn.Parameter(torch.zeros(n_hand_features, device=device))
    params = [{"params": [theta], "weight_decay": l2_lambda}]
    mlp = None
    if use_mlp:
        mlp = PerFootMLP(input_dim=PER_FOOT_DIM, hidden_dim=mlp_hidden).to(device)
        params.append({"params": mlp.parameters(), "weight_decay": weight_decay})
    opt = optim.AdamW(params, lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_lines = 0
        for pre in precomputed:
            hand_t = torch.from_numpy(pre.hand_features).to(device)
            dense_t = torch.from_numpy(pre.dense_features).to(device)
            linear_e = hand_t @ theta
            if use_mlp:
                n_cands = dense_t.shape[0]
                mlp_per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6).squeeze(-1)
                mlp_e = mlp_per_foot.sum(dim=1)
                energies = linear_e + mlp_e
            else:
                energies = linear_e
            log_z = torch.logsumexp(-energies, dim=0)
            gold_e = energies[pre.gold_indices]
            log_num = torch.logsumexp(-gold_e, dim=0)
            loss = -(log_num - log_z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_lines += 1
        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d loss=%.4f n=%d elapsed=%.1fs",
            epoch + 1,
            epochs,
            epoch_loss / max(n_lines, 1),
            n_lines,
            elapsed,
        )

    return theta.detach().cpu().numpy(), mlp


def predict_argmin(pre: PrecomputedLineV2, theta_np: np.ndarray, mlp, use_mlp: bool) -> int:
    hand_t = torch.from_numpy(pre.hand_features)
    linear_e = hand_t @ torch.from_numpy(theta_np).float()
    if use_mlp:
        dense_t = torch.from_numpy(pre.dense_features)
        n_cands = dense_t.shape[0]
        mlp_per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6).squeeze(-1)
        mlp_e = mlp_per_foot.sum(dim=1)
        energies = linear_e + mlp_e
    else:
        energies = linear_e
    return int(energies.argmin().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", action="append", required=False)
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--test-spec", default=None,
                   help="Format: <basename>:<comma-books>, used with multi-xml")
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--mlp-hidden", type=int, default=32)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=20)
    p.add_argument("--no-mlp", action="store_true", help="Skip MLP head (linear-only baseline)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )

    xml_paths = args.xml or ["pedecerto-raw/VERG-aene.xml"]
    all_examples = []
    for xml_path in xml_paths:
        result = parse_xml(Path(xml_path), lexicon=lexicon)
        corpus_id = Path(xml_path).stem
        for ex in result.examples:
            ex.line.author = corpus_id
            all_examples.append(ex)

    if args.test_spec:
        tcid, books_str = args.test_spec.split(":")
        test_books = set(books_str.split(","))
        train_ex = [
            e for e in all_examples
            if not (e.line.author == tcid and e.line.book in test_books)
        ]
        test_ex = [
            e for e in all_examples
            if e.line.author == tcid and e.line.book in test_books
        ]
    else:
        test_books = set(args.test_books.split(","))
        train_ex = [e for e in all_examples if e.line.book not in test_books]
        test_ex = [e for e in all_examples if e.line.book in test_books]

    logger.info("Train: %d  Test: %d", len(train_ex), len(test_ex))

    # Build feature index from training data
    logger.info("Building feature index...")
    lines = [ex.line for ex in train_ex]
    parses_per_line = [enumerate_parses(ex.line) for ex in train_ex]
    feature_index = build_feature_index(lines, parses_per_line, lexicon=lexicon)
    logger.info("Feature index: %d features", feature_index.n_features)

    logger.info("Precomputing train data...")
    train_data = precompute_v2(train_ex, feature_index, lexicon)
    logger.info("Precomputing test data...")
    test_data = precompute_v2(test_ex, feature_index, lexicon)
    logger.info("train_data=%d  test_data=%d", len(train_data), len(test_data))

    theta_np, mlp = train_joint(
        train_data,
        n_hand_features=feature_index.n_features,
        mlp_hidden=args.mlp_hidden,
        lr=args.mlp_lr,
        epochs=args.mlp_epochs,
        l2_lambda=args.l2,
        use_mlp=not args.no_mlp,
    )

    # Evaluate
    def _eval(data):
        preds = []
        for pre in data:
            idx = predict_argmin(pre, theta_np, mlp, not args.no_mlp)
            preds.append((pre.line, pre.candidates[idx], pre.gold_parse))
        return evaluate(preds)

    train_eval = _eval(train_data)
    test_eval = _eval(test_data)

    out = {
        "config": {
            "l2": args.l2,
            "mlp_hidden": args.mlp_hidden,
            "mlp_lr": args.mlp_lr,
            "mlp_epochs": args.mlp_epochs,
            "use_mlp": not args.no_mlp,
            "n_features": feature_index.n_features,
            "n_train": len(train_data),
            "n_test": len(test_data),
        },
        "train": {
            "foot_accuracy": train_eval.foot_pattern_accuracy,
            "line_exact_match": train_eval.line_exact_match,
            "syllable_accuracy": train_eval.syllable_accuracy,
        },
        "test": {
            "foot_accuracy": test_eval.foot_pattern_accuracy,
            "line_exact_match": test_eval.line_exact_match,
            "syllable_accuracy": test_eval.syllable_accuracy,
            "caesura_accuracy": test_eval.caesura_accuracy,
            "elision_f1": test_eval.elision_f1,
            "synizesis_f1": test_eval.synizesis_f1,
            "diphthong_f1": test_eval.diphthong_f1,
            "mcl_f1": test_eval.mcl_f1,
            "per_book": test_eval.per_book,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("Test foot accuracy: %.4f", test_eval.foot_pattern_accuracy)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
