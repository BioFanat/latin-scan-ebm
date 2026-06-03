"""Train hybrid linear+MLP EBM with L-BFGS for linear part, AdamW for MLP residual.

Strategy:
1. Build feature index from training enumerations
2. Optionally restrict lemma features by frequency (set_lemma_allowlist)
3. Train linear EBM via scipy L-BFGS-B (fast, no gradient issues)
4. Optionally fit a small MLP residual head via PyTorch:
   - Either two-stage (linear frozen) or joint (linear + MLP via AdamW)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import evaluate
from latin_ebm.features import (
    FeatureIndex,
    build_feature_index,
    extract_features,
    set_lemma_allowlist,
)
from latin_ebm.features_v2 import PER_FOOT_DIM, per_foot_dense_features
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.mlp import PerFootMLP
from latin_ebm.train import train_nll
from latin_ebm.types import Parse

logger = logging.getLogger(__name__)


def build_lemma_allowlist(examples, lexicon, min_count: int = 3) -> frozenset[str]:
    """Allow only lemmas appearing in at least min_count training words."""
    counts: Counter = Counter()
    for ex in examples:
        for word in ex.line.words:
            lem = lexicon.lemma(word)
            if lem is not None and len(lem) >= 2:
                counts[lem] += 1
    return frozenset(l for l, c in counts.items() if c >= min_count)


@dataclass
class PrecomputedV3:
    line: object
    candidates: list
    gold_parse: Parse
    gold_indices: list[int]
    hand_features: np.ndarray
    dense_features: np.ndarray


def precompute(examples, feature_index, lexicon) -> list[PrecomputedV3]:
    out: list[PrecomputedV3] = []
    for ex in examples:
        cands = enumerate_parses(ex.line)
        if not cands:
            continue
        hand = np.stack(
            [extract_features(ex.line, c, feature_index, lexicon=lexicon) for c in cands]
        ).astype(np.float32)
        dense = np.stack(
            [per_foot_dense_features(ex.line, c) for c in cands]
        ).astype(np.float32)
        gold = ex.gold_parse
        gi = [
            i
            for i, c in enumerate(cands)
            if c.foot_types == gold.foot_types and c.slots == gold.slots
        ]
        if not gi:
            continue
        out.append(
            PrecomputedV3(
                line=ex.line, candidates=cands, gold_parse=gold,
                gold_indices=gi, hand_features=hand, dense_features=dense,
            )
        )
    return out


def fit_linear_adamw(
    precomputed: list[PrecomputedV3],
    n_features: int,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    epochs: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Train linear EBM via AdamW. Faster than L-BFGS when features ≫ 1000."""
    torch.manual_seed(seed)
    theta = nn.Parameter(torch.zeros(n_features, dtype=torch.float32))
    opt = optim.AdamW([theta], lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_correct = 0
        n = 0
        for pre in precomputed:
            hand_t = torch.from_numpy(pre.hand_features)
            energies = hand_t @ theta
            log_z = torch.logsumexp(-energies, dim=0)
            gold_e = energies[pre.gold_indices]
            log_num = torch.logsumexp(-gold_e, dim=0)
            loss = -(log_num - log_z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            if int(energies.argmin().item()) in pre.gold_indices:
                n_correct += 1
            n += 1
        scheduler.step()
        logger.info(
            "Linear epoch %d/%d loss=%.4f train_acc=%.3f elapsed=%.1fs",
            epoch + 1, epochs, epoch_loss / max(n, 1), n_correct / max(n, 1),
            time.time() - t0,
        )
    return theta.detach().cpu().numpy()


def fit_mlp_residual(
    precomputed: list[PrecomputedV3],
    theta_np: np.ndarray,
    mlp_hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    finetune_linear: bool = True,
    finetune_lr: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, PerFootMLP]:
    """Fit an MLP residual on top of the linear part. theta_np starts from L-BFGS solution.

    If finetune_linear=True, theta is also a torch parameter optimized jointly.
    """
    torch.manual_seed(seed)
    device = "cpu"
    theta = torch.tensor(theta_np, dtype=torch.float32, device=device, requires_grad=finetune_linear)
    mlp = PerFootMLP(input_dim=PER_FOOT_DIM, hidden_dim=mlp_hidden).to(device)

    params = [{"params": mlp.parameters(), "weight_decay": weight_decay, "lr": lr}]
    if finetune_linear:
        params.append({"params": [theta], "weight_decay": 0.0, "lr": finetune_lr})
    opt = optim.AdamW(params)

    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for pre in precomputed:
            hand_t = torch.from_numpy(pre.hand_features).to(device)
            dense_t = torch.from_numpy(pre.dense_features).to(device)
            n_cands = dense_t.shape[0]
            linear_e = hand_t @ theta
            mlp_per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6).squeeze(-1)
            mlp_e = mlp_per_foot.sum(dim=1)
            energies = linear_e + mlp_e
            log_z = torch.logsumexp(-energies, dim=0)
            gold_e = energies[pre.gold_indices]
            log_num = torch.logsumexp(-gold_e, dim=0)
            loss = -(log_num - log_z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        logger.info(
            "MLP epoch %d/%d loss=%.4f elapsed=%.1fs",
            epoch + 1, epochs, epoch_loss / max(len(precomputed), 1), time.time() - t0,
        )
    return theta.detach().cpu().numpy(), mlp


def predict_with(pre: PrecomputedV3, theta_np: np.ndarray, mlp: PerFootMLP | None) -> int:
    hand = pre.hand_features @ theta_np
    if mlp is None:
        return int(np.argmin(hand))
    dense_t = torch.from_numpy(pre.dense_features)
    with torch.no_grad():
        n_cands = dense_t.shape[0]
        per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6).squeeze(-1)
        mlp_e = per_foot.sum(dim=1).cpu().numpy()
    return int(np.argmin(hand + mlp_e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--lemma-min-count", type=int, default=3)
    p.add_argument("--optimizer", choices=["lbfgs", "adamw"], default="adamw")
    p.add_argument("--linear-lr", type=float, default=5e-3)
    p.add_argument("--linear-epochs", type=int, default=30)
    p.add_argument("--linear-wd", type=float, default=1e-3)
    p.add_argument("--mlp", action="store_true", help="Fit MLP residual head")
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=20)
    p.add_argument("--finetune-linear", action="store_true")
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]
    logger.info("Train: %d  Test: %d", len(train_ex), len(test_ex))

    if args.lemma_min_count > 0:
        allow = build_lemma_allowlist(train_ex, lexicon, min_count=args.lemma_min_count)
        logger.info("Lemma allowlist: %d lemmas", len(allow))
        set_lemma_allowlist(allow)

    if args.optimizer == "lbfgs":
        model, train_result = train_nll(
            train_ex,
            l2_lambda=args.l2,
            max_iter=args.max_iter,
            lexicon=lexicon,
        )
        feature_index = model.feature_index
        logger.info("n_features=%d converged=%s iter=%d loss=%.4f",
                    feature_index.n_features, train_result.converged,
                    train_result.n_iterations, train_result.final_loss)
        theta_np = model.theta.astype(np.float32)

        # Precompute for evaluation (and MLP if used)
        logger.info("Precomputing train data...")
        train_data = precompute(train_ex, feature_index, lexicon)
        logger.info("Precomputing test data...")
        test_data = precompute(test_ex, feature_index, lexicon)
        logger.info("train_data=%d test_data=%d", len(train_data), len(test_data))
    else:  # adamw
        # Build feature index first
        logger.info("Building feature index...")
        lines = [ex.line for ex in train_ex]
        parses_per_line = [enumerate_parses(ex.line) for ex in train_ex]
        feature_index = build_feature_index(lines, parses_per_line, lexicon=lexicon)
        logger.info("n_features=%d", feature_index.n_features)

        logger.info("Precomputing train data...")
        train_data = precompute(train_ex, feature_index, lexicon)
        logger.info("Precomputing test data...")
        test_data = precompute(test_ex, feature_index, lexicon)
        logger.info("train_data=%d test_data=%d", len(train_data), len(test_data))

        theta_np = fit_linear_adamw(
            train_data, feature_index.n_features,
            lr=args.linear_lr,
            weight_decay=args.linear_wd,
            epochs=args.linear_epochs,
        )
        train_result = None

    mlp = None
    if args.mlp:
        theta_np, mlp = fit_mlp_residual(
            train_data, theta_np,
            mlp_hidden=args.mlp_hidden,
            lr=args.mlp_lr,
            epochs=args.mlp_epochs,
            finetune_linear=args.finetune_linear,
            finetune_lr=args.finetune_lr,
        )

    # Evaluate
    def _eval(data):
        preds = []
        for pre in data:
            idx = predict_with(pre, theta_np, mlp)
            preds.append((pre.line, pre.candidates[idx], pre.gold_parse))
        return evaluate(preds)

    train_eval = _eval(train_data)
    test_eval = _eval(test_data)
    out = {
        "config": vars(args),
        "n_features": feature_index.n_features,
        "n_train": len(train_data),
        "n_test": len(test_data),
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
        "linear_training": (
            {
                "loss": train_result.final_loss,
                "iterations": train_result.n_iterations,
                "converged": train_result.converged,
            } if train_result is not None else None
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    logger.info("Train foot acc: %.4f  Test foot acc: %.4f",
                train_eval.foot_pattern_accuracy, test_eval.foot_pattern_accuracy)


if __name__ == "__main__":
    main()
