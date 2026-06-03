"""Optimized hybrid linear+MLP trainer.

Key speed wins over train_v3:
1. Single enumeration pass (avoids redundant enumerate_parses across build/precompute)
2. Two-pass feature extraction (collect names, freeze index, vectorize once)
3. Lazy dense MLP features (only on train_data once)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import evaluate
from latin_ebm.features import FeatureIndex, extract_features, set_lemma_allowlist
from latin_ebm.features_v2 import PER_FOOT_DIM, per_foot_dense_features
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.mlp import PerFootMLP
from latin_ebm.types import Parse

logger = logging.getLogger(__name__)


def build_lemma_allowlist(examples, lexicon, min_count: int = 3) -> frozenset[str]:
    counts: Counter = Counter()
    for ex in examples:
        for word in ex.line.words:
            lem = lexicon.lemma(word)
            if lem is not None and len(lem) >= 2:
                counts[lem] += 1
    return frozenset(l for l, c in counts.items() if c >= min_count)


@dataclass
class LineData:
    line: object
    candidates: list
    gold_parse: Parse
    gold_indices: list[int]
    feature_dicts: list[dict[str, float]]  # one per candidate, lazy → vectorized later
    hand_features: np.ndarray | None = None
    dense_features: np.ndarray | None = None


def _features_dict(line, parse, lexicon) -> dict[str, float]:
    """Run extract_features but capture the features dict (instead of dense vec).
    We do this by passing a sentinel FeatureIndex that doesn't grow."""
    # Reuse extract_features by passing a FeatureIndex; we don't care about vec yet.
    # The function mutates a local `features` dict — we need to capture it.
    # Simplest: just call extract_features with a fresh index, then read the index
    # back via vec. But that loses the name→value mapping.
    # Instead, monkey-patch: call extract_features once with a Recorder that captures.
    # Cleaner: reimplement by importing and calling the same logic with a wrapper.
    idx = _RecorderIndex()
    extract_features(line, parse, idx, lexicon=lexicon)  # type: ignore
    return idx.captured


class _RecorderIndex:
    """Drop-in for FeatureIndex that captures name→value via get_or_add calls.

    Because extract_features assigns vec[idx] = value at end, we map idx→name
    and then read the value from vec... but vec is computed inside the function.

    Simpler: just collect names that get registered.
    """

    def __init__(self):
        self.captured: dict[str, float] = {}
        self._name_to_idx: dict[str, int] = {}
        self._values: dict[str, float] = {}
        self._frozen = False
        self.n_features = 0
        # Track which idx maps to which name for value lookup
        self._idx_to_name: list[str] = []

    def get_or_add(self, name: str) -> int:
        if name in self._name_to_idx:
            return self._name_to_idx[name]
        idx = len(self._name_to_idx)
        self._name_to_idx[name] = idx
        self._idx_to_name.append(name)
        self.n_features = len(self._name_to_idx)
        return idx

    def get(self, name: str) -> int:
        return self._name_to_idx.get(name, -1)

    @property
    def names(self):
        return list(self._idx_to_name)


def collect_feature_dicts(examples, lexicon, log_every: int = 1000) -> list[LineData]:
    """Single pass: enumerate + collect feature names per candidate."""
    out: list[LineData] = []
    skipped = 0
    # Patch: call extract_features but capture features dict before vectorization.
    # We do this by wrapping the function to copy the local defaultdict.
    # Approach: import the function, copy its source, and intercept.
    # Pragmatic: just call extract_features and read which names got registered;
    # but values matter for non-binary features (lex_agree_ratio, etc.)
    # So we re-implement by getting the dict directly via a thread-local hack:
    # — Cleaner solution: extract the features dict via inspecting the FeatureIndex
    #   after a fresh-per-line call, but values can be > 1.0 (counts).
    # The proper fix: parse the dense vector back to (name, value) pairs.
    for i, ex in enumerate(examples):
        cands = enumerate_parses(ex.line)
        if not cands:
            skipped += 1
            continue
        gold = ex.gold_parse
        gi = [
            j for j, c in enumerate(cands)
            if c.foot_types == gold.foot_types and c.slots == gold.slots
        ]
        if not gi:
            skipped += 1
            continue
        # Per-candidate feature dict — extract from dense vector
        feature_dicts: list[dict[str, float]] = []
        for c in cands:
            recorder = _RecorderIndex()
            vec = extract_features(ex.line, c, recorder, lexicon=lexicon)  # type: ignore
            d: dict[str, float] = {}
            for idx_in_vec, name in enumerate(recorder._idx_to_name):
                if idx_in_vec < len(vec):
                    v = float(vec[idx_in_vec])
                    if v != 0.0:
                        d[name] = v
            feature_dicts.append(d)
        out.append(
            LineData(
                line=ex.line,
                candidates=cands,
                gold_parse=gold,
                gold_indices=gi,
                feature_dicts=feature_dicts,
            )
        )
        if (i + 1) % log_every == 0:
            logger.info("collect_feature_dicts: %d/%d (%d skipped)", i + 1, len(examples), skipped)
    logger.info("collect_feature_dicts done: %d kept, %d skipped", len(out), skipped)
    return out


def vectorize(data: list[LineData], feature_names: list[str], include_dense: bool):
    """Convert feature_dicts → numpy arrays using a fixed feature ordering."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    n_feat = len(feature_names)
    for d in data:
        n_cands = len(d.candidates)
        h = np.zeros((n_cands, n_feat), dtype=np.float32)
        for ci, fd in enumerate(d.feature_dicts):
            for name, v in fd.items():
                idx = name_to_idx.get(name, -1)
                if idx >= 0:
                    h[ci, idx] = v
        d.hand_features = h
        if include_dense:
            d.dense_features = np.stack(
                [per_foot_dense_features(d.line, c) for c in d.candidates]
            ).astype(np.float32)
        d.feature_dicts = []  # free memory


def fit_linear_adamw(
    data: list[LineData], n_features: int,
    lr: float = 5e-3, weight_decay: float = 1e-3, epochs: int = 30, seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)
    theta = nn.Parameter(torch.zeros(n_features, dtype=torch.float32))
    opt = optim.AdamW([theta], lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_correct = 0
        n = 0
        for d in data:
            hand_t = torch.from_numpy(d.hand_features)
            energies = hand_t @ theta
            log_z = torch.logsumexp(-energies, dim=0)
            gold_e = energies[d.gold_indices]
            log_num = torch.logsumexp(-gold_e, dim=0)
            loss = -(log_num - log_z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            if int(energies.argmin().item()) in d.gold_indices:
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
    data: list[LineData], theta_np: np.ndarray,
    mlp_hidden: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 30,
    finetune_linear: bool = True, finetune_lr: float = 5e-4, seed: int = 0,
):
    torch.manual_seed(seed)
    theta = torch.tensor(theta_np, dtype=torch.float32, requires_grad=finetune_linear)
    mlp = PerFootMLP(input_dim=PER_FOOT_DIM, hidden_dim=mlp_hidden)
    params = [{"params": mlp.parameters(), "weight_decay": weight_decay, "lr": lr}]
    if finetune_linear:
        params.append({"params": [theta], "weight_decay": 0.0, "lr": finetune_lr})
    opt = optim.AdamW(params)
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for d in data:
            hand_t = torch.from_numpy(d.hand_features)
            dense_t = torch.from_numpy(d.dense_features)
            n_cands = dense_t.shape[0]
            linear_e = hand_t @ theta
            mlp_per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6).squeeze(-1)
            mlp_e = mlp_per_foot.sum(dim=1)
            energies = linear_e + mlp_e
            log_z = torch.logsumexp(-energies, dim=0)
            gold_e = energies[d.gold_indices]
            log_num = torch.logsumexp(-gold_e, dim=0)
            loss = -(log_num - log_z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        logger.info("MLP epoch %d/%d loss=%.4f elapsed=%.1fs",
                    epoch + 1, epochs, epoch_loss / max(len(data), 1), time.time() - t0)
    return theta.detach().cpu().numpy(), mlp


def predict_with(d: LineData, theta_np: np.ndarray, mlp) -> int:
    hand = d.hand_features @ theta_np
    if mlp is None:
        return int(np.argmin(hand))
    dense_t = torch.from_numpy(d.dense_features)
    with torch.no_grad():
        n_cands = dense_t.shape[0]
        per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6).squeeze(-1)
        mlp_e = per_foot.sum(dim=1).cpu().numpy()
    return int(np.argmin(hand + mlp_e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--lemma-min-count", type=int, default=3)
    p.add_argument("--linear-lr", type=float, default=5e-3)
    p.add_argument("--linear-epochs", type=int, default=30)
    p.add_argument("--linear-wd", type=float, default=1e-3)
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--finetune-lr", type=float, default=5e-4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t_start = time.time()
    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]
    logger.info("Train: %d  Test: %d", len(train_ex), len(test_ex))

    allow = build_lemma_allowlist(train_ex, lexicon, min_count=args.lemma_min_count)
    set_lemma_allowlist(allow)
    logger.info("Lemma allowlist: %d", len(allow))

    # SINGLE-PASS enumeration + feature collection
    t0 = time.time()
    train_data = collect_feature_dicts(train_ex, lexicon)
    test_data = collect_feature_dicts(test_ex, lexicon)
    logger.info("collect_feature_dicts: %.1fs", time.time() - t0)

    # Build the union feature index from TRAINING data
    t0 = time.time()
    name_counts: Counter = Counter()
    for d in train_data:
        for fd in d.feature_dicts:
            for name in fd:
                name_counts[name] += 1
    feature_names = sorted(name_counts.keys())
    logger.info("feature index built: %d features (%.1fs)", len(feature_names), time.time() - t0)

    # Vectorize
    t0 = time.time()
    vectorize(train_data, feature_names, include_dense=not args.no_mlp)
    vectorize(test_data, feature_names, include_dense=not args.no_mlp)
    logger.info("vectorize: %.1fs", time.time() - t0)

    theta_np = fit_linear_adamw(
        train_data, len(feature_names),
        lr=args.linear_lr, weight_decay=args.linear_wd, epochs=args.linear_epochs,
    )
    mlp = None
    if not args.no_mlp:
        theta_np, mlp = fit_mlp_residual(
            train_data, theta_np,
            mlp_hidden=args.mlp_hidden, lr=args.mlp_lr, epochs=args.mlp_epochs,
            finetune_linear=True, finetune_lr=args.finetune_lr,
        )

    # Evaluate
    def _eval(data):
        preds = []
        for d in data:
            idx = predict_with(d, theta_np, mlp)
            preds.append((d.line, d.candidates[idx], d.gold_parse))
        return evaluate(preds)

    train_eval = _eval(train_data)
    test_eval = _eval(test_data)
    out = {
        "config": vars(args),
        "n_features": len(feature_names),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "total_pipeline_seconds": time.time() - t_start,
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
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    logger.info("Train: %.4f  Test: %.4f  Total: %.1fs",
                train_eval.foot_pattern_accuracy, test_eval.foot_pattern_accuracy,
                time.time() - t_start)


if __name__ == "__main__":
    main()
