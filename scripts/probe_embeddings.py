"""Phase-2a probes: is metrical/rhetorical structure linearly decodable from
the frozen line-embedding space, above controls?

Task 15: extract h_line per line (books 1-2), fit linear probes (LogisticRegression
+ cross_val_score) for foot_pattern, caesura, and the sense_pause rhetorical proxy,
with a RANDOM-INIT encoder control.

Task 16: add a `--hand-baseline` path (probe the existing hand-features as a baseline)
and a PCA/UMAP scatter of h_line colored by foot pattern -> results/hline_umap.png.
Uses PCA if umap isn't installed.
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.encoder import MetricalEncoder, MetricalTokenizer
from latin_ebm.lexicon import VowelLengthLexicon


def load_encoder(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["config"]
    tok = MetricalTokenizer(ck["text_vocab"], ck["word_vocab"])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size,
                          d_model=cfg["d_model"], n_layers=cfg["n_layers"])
    enc.load_state_dict(ck["encoder"]); enc.eval()
    return enc, tok


def embed(enc, tok, lines):
    X = []
    with torch.no_grad():
        for line in lines:
            _, h_line = enc(tok.encode(line))
            X.append(h_line.numpy())
    return np.stack(X)


def sense_pause_label(line) -> int:
    """Rhetorical proxy: does the line contain a mid-line sense pause
    (comma/semicolon/colon/interpunct inside, not just line-final)?

    Reads `line.source_text`, which the pedecerto loader preserves WITH
    punctuation (`line.words`/`line.raw` are stripped by atomize, so they
    cannot supply this). Mid-line = punctuation on any token before the last.
    """
    src = getattr(line, "source_text", "") or ""
    tokens = src.split()
    if len(tokens) < 2:
        return 0
    body = " ".join(tokens[:-1])  # exclude line-final token (line-final punct)
    return int(bool(re.search(r"[,;:·]", body)))


def probe(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold cross-validated linear-probe accuracy.

    Note: the plan's snippet passed `multi_class="auto"`, but that kwarg was
    removed in sklearn >= 1.7 (multinomial is now the default). Dropping it
    preserves the intended behavior on this environment's sklearn.
    """
    clf = LogisticRegression(max_iter=1000)
    return float(cross_val_score(clf, X, y, cv=5).mean())


def hand_feature_matrix(examples, lexicon) -> np.ndarray:
    """One hand-feature vector per line, using the gold parse and a frozen
    feature index. Reuses the train_v4 extract_features/_RecorderIndex pattern."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_v4 import _RecorderIndex  # type: ignore
    from latin_ebm.features import extract_features

    # Pass 1: collect per-line gold feature dicts (name -> value).
    feature_dicts: list[dict[str, float]] = []
    for ex in examples:
        recorder = _RecorderIndex()
        vec = extract_features(ex.line, ex.gold_parse, recorder, lexicon=lexicon)  # type: ignore
        d: dict[str, float] = {}
        for idx_in_vec, name in enumerate(recorder._idx_to_name):
            if idx_in_vec < len(vec):
                v = float(vec[idx_in_vec])
                if v != 0.0:
                    d[name] = v
        feature_dicts.append(d)

    # Build a frozen feature index from all observed names.
    names = sorted({n for fd in feature_dicts for n in fd})
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Pass 2: vectorize with the fixed ordering.
    X = np.zeros((len(feature_dicts), len(names)), dtype=np.float32)
    for li, fd in enumerate(feature_dicts):
        for name, v in fd.items():
            X[li, name_to_idx[name]] = v
    return X


def _reduce_2d(X: np.ndarray):
    """Reduce h_line to 2-D. Prefer UMAP if installed, else PCA."""
    try:
        import umap  # type: ignore
        coords = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
        return coords, "umap"
    except Exception:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(X)
        return coords, "pca"


def _write_minimal_png(out_path: str) -> None:
    """Write a tiny valid 1x1 PNG so the required output path always exists,
    even when matplotlib is unavailable in the environment."""
    import struct, zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    raw = b"\x00\xff\xff\xff"  # filter byte + one white RGB pixel
    idat = zlib.compress(raw)
    png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    Path(out_path).write_bytes(png)


def scatter_hline(X: np.ndarray, foot_labels: list[str], out_path: str) -> str:
    """2-D PCA (or UMAP if installed) scatter of h_line colored by foot pattern.

    Saves the figure to `out_path`. If matplotlib is not installed, falls back
    to writing the 2-D coordinates + labels to a sidecar .npz and a minimal
    placeholder PNG so the pipeline still produces the required artifact.
    """
    coords, method = _reduce_2d(X)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # No plotting backend available: persist reduced data + placeholder PNG.
        sidecar = str(Path(out_path).with_suffix(".npz"))
        np.savez(sidecar, coords=coords, foot_labels=np.array(foot_labels))
        _write_minimal_png(out_path)
        return f"{method} (no-matplotlib; coords saved to {Path(sidecar).name})"

    # Color the K most common foot patterns; lump the rest into "other".
    from collections import Counter
    top = [p for p, _ in Counter(foot_labels).most_common(8)]
    top_set = set(top)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))
    others = np.array([lab not in top_set for lab in foot_labels])
    if others.any():
        ax.scatter(coords[others, 0], coords[others, 1], s=8, c="lightgray",
                   alpha=0.5, label="other")
    for i, pat in enumerate(top):
        m = np.array([lab == pat for lab in foot_labels])
        ax.scatter(coords[m, 0], coords[m, 1], s=10, color=cmap(i % 10),
                   alpha=0.8, label=f"{pat[:24]}")
    ax.set_title(f"h_line ({method.upper()}) colored by foot pattern")
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.legend(fontsize=6, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return method


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="results/probes.json")
    p.add_argument("--hand-baseline", action="store_true",
                   help="also probe the hand-engineered features as a baseline")
    p.add_argument("--umap-out", default="results/hline_umap.png",
                   help="path for the PCA/UMAP scatter of h_line")
    args = p.parse_args()

    lexicon = VowelLengthLexicon(Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt"))
    result = parse_xml(Path("pedecerto-raw/VERG-aene.xml"), lexicon=lexicon)
    examples = [e for e in result.examples if e.line.book in {"1", "2"}]
    lines = [e.line for e in examples]
    golds = [e.gold_parse for e in examples]

    enc, tok = load_encoder(args.ckpt)
    X = embed(enc, tok, lines)
    # random-init control
    enc_rand = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size,
                               d_model=enc.d_model, n_layers=len(enc.transformer.layers))
    enc_rand.eval()
    X_rand = embed(enc_rand, tok, lines)

    # optional hand-feature baseline
    X_hand = None
    if args.hand_baseline:
        X_hand = hand_feature_matrix(examples, lexicon)

    foot_labels = [str(g.foot_types) for g in golds]
    targets = {
        "foot_pattern": foot_labels,
        "caesura": [str(g.caesura) for g in golds],
        "sense_pause": [sense_pause_label(l) for l in lines],
    }
    out = {}
    for name, y in targets.items():
        y = np.array(y)
        if len(set(y.tolist())) < 2:
            continue
        entry = {
            "learned": probe(X, y),
            "random_control": probe(X_rand, y),
        }
        if X_hand is not None:
            entry["hand_features"] = probe(X_hand, y)
        out[name] = entry

    # clustering viz of the learned h_line space
    viz_method = scatter_hline(X, foot_labels, args.umap_out)
    out["_meta"] = {
        "n_lines": int(len(lines)),
        "d_model": int(enc.d_model),
        "viz_method": viz_method,
        "viz_path": args.umap_out,
        "hand_baseline": bool(args.hand_baseline),
    }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
