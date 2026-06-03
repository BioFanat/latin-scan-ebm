"""Profile the train_v3 pipeline to find hotspots."""
from __future__ import annotations

import cProfile
import io
import logging
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_v3 import build_lemma_allowlist, fit_linear_adamw, fit_mlp_residual, precompute  # noqa: E402

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import build_feature_index, set_lemma_allowlist
from latin_ebm.lexicon import VowelLengthLexicon


def main():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    timings: dict[str, float] = {}

    t0 = time.time()
    result = parse_xml(Path("pedecerto-raw/VERG-aene.xml"), lexicon=lexicon)
    timings["parse_xml"] = time.time() - t0
    test_books = {"1", "2"}
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]

    t0 = time.time()
    allow = build_lemma_allowlist(train_ex, lexicon, min_count=3)
    set_lemma_allowlist(allow)
    timings["lemma_allowlist"] = time.time() - t0

    # Profile feature-index build
    pr = cProfile.Profile()
    pr.enable()
    lines = [ex.line for ex in train_ex]
    parses_per_line = [enumerate_parses(ex.line) for ex in train_ex]
    feature_index = build_feature_index(lines, parses_per_line, lexicon=lexicon)
    pr.disable()
    timings["build_feature_index"] = pstats.Stats(pr).total_tt
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    log.info("Feature index profile (top):\n%s", s.getvalue())
    log.info("n_features=%d", feature_index.n_features)

    # Profile precompute
    pr2 = cProfile.Profile()
    pr2.enable()
    train_data = precompute(train_ex, feature_index, lexicon)
    test_data = precompute(test_ex, feature_index, lexicon)
    pr2.disable()
    timings["precompute"] = pstats.Stats(pr2).total_tt
    s = io.StringIO()
    ps = pstats.Stats(pr2, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    log.info("Precompute profile (top):\n%s", s.getvalue())

    # Profile linear training
    t0 = time.time()
    theta_np = fit_linear_adamw(
        train_data, feature_index.n_features, lr=5e-3, epochs=5, weight_decay=1e-3
    )
    timings["linear_train_5_epochs"] = time.time() - t0

    # Profile MLP training
    t0 = time.time()
    theta_np, mlp = fit_mlp_residual(
        train_data, theta_np, mlp_hidden=64, lr=1e-3, epochs=5, finetune_linear=True, finetune_lr=5e-4
    )
    timings["mlp_train_5_epochs"] = time.time() - t0

    print("\n=== PIPELINE TIMINGS ===")
    for k, v in timings.items():
        print(f"  {k}: {v:.1f}s")


if __name__ == "__main__":
    main()
