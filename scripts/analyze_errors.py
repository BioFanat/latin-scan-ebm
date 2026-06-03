"""Per-line prediction analyzer.

Trains a Linear EBM on training books, then for each test line:
  1. Enumerates candidates
  2. Scores them with the trained energy
  3. Compares argmin to gold's foot_types
  4. If wrong-but-gold-reachable: dumps per-site decision diffs
  5. If unreachable: tags status accordingly

Output: JSONL, one row per test line, with all fields for downstream
aggregation (e.g., counting elision_diffs, MCL_diffs, etc.)
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.energy import LinearEBM
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import FeatureIndex, build_feature_index, extract_features
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.train import train_nll
from latin_ebm.types import Parse

logger = logging.getLogger(__name__)


@dataclass
class DecisionDiff:
    site_index: int
    site_type: str
    predicted: str
    gold: str


@dataclass
class ErrorRow:
    line_id: str
    status: str
    predicted_foot_types: list[str]
    gold_foot_types: list[str]
    n_candidates: int
    predicted_rank_of_gold: Optional[int]
    decision_diffs: list


def _find_gold_index(candidates: list[Parse], gold: Parse) -> Optional[int]:
    for i, c in enumerate(candidates):
        if c.foot_types == gold.foot_types and c.slots == gold.slots:
            return i
    return None


def _decision_diffs(pred: Parse, gold: Parse, sites) -> list[DecisionDiff]:
    diffs = []
    for site in sites:
        p_choice = pred.decisions.get(site.index, site.default)
        g_choice = gold.decisions.get(site.index, site.default)
        if p_choice != g_choice:
            diffs.append(DecisionDiff(
                site_index=site.index,
                site_type=site.site_type.name,
                predicted=p_choice.name,
                gold=g_choice.name,
            ))
    return diffs


def analyze(test_examples, train_examples, lexicon, l2: float, max_iter: int):
    model, _ = train_nll(
        train_examples,
        l2_lambda=l2,
        max_iter=max_iter,
        lexicon=lexicon,
    )
    rows: list[ErrorRow] = []
    for ex in test_examples:
        candidates = enumerate_parses(ex.line)
        n_cands = len(candidates)
        if not candidates:
            rows.append(ErrorRow(
                line_id=ex.line.corpus_id, status="unreachable",
                predicted_foot_types=[],
                gold_foot_types=[f.name for f in ex.gold_parse.foot_types],
                n_candidates=0, predicted_rank_of_gold=None, decision_diffs=[],
            ))
            continue

        feats = [extract_features(ex.line, c, model.feature_index, lexicon=lexicon)
                 for c in candidates]
        energies = np.array([model.energy(f) for f in feats])
        order = np.argsort(energies)
        ranked = [candidates[i] for i in order]
        pred = ranked[0]

        gold_rank = _find_gold_index(ranked, ex.gold_parse)
        if pred.foot_types == ex.gold_parse.foot_types and pred.slots == ex.gold_parse.slots:
            status = "correct"
        elif gold_rank is not None:
            status = "wrong_reachable"
        else:
            status = "unreachable"

        rows.append(ErrorRow(
            line_id=ex.line.corpus_id,
            status=status,
            predicted_foot_types=[f.name for f in pred.foot_types],
            gold_foot_types=[f.name for f in ex.gold_parse.foot_types],
            n_candidates=n_cands,
            predicted_rank_of_gold=gold_rank,
            decision_diffs=[asdict(d) for d in _decision_diffs(pred, ex.gold_parse, ex.line.sites)],
        ))
    return rows


def summarize(jsonl_path: Path) -> dict:
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l]
    total = len(rows)
    status_counts = Counter(r["status"] for r in rows)
    diff_type_counts: Counter = Counter()
    diff_direction_counts: Counter = Counter()
    for r in rows:
        if r["status"] != "wrong_reachable":
            continue
        for d in r["decision_diffs"]:
            diff_type_counts[d["site_type"]] += 1
            diff_direction_counts[f"{d['site_type']}:{d['gold']}->{d['predicted']}"] += 1
    return {
        "total": total,
        "by_status": dict(status_counts),
        "accuracy": status_counts.get("correct", 0) / total if total else 0.0,
        "wrong_reachable_by_site_type": dict(diff_type_counts),
        "wrong_reachable_by_direction": dict(diff_direction_counts.most_common(20)),
        "mean_candidates_when_wrong": (
            sum(r["n_candidates"] for r in rows if r["status"] == "wrong_reachable")
            / max(status_counts.get("wrong_reachable", 1), 1)
        ),
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    run_p.add_argument("--test-books", default="1,2")
    run_p.add_argument("--l2", type=float, default=0.01)
    run_p.add_argument("--max-iter", type=int, default=500)
    run_p.add_argument("--out", required=True)
    sum_p = sub.add_parser("summarize")
    sum_p.add_argument("jsonl")
    args = p.parse_args()

    if args.cmd == "summarize":
        print(json.dumps(summarize(Path(args.jsonl)), indent=2))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lexicon = VowelLengthLexicon(
        mqdq_path=Path("data/MqDqMacrons.json"),
        morpheus_path=Path("data/MorpheusMacrons.txt"),
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train = [e for e in result.examples if e.line.book not in test_books]
    test = [e for e in result.examples if e.line.book in test_books]
    rows = analyze(test, train, lexicon, args.l2, args.max_iter)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
