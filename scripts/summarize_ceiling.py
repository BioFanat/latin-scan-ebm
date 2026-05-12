"""Aggregate diagnose_ceiling TSV into summary stats.

Ceiling = fraction of test lines where gold foot pattern is in candidate set
       = (total - no_candidates - gold_unreachable) / total
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def summarize(tsv_path: Path) -> dict:
    with open(tsv_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    statuses = Counter(r["status"] for r in rows)
    reasons = Counter((r["status"], r["reason"]) for r in rows if r["reason"])
    total = len(rows)
    unreachable = statuses.get("no_candidates", 0) + statuses.get("gold_unreachable", 0)
    return {
        "total": total,
        "by_status": dict(statuses),
        "by_reason": {f"{s}/{r}": n for (s, r), n in reasons.items()},
        "ceiling": (total - unreachable) / total if total else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tsv")
    p.add_argument("--compare", default=None)
    args = p.parse_args()
    a = summarize(Path(args.tsv))
    print(json.dumps(a, indent=2))
    if args.compare:
        b = summarize(Path(args.compare))
        delta = a["ceiling"] - b["ceiling"]
        print(f"\nCeiling: {a['ceiling']:.3%} (was {b['ceiling']:.3%}, delta {delta:+.3%})")


if __name__ == "__main__":
    main()
