#!/usr/bin/env bash
# Leave-One-Book-Out evaluation across all 12 books of Aeneid.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/lobo
for b in 1 2 3 4 5 6 7 8 9 10 11 12; do
  out="results/lobo/book_${b}.json"
  if [ -f "$out" ]; then
    echo "Skip book $b (exists)"
    continue
  fi
  echo "=== Training with book $b held out ==="
  mamba run -n latin-ebm python scripts/train_v4.py \
    --test-books $b \
    --linear-epochs 30 --mlp-epochs 30 \
    --out "$out" 2>&1 | tail -5
done
echo "LOBO done"
