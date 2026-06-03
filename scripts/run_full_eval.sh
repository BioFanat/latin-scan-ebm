#!/usr/bin/env bash
# Full-run driver for the metrical encoder (inline, no agents).
# Runs every stage's training + measurement sequentially, logging to
# results/full_run/. Real configs (d_model=128, n_layers=3, 30 epochs,
# batch_size=16) so the gates can be checked against the Phase-F baselines.
# NO git commits — artifacts are left for human review.
set -euo pipefail
cd "$(dirname "$0")/.."

R=results/full_run
mkdir -p "$R"

# Single-instance guard: refuse to start if another run holds the lock.
LOCK="$R/.run.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "Another run is active (pid $(cat "$LOCK")). Aborting."; exit 3
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
COMMON="--d-model 128 --n-layers 3 --epochs 30 --batch-size 16"
run() { echo "=== $(date '+%H:%M:%S') $* ==="; mamba run -n latin-ebm python "$@"; }

# Stage 0 — baseline ceiling split (fast)
run scripts/measure_ceiling_split.py --out "$R/ceiling_b12.json"

# Stage 1.1 — supervised-only (aux off): gate >= 0.90
run scripts/train_v5.py $COMMON --lambda-mlm 0 --lambda-mwm 0 --out "$R/s11_supervised.json"

# Stage 1.2 — full multitask: gate >= 0.9259 ; + two ablations for MLM/MWM deltas
run scripts/train_v5.py $COMMON --lambda-mlm 0.1 --lambda-mwm 0.1 --out "$R/s12_full.json"
run scripts/train_v5.py $COMMON --lambda-mlm 0   --lambda-mwm 0.1 --out "$R/s12_nomlm.json"
run scripts/train_v5.py $COMMON --lambda-mlm 0.1 --lambda-mwm 0   --out "$R/s12_nomwm.json"

# Stage 1.2 — LOBO books 1..6: gate mean >= 0.9313
for B in 1 2 3 4 5 6; do
  run scripts/train_v5.py $COMMON --lambda-mlm 0.1 --lambda-mwm 0.1 --test-books "$B" --out "$R/lobo_book$B.json"
done

# Stage 1.3 — proposal net: gate ceiling >= 0.96 and +>=1pp over s12_full
run scripts/train_v5.py $COMMON --lambda-mlm 0.1 --lambda-mwm 0.1 \
    --proposal-topk 8 --lambda-proposal 0.2 --out "$R/s13_proposal.json"
run scripts/measure_ceiling_split.py --proposal --proposal-topk 8 \
    --ckpt "$R/s13_proposal.pt" --out "$R/ceiling_proposal.json"

# Stage 2.1 — probes off the full multitask checkpoint (incl. fixed sense_pause)
run scripts/probe_embeddings.py --ckpt "$R/s12_full.pt" --hand-baseline --out "$R/probes.json"

echo "=== $(date '+%H:%M:%S') ALL DONE -> $R ==="
