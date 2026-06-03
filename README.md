# latin-scan-ebm

A learned **energy-based model (EBM) for Latin hexameter scansion**. It rivals rule-based
dictionary scanners (e.g. *anceps*) on the *Aeneid* **with full coverage and zero
hand-encoded phonology** — learning metrical representations directly from <10,000 lines
of verse instead of encoding linguistic rules.

The model scans a line by **reranking an enumerated candidate set**: a small Transformer
encodes each line once, a structured energy head scores every candidate parse off that
single encoding, and the lowest-energy parse wins. Because all candidates are scored
against one cached encoding, the partition function `Z = logsumexp(−E)` stays an **exact
finite sum** — the EBM's core invariant.

## How it works

```
raw line ─▶ normalize ─▶ atomize ─▶ enumerate ─▶ MetricalEncoder ─▶ StructuredEnergyHead ─▶ argmin E
            (NFC,         (atoms,     (all valid    (Transformer,      (E_site+E_syll          (predicted
             lowercase)    bridges,    hexameter     encode ONCE        +E_foot+E_global)        scansion)
                           sites)      parses)       per line)
```

**Tokenizer & encoder** (`src/latin_ebm/encoder.py`)
- `MetricalTokenizer` turns a line into a candidate-invariant token sequence of **atoms**
  (vocalic nuclei) and **bridges** (consonantal material between them), interleaved so
  atom *k* → position `2k`, bridge *k* → `2k+1` (`n_tokens = 2·n_atoms − 1`).
- `MetricalEncoder` is a small Transformer (default `d_model=128`, 3 layers, 4 heads) that
  sums text + token-kind + word + word-boundary + positional embeddings, then produces
  per-token states `h` and an attention-pooled line vector `h_line`. Encoded **once per
  line** and reused across all candidates (`forward_batch` encodes a whole minibatch at once).

**Structured energy head** (`src/latin_ebm/energy_neural.py`)
- Four MLP sub-heads read each candidate's decisions off the cached `h` via pooling:
  `E = E_site + E_syll + E_foot + E_global` (ambiguity-site choices, per-syllable weights,
  per-foot types, and line-global features like caesura). The decomposition doubles as a
  learned, inspectable attribution of *why* a parse scored as it did.
- `NeuralScorer.predict` = `argmin(E)` over the candidate set; `candidate_energies` /
  `batched_energies` are the vectorized scorers that keep `Z` exact.

**Proposal network** (`src/latin_ebm/proposal.py`)
- Optionally augments the rule-enumerated candidate set with learned per-syllable weight
  bundles to lift the gold-reachability ceiling, deduplicated so the partition stays exact.

**Training** (`scripts/train_v5.py`) — multi-task: scansion NLL over the candidate set,
plus optional masked-atom (MLM) and masked-weight (MWM) self-supervision, and an optional
proposal-recall loss.

## Results

| Model | Aeneid 1–2 foot acc | Coverage |
|---|---|---|
| Learned EBM (multi-task) | **92.2%** | 100% (never abstains) |
| Learned EBM (supervised-only) | 91.4% | 100% |
| Hand-feature linear+MLP baseline (Phase F) | 92.6% | 100% |
| *anceps* (rule + dictionary) | 98.9% *on attempts* | ~88% (abstains otherwise) |

The learned model approaches the hand-engineered baseline **with no hand-crafted features**
and matches *anceps*'s effective coverage-weighted accuracy while never abstaining.
Gold-reachable ceiling of the candidate set is ~94.3%.

## Usage

All Python runs in the `latin-ebm` conda/mamba environment. Required data (relative to repo
root): `pedecerto-raw/VERG-aene.xml`, `data/MqDqMacrons.json`, `data/MorpheusMacrons.txt`.

```bash
# Train the full multi-task model, holding out Aeneid books 1–2 as the test set
mamba run -n latin-ebm python scripts/train_v5.py \
  --d-model 128 --n-layers 3 --epochs 30 --batch-size 16 \
  --lambda-mlm 0.1 --lambda-mwm 0.1 \
  --out results/run.json
# → writes results/run.json (foot/syllable/line/caesura accuracy + per-book)
#   and results/run.pt (encoder + head + vocab + config)

# Supervised-only (aux losses off)
mamba run -n latin-ebm python scripts/train_v5.py --lambda-mlm 0 --lambda-mwm 0 --out results/sup.json

# With the proposal network (ceiling lift)
mamba run -n latin-ebm python scripts/train_v5.py --proposal-topk 8 --lambda-proposal 0.2 --out results/prop.json
```

Key flags: `--test-books` (default `1,2`), `--d-model` (128), `--n-layers` (3),
`--epochs` (20), `--lr` (1e-3), `--batch-size` (16), `--lambda-mlm`/`--lambda-mwm` (0.1),
`--proposal-topk`/`--lambda-proposal` (0), `--patience`/`--min-delta` (early stopping),
`--threads`, `--compile`, `--seed`.

**Other scripts**
- `scripts/measure_ceiling_split.py` — gold-reachability ceiling and its split into
  *weight-unreachable* vs *no-candidate* buckets (`--proposal --ckpt run.pt` for the lifted ceiling).
- `scripts/probe_embeddings.py` — linear probes on the frozen `h_line` space (metrical +
  rhetorical structure) vs random-init and hand-feature controls.
- `scripts/run_full_eval.sh` — end-to-end driver: ceiling → supervised → multi-task → LOBO.

**Tests**
```bash
mamba run -n latin-ebm python -m pytest tests/
```
Covers the tokenizer/encoder, pooling, and the **exact-partition-function invariant**
(`logsumexp(−E)` equals brute-force `log Σ exp(−E)`), plus equivalence guards for the
vectorized and batched scorers.
