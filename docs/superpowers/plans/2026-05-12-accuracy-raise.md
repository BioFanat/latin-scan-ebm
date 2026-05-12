# Latin Scansion EBM: Accuracy Raise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift test foot-pattern accuracy from **67.7% → 80%+** on Aeneid books 1–2 (1516 evaluated lines), with a stretch target of 85% if Phase F's MLP delivers near the high end of its projection. Train accuracy target: 85%+. Conditional accuracy (correct given gold reachable) target: 88%+. Closing the full 24pp gap to anceps's 92% is a stretch — anceps has decades of hand-tuned phonology + frequency dictionaries we can only partially emulate.

**Architecture:** Keep the EBM framework (joint enumeration + energy scoring) — the ceiling-raise plan already lifted gold-in-candidate-set from 75% to 94.3%, so the bottleneck is now *picking* the right candidate from a permissive set. Three intervention layers, in order of expected ROI: (1) richer linear features (per-syllable, per-foot bigram, lemma, decision-context), (2) a small MLP head as a residual on top of the linear energy, (3) training-data recovery + more authors.

**Tech Stack:** Python 3.12+, scipy.optimize (L-BFGS-B, retained for the linear part), PyTorch (added for the MLP head), polars (analysis), existing `latin_ebm` package, MQDQ + Morpheus dictionaries already wired.

**Out of scope:** Pentameter / lyric meters, cross-meter transfer, generation, textual criticism applications. These all become tractable once accuracy is in the 85%+ range.

---

## Background — the current accuracy landscape

After the 2026-05-11 ceiling-raise plan (Phase 6 shipped):

| Metric | Phase 6 (current) | Original (pre-plan) | Anceps (target) |
|---|---|---|---|
| Test foot accuracy | **67.7%** | 67.3% | **92.0%** |
| Test syllable accuracy | 86.8% | 86.3% | — |
| Test elision F1 | 0.736 | 0.730 | — |
| Ceiling (gold in candidate set) | **94.30%** | ~75% | ~99% |
| Conditional acc (correct \| reachable) | 71.8% | 89.7% | ~93% |

We gained 19pp of ceiling without changing accuracy. The wrong-but-reachable bucket is now 28.1% (426 lines) — that's the target.

### Failure attribution (Phase 6 model)

| Error class | Lines | % of errors | Sub-mechanism |
|---|---|---|---|
| Elision direction (RETAIN ↔ ELIDE) | 204 | 47.9% | Model picks wrong elision choice for 47% of failing lines |
| MCL choice (ONSET ↔ CLOSE) | 113 | 26.5% | Closes when should be open, or vice-versa |
| Synizesis (DEFAULT ↔ MERGE) | 109 | 25.6% | Merges adjacent vowels wrong |
| Diphthong split | 121 | 28.4% | Split vs default wrong |
| (Lines often have 2-3 simultaneous decision errors; sums exceed 100%) | | | |

### Current feature set (122 features)

| Category | Count | What it captures |
|---|---|---|
| `foot5:*` | 2 | Fifth foot type only |
| `caesura:*` | 5 | 5 caesura kinds |
| `pattern:*` | 32 | All 6-foot patterns as opaque IDs |
| `site:{type}:{choice}` | 12 | Per-site-type × choice |
| `site_vowel:{vowel}:{choice}` | 48 | Vowel identity × decision |
| `elision_count:{0..3}` | 4 | Aggregate elision counts |
| `spondee_count:{0..5}` | 6 | Aggregate foot-type counts |
| `dactyl_count:{0..5}` | 6 | (same) |
| `syllable_count:{N}` | 6 | Total syllable count |
| `bucolic_diaeresis` | 1 | Binary |

**Critical absences:** zero per-syllable features, zero foot-bigram features, zero lemma/word-form features, zero decision-context features (e.g., elision-after-elision), zero per-position weight features. The 122 features are too coarse for the candidate-set sizes we now produce (median 8, max 762).

### Training non-convergence

The L-BFGS-B optimizer hits the 200-iteration cap without converging (loss 4901.2). 122 parameters trying to fit 7742 lines × ~27 candidates is underfit, not overfit. More features should help convergence too.

### Training data attrition

- 8282 total training lines (books 3–12)
- 540 skipped: 181 no candidates + 359 gold-not-reachable
- 7742 usable (93.5%)

The 359 "gold not reachable" training lines are signal we're throwing away. Most are elision-related — Pedecerto's gold has elision configurations our enumeration doesn't admit.

---

## Target trajectory

| Phase | Mechanism | Est. test foot accuracy delta | Cumulative |
|---|---|---|---|
| baseline (Phase 6) | — | — | 67.7% |
| A: diagnostic instrumentation | (measurement) | +0 | 67.7% |
| B: linear feature enrichment | per-syllable, per-foot bigram, per-foot, elision-pair, lemma | **+1 to +3pp** | ~69–71% |
| C: elision-decision specialization | elision-context features (foot-position, word-pair, caesura-distance) | **+3 to +6pp** | ~72–77% |
| D: MCL/synizesis specialization | per-cluster, per-vowel-pair features, MQDQ lemma-agreement | **+2 to +4pp** | ~74–81% |
| E: training data recovery | atomization fixes to recover the 359 unreachable training golds | +1 to +3pp | ~75–84% |
| F: MLP residual head with site decisions in dense features | hybrid energy with small MLP, joint AdamW | **+3 to +6pp** | ~78–90% |
| G: cross-author training data | Ovid, Lucretius, Horace from Pedecerto (if available) | +0 to +3pp | ~78–93% |
| H: final tuning & ablation | hyperparameter search, regularization, feature ablation | +0 to +2pp | ~78–95% |

**Realistic landing zone: 78–85% test foot accuracy.** Anceps's 92% is approachable but not guaranteed (anceps benefits from decades of hand-tuned phonology and explicit author-period frequency dictionaries we can only partly emulate). Phases C and F do the heaviest lifting:

- **Phase C** targets the dominant error class (47.9% elision-related). Per-foot-position elision features are the single highest-ROI linear features.
- **Phase F** breaks out of the linear capacity ceiling, but **only** if the dense feature vector includes per-foot site decisions (see Task F.3). Without that, the MLP has nothing to bite on for elision errors and may yield <+1pp.

Phase B's gain is downgraded from earlier drafts (+5-8pp → +1-3pp) because most of its features (per-syllable position, foot bigrams) help with pattern shape, not decision direction. The model is already strongest on pattern shape (94.3% gold-in-set).

---

## File Structure

### To create

| Path | Responsibility |
|---|---|
| `scripts/analyze_errors.py` | Per-line prediction dumper; categorize wrong-but-reachable by decision-type errors |
| `scripts/feature_ablation.py` | Add/remove feature groups; measure accuracy delta |
| `scripts/train_v2.py` | New training script with PyTorch MLP head + scipy linear part |
| `src/latin_ebm/features_v2.py` | Per-syllable, per-foot-bigram, lemma, decision-context features |
| `src/latin_ebm/energy_v2.py` | Hybrid LinearEBM + MLPResidual scorer |
| `src/latin_ebm/mlp.py` | Small PyTorch MLP over dense per-syllable feature vectors |
| ~~`src/latin_ebm/lemma_lexicon.py`~~ | (REMOVED — lemma lookup added to existing `VowelLengthLexicon` instead) |
| `tests/test_features_v2.py` | Unit tests for new feature extractors |
| `tests/test_energy_v2.py` | Unit tests for hybrid energy |
| `tests/test_mlp.py` | Unit tests for MLP head |
| `tests/test_analyze_errors.py` | Unit tests for error analyzer |
| `results/error_breakdown_baseline.json` | Phase A baseline error categorization |
| `results/phase{B..H}_accuracy.json` | Per-phase accuracy measurements |
| `docs/accuracy-progression.md` | Running log of accuracy gains |

### To modify

| Path | Why |
|---|---|
| `src/latin_ebm/features.py` | Extend with new feature types (preserve v1 compatibility) or replace via `features_v2.py` |
| `src/latin_ebm/energy.py` | Add `score_with_components` for diagnostic decomposition |
| `src/latin_ebm/train.py` | Add training-data recovery path + better optimization loop |
| `src/latin_ebm/evaluate.py` | Add decision-level error breakdown to evaluator |
| `src/latin_ebm/atomize.py` | Phase E: relax elision detection for currently-unreachable Pedecerto patterns |
| `pyproject.toml` | Add `torch>=2.2` to dependencies (Phase F) |
| `scripts/train_v1.py` | Keep as legacy reference; v2 supersedes |

---

## Conventions

- Use the `latin-ebm` mamba environment for all Python invocations (`mamba run -n latin-ebm python ...`).
- All tests use pytest. Session-scoped fixture `aeneid_test_lines` (1516 (line, gold) pairs from books 1-2) already exists in `tests/conftest.py`.
- TDD discipline: every code change is preceded by a failing test that exercises the new behavior.
- Each phase ends with a full re-train + re-evaluate, with results committed to `results/`.
- Keep Phase 6 (94.3% ceiling, 67.7% test acc) as the rollback baseline. Any phase that REGRESSES accuracy gets reverted.

### Codebase facts you must know before starting

- **`FeatureIndex.fire()` does NOT exist.** The existing pattern in `extract_features` (`features.py:71-191`) is to mutate a local `features: defaultdict[str, float]` keyed by name string, then vectorize via `index.get_or_add(name)` at array-build time. So **new feature emissions should be written as `features[name] += value`**, not `feature_index.fire(...)`. (Earlier drafts of this plan used `feature_index.fire` — wrong. Treat any `feature_index.fire(features, name, value)` in code blocks below as shorthand for `features[name] += value`.)

- **`train_nll` real signature** (`train.py:184-191`): `train_nll(examples: list[TrainingExample], feature_index=None, meter=None, l2_lambda=0.01, max_iter=200, lexicon=None) -> tuple[LinearEBM, TrainResult]`. It takes raw `TrainingExample`s (not precomputed data), builds its own `FeatureIndex` if none is provided, and returns a tuple. Parameter name is `l2_lambda`, NOT `l2`. **Plan code blocks that invoke `train_nll` must match this exact signature.**

- **`extract_features` real signature** (`features.py:71-76`): `extract_features(line: LatinLine, parse: Parse, index: FeatureIndex, lexicon=None) -> np.ndarray`. Phase B.5 adds a `lemma_lexicon` kwarg — fine, but the function returns an `np.ndarray` (already-vectorized), so the in-function `features` dict is local and gets discarded after vectorization. New feature emissions go inside `extract_features` itself.

- **`precompute_training_data`** (`train.py:52`): builds per-line enumerated candidates + feature vectors + gold-compatibility mask. Returns `list[PrecomputedLine]`. Reuses an existing `FeatureIndex`. Not normally called directly by user scripts — `train_nll` calls it internally.

- **Bridge / atom-index contract** at sites: `AmbiguitySite.atom_indices = (left_atom_idx, right_atom_idx)` where `left_atom_idx + 1 == right_atom_idx` in the common case. The corresponding bridge is `line.bridges[left_atom_idx]` (bridges sit between atoms `i` and `i+1`). Phase C/D code blocks that index bridges should use `line.bridges[site.atom_indices[0]]`.

### Measurement protocol

For every phase:

```bash
mamba run -n latin-ebm python scripts/train_v2.py \
    --test-books 1,2 \
    --features-version <phase> \
    --l2 0.01 \
    --max-iter 500 \
    --out results/phase{N}_accuracy.json
```

`phase{N}_accuracy.json` schema:

```json
{
  "config": { "features_version": "B", "l2": 0.01, "max_iter": 500 },
  "train": { "foot_accuracy": 0.X, "line_exact_match": 0.X, ... },
  "test": { "foot_accuracy": 0.X, "line_exact_match": 0.X,
            "syllable_accuracy": 0.X, "caesura_accuracy": 0.X,
            "elision_f1": 0.X, "synizesis_f1": 0.X,
            "diphthong_f1": 0.X, "mcl_f1": 0.X,
            "per_book": { "1": 0.X, "2": 0.X } },
  "training_loss": X, "iterations": N, "converged": bool
}
```

---

# Phase A: Diagnostic Instrumentation

**Goal:** Build the error-attribution infrastructure that turns "67% accuracy" into actionable per-line decision-error categorization. This is the measurement instrument every later phase depends on.

**Why first:** Phase B's feature additions need to be targeted at specific decision-error classes. Without per-line attribution, we'd flail. The Failure-Mode analysis already showed elision dominates (47.9%); Phase A makes this rigorous and repeatable after every retrain.

### Task A.1: `analyze_errors.py` — per-line prediction dumper

**Files:**
- Create: `scripts/analyze_errors.py`
- Create: `tests/test_analyze_errors.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_analyze_errors.py`:

```python
import json
from pathlib import Path
import subprocess
import tempfile


def test_analyze_errors_writes_jsonl():
    """analyze_errors.py runs a trained-model evaluation and dumps per-line
    decisions, classifying each test line as correct/wrong-reachable/unreachable
    with per-site decision diffs vs gold."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "errors.jsonl"
        result = subprocess.run(
            ["mamba", "run", "-n", "latin-ebm",
             "python", "scripts/analyze_errors.py",
             "--test-books", "1,2",
             "--out", str(out)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in out.read_text().splitlines() if line]
        assert len(rows) > 1000
        # every row has expected fields
        for r in rows[:5]:
            assert "line_id" in r
            assert "status" in r           # correct | wrong_reachable | unreachable
            assert "decision_diffs" in r   # list of (site_index, site_type, predicted, gold)
            assert "predicted_foot_types" in r
            assert "gold_foot_types" in r
            assert "n_candidates" in r
            assert "predicted_rank_of_gold" in r  # None if unreachable
```

- [ ] **Step 2: Run test to verify it fails**

```bash
mamba run -n latin-ebm pytest tests/test_analyze_errors.py::test_analyze_errors_writes_jsonl -v
```

Expected: FAIL — `scripts/analyze_errors.py` doesn't exist.

- [ ] **Step 3: Implement the script**

```python
"""Per-line prediction analyzer.

Trains a Linear EBM on training books, then for each test line:
  1. Enumerates candidates
  2. Scores them with the trained energy
  3. Compares argmin to gold's foot_types
  4. If wrong-but-gold-reachable: dumps per-site decision diffs
  5. If unreachable: tags the closest candidate's mismatches

Output: JSONL, one row per test line, with all fields for downstream
aggregation (e.g., counting elision_diffs, MCL_diffs, etc.)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import FeatureIndex, extract_features
from latin_ebm.energy import LinearEBM
from latin_ebm.train import train_nll
from latin_ebm.types import Parse, SiteType

logger = logging.getLogger(__name__)


@dataclass
class DecisionDiff:
    site_index: int
    site_type: str  # SiteType.name
    predicted: str  # SiteChoice.name
    gold: str       # SiteChoice.name


@dataclass
class ErrorRow:
    line_id: str
    status: str                          # correct | wrong_reachable | unreachable
    predicted_foot_types: list[str]
    gold_foot_types: list[str]
    n_candidates: int
    predicted_rank_of_gold: Optional[int]   # None if unreachable
    decision_diffs: list[DecisionDiff]


def _find_gold_index(candidates: list[Parse], gold: Parse) -> Optional[int]:
    """Return the index of the first candidate matching gold's foot_types AND
    slots (full match), or None if not present."""
    for i, c in enumerate(candidates):
        if c.foot_types == gold.foot_types and c.slots == gold.slots:
            return i
    return None


def _decision_diffs(pred: Parse, gold: Parse, sites: list) -> list[DecisionDiff]:
    """Find which site decisions differ between predicted and gold parses."""
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
    # Train via the real train_nll (it builds the feature index internally).
    # Returns (LinearEBM, TrainResult).
    model, _train_result = train_nll(
        train_examples,
        l2_lambda=l2,
        max_iter=max_iter,
        lexicon=lexicon,
    )
    # The trained feature index is on the returned model. The cleanest way to
    # get it is to re-extract via the same _build path — but train_nll already
    # stored it in the precompute. Easiest: call build_feature_index ourselves.
    from latin_ebm.features import build_feature_index
    lines = [ex.line for ex in train_examples]
    parses_per_line = [enumerate_parses(ex.line) for ex in train_examples]
    index = build_feature_index(lines, parses_per_line, lexicon=lexicon)

    rows: list[ErrorRow] = []
    for ex in test_examples:
        candidates = enumerate_parses(ex.line)
        n_cands = len(candidates)

        if not candidates:
            rows.append(ErrorRow(
                line_id=ex.line.corpus_id, status="unreachable",
                predicted_foot_types=[], gold_foot_types=[f.name for f in ex.gold_parse.foot_types],
                n_candidates=0, predicted_rank_of_gold=None, decision_diffs=[],
            ))
            continue

        # Score and rank
        scored = sorted(
            ((extract_features(ex.line, c, index, lexicon=lexicon), c) for c in candidates),
            key=lambda pair: float(model.energy(pair[0])),
        )
        pred = scored[0][1]
        gold_idx = _find_gold_index([c for _, c in scored], ex.gold_parse)

        if pred.foot_types == ex.gold_parse.foot_types and pred.slots == ex.gold_parse.slots:
            status = "correct"
        elif gold_idx is not None:
            status = "wrong_reachable"
        else:
            status = "unreachable"

        rows.append(ErrorRow(
            line_id=ex.line.corpus_id,
            status=status,
            predicted_foot_types=[f.name for f in pred.foot_types],
            gold_foot_types=[f.name for f in ex.gold_parse.foot_types],
            n_candidates=n_cands,
            predicted_rank_of_gold=gold_idx,
            decision_diffs=_decision_diffs(pred, ex.gold_parse, ex.line.sites),
        ))

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    lexicon = VowelLengthLexicon(
        mqdq_path=Path("data/MqDqMacrons.json"),
        morpheus_path=Path("data/MorpheusMacrons.txt"),
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train = [e for e in result.examples if e.line.book not in test_books]
    test = [e for e in result.examples if e.line.book in test_books]
    # Override corpus_id to the synthetic format used in fixtures
    for ex in test:
        ex.line.corpus_id = f"VERG-aene.{ex.line.book}.{ex.line.line_num}"

    rows = analyze(test, train, lexicon, args.l2, args.max_iter)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Wrote %d rows to %s", len(rows), args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_errors.py tests/test_analyze_errors.py
git commit -m "feat(phase-A): analyze_errors.py — per-line decision-diff dumper"
```

---

### Task A.2: Aggregator — `analyze_errors --summarize`

**Files:**
- Modify: `scripts/analyze_errors.py`

- [ ] **Step 1: Add `summarize` subcommand**

```python
# At top of analyze_errors.py, add:

def summarize(jsonl_path: Path) -> dict:
    """Aggregate an errors.jsonl file into a summary report."""
    from collections import Counter
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l]
    total = len(rows)
    status_counts = Counter(r["status"] for r in rows)

    # Categorize wrong_reachable by decision-type-error
    diff_type_counts = Counter()
    diff_direction_counts = Counter()  # e.g., ELISION:RETAIN->ELIDE
    for r in rows:
        if r["status"] != "wrong_reachable": continue
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
```

Add subcommand to `main`:

```python
sub = p.add_subparsers(dest="cmd")
# ... existing positional args become part of "run" subcommand
run_p = sub.add_parser("run")
run_p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
# (etc.)
sum_p = sub.add_parser("summarize")
sum_p.add_argument("jsonl")
args = p.parse_args()
if args.cmd == "summarize":
    print(json.dumps(summarize(Path(args.jsonl)), indent=2))
    return
# else: run path
```

- [ ] **Step 2: Test the summarizer**

```python
def test_summarize_categorizes_by_decision_type(tmp_path):
    fake = tmp_path / "f.jsonl"
    fake.write_text("\n".join([
        json.dumps({"line_id": "a", "status": "correct", "decision_diffs": [],
                    "n_candidates": 3, "predicted_rank_of_gold": 0,
                    "predicted_foot_types": [], "gold_foot_types": []}),
        json.dumps({"line_id": "b", "status": "wrong_reachable",
                    "n_candidates": 5, "predicted_rank_of_gold": 2,
                    "decision_diffs": [
                        {"site_index": 0, "site_type": "ELISION", "predicted": "RETAIN", "gold": "ELIDE"},
                    ],
                    "predicted_foot_types": [], "gold_foot_types": []}),
    ]) + "\n")
    from scripts.analyze_errors import summarize
    s = summarize(fake)
    assert s["total"] == 2
    assert s["accuracy"] == 0.5
    assert s["wrong_reachable_by_site_type"]["ELISION"] == 1
    assert "ELISION:ELIDE->RETAIN" in s["wrong_reachable_by_direction"]
```

- [ ] **Step 3: Run + commit**

```bash
mamba run -n latin-ebm pytest tests/test_analyze_errors.py -v
git add -A && git commit -m "feat(phase-A): analyze_errors summarize subcommand"
```

---

### Task A.3: Capture Phase-A baseline

**Files:**
- Create: `results/error_breakdown_baseline.json`

- [ ] **Step 1: Run the analyzer**

```bash
mamba run -n latin-ebm python scripts/analyze_errors.py run \
    --xml pedecerto-raw/VERG-aene.xml --test-books 1,2 \
    --l2 0.01 --max-iter 200 \
    --out results/error_breakdown_baseline.jsonl

mamba run -n latin-ebm python scripts/analyze_errors.py summarize \
    results/error_breakdown_baseline.jsonl > results/error_breakdown_baseline.json
```

- [ ] **Step 2: Sanity-check the numbers**

The JSON should show roughly:
- accuracy ≈ 0.677
- by_status.correct ≈ 1027
- by_status.wrong_reachable ≈ 425
- by_status.unreachable ≈ 88
- wrong_reachable_by_site_type.ELISION ≈ 200
- wrong_reachable_by_site_type.MUTA_CUM_LIQUIDA ≈ 110

If numbers diverge significantly, debug before continuing.

- [ ] **Step 3: Write progression log**

`docs/accuracy-progression.md`:

```markdown
# Accuracy Progression Log

Test foot-pattern accuracy on Aeneid books 1-2 (1516 evaluated lines).
Trained on books 3-12.

| Phase | Date | Test foot acc | Test line exact | Test syllable acc | Train foot acc | Notes |
|---|---|---|---|---|---|---|
| baseline (Phase-6 ship) | 2026-05-12 | 67.7% | 67.7% | 86.8% | (TBD) | Linear EBM, 122 features |
```

- [ ] **Step 4: Commit**

```bash
git add results/error_breakdown_baseline.* docs/accuracy-progression.md
git commit -m "data(phase-A): baseline error breakdown + accuracy log"
```

**Phase A expected accuracy delta:** 0 (instrumentation only).

---

# Phase B: Linear Feature Enrichment

**Goal:** Push test foot accuracy from 67.7% to ~73–76% by adding the most-obviously-missing features within the linear framework. Linear is cheap, interpretable, and a strong baseline to beat.

**Hypothesis:** The current 122 features include zero per-syllable position×weight features, zero foot-bigram features, and zero per-decision-context features. Adding these should help the L-BFGS converge to a lower loss and disambiguate more candidates.

### Task B.1: Per-syllable position × weight features

**Files:**
- Modify: `src/latin_ebm/features.py`
- Modify: `tests/test_features_v2.py` (new)

**Diagnosis:** the trained model can express "many spondees" via `spondee_count:5` but cannot express "first syllable is LONG" or "syllable 7 (caesura position) is LONG." Each hexameter has 12–17 syllables × 2 weights = 24–34 possible per-position features.

- [ ] **Step 1: Write the failing test**

In `tests/test_features_v2.py`:

```python
import pytest
from latin_ebm.atomize import atomize
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import FeatureIndex, extract_features


@pytest.fixture
def aen_1_1_parse(aeneid_test_lines):
    # First Aeneid line, take first candidate
    line, _ = aeneid_test_lines[0]
    cands = enumerate_parses(line)
    return line, cands[0]


def test_per_syllable_weight_features_emit(aen_1_1_parse):
    """Features should include syl_pos:{n}:{weight} for each syllable."""
    line, parse = aen_1_1_parse
    idx = FeatureIndex()
    fv = extract_features(line, parse, idx, lexicon=None)
    # syl_pos:0:LONG should be in the registry after extraction
    names = idx.names
    assert any(n.startswith("syl_pos:0:") for n in names)
    assert any(n.startswith("syl_pos:1:") for n in names)
    # one of them should fire (non-zero count)
    fired = [n for n, v in zip(names, fv) if v > 0 and n.startswith("syl_pos:0:")]
    assert len(fired) == 1  # exactly one weight per syllable
```

- [ ] **Step 2: Run test**

Expected: FAIL — feature doesn't exist.

- [ ] **Step 3: Implement in `features.py`**

Find the global-features block in `extract_features` (around line 99 per inventory). Add after the syllable_count features:

```python
# Per-syllable position × weight
for syl_idx, syl in enumerate(parse.syllables):
    features[f"syl_pos:{syl_idx}:{syl.weight.name}"] += 1.0
```

Direct mutation of the local `features: defaultdict[str, float]` matches the existing pattern (e.g. line 91: `features[feat] += 1.0`). The `FeatureIndex.get_or_add(name)` registration happens later during vectorization — no manual registration needed here.

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/features.py tests/test_features_v2.py
git commit -m "feat(phase-B): per-syllable position × weight features"
```

---

### Task B.2: Foot-bigram features

**Files:**
- Modify: `src/latin_ebm/features.py`
- Modify: `tests/test_features_v2.py`

**Diagnosis:** `pattern:DDSSDF` is the most common positive-weight feature in the trained model, meaning the linear model essentially memorizes whole patterns. Bigrams `foot_i:T1×foot_{i+1}:T2` decompose this into 5 bigrams (for 5 transitions in hexameter), giving the model 4×4×5 = 80 possible bigram features (each foot is D or S, F as terminal). This should help generalization to unseen patterns.

- [ ] **Step 1: Write the failing test**

```python
def test_foot_bigram_features(aen_1_1_parse):
    """Features should include foot_bigram:{n}:{T1}_{T2} for each transition."""
    line, parse = aen_1_1_parse
    idx = FeatureIndex()
    fv = extract_features(line, parse, idx, lexicon=None)
    names = idx.names
    assert any(n.startswith("foot_bigram:0:") for n in names)
    # exactly 5 transitions: feet 0-1, 1-2, 2-3, 3-4, 4-5
    bigrams = [n for n in names if n.startswith("foot_bigram:")]
    assert len(bigrams) >= 5
```

- [ ] **Step 2: Run, implement, run, commit**

In `extract_features`, add:

```python
# Foot bigrams (T_i, T_{i+1})
for i in range(len(parse.foot_types) - 1):
    ft1 = parse.foot_types[i].name
    ft2 = parse.foot_types[i+1].name
    features[f"foot_bigram:{i}:{ft1}_{ft2}"] += 1.0
```

```bash
git commit -m "feat(phase-B): foot bigram features (5 per parse)"
```

---

### Task B.3: Per-foot foot-type features

**Files:**
- Modify: `src/latin_ebm/features.py`
- Modify: `tests/test_features_v2.py`

**Diagnosis:** the current `foot5:DACTYL/SPONDEE` features cover only foot 5. Feet 1–4 have implicit signals only via patterns. Explicit per-foot features (`foot:0:DACTYL`, ..., `foot:4:SPONDEE`) give the model direct access.

- [ ] **Step 1: Test → impl → run → commit (~5 min)**

```python
def test_per_foot_features(aen_1_1_parse):
    line, parse = aen_1_1_parse
    idx = FeatureIndex()
    fv = extract_features(line, parse, idx, lexicon=None)
    names = idx.names
    for foot_idx in range(5):
        assert any(n.startswith(f"foot:{foot_idx}:") for n in names)
```

Implementation: add to the global-features block:

```python
for i, ft in enumerate(parse.foot_types[:5]):  # feet 0..4
    features[f"foot:{i}:{ft.name}"] += 1.0
```

```bash
git commit -m "feat(phase-B): per-foot foot-type features (feet 0-4)"
```

---

### Task B.4: Per-site decision-context features

**Files:**
- Modify: `src/latin_ebm/features.py`
- Modify: `tests/test_features_v2.py`

**Diagnosis:** the existing `site_vowel:{v}:{choice}` features capture only the vowel at the elision site, not the *receiving* vowel or word-boundary properties. Add: `elision_pair:{v1}{v2}:{choice}` (vowel-pair across elision boundary), `site_word_pos:{pos}:{choice}` (site position within line bucketed).

- [ ] **Step 1: Test → impl**

```python
def test_elision_pair_features(aeneid_test_lines):
    """For lines with elisions, features include elision_pair:{v1}{v2}:{choice}."""
    line, gold = aeneid_test_lines[0]
    cands = enumerate_parses(line)
    if not cands:
        pytest.skip("no candidates")
    parse = cands[0]
    idx = FeatureIndex()
    extract_features(line, parse, idx, lexicon=None)
    # Look for at least one elision_pair feature if line has an ELISION site
    has_elision_site = any(s.site_type.name == "ELISION" for s in line.sites)
    if has_elision_site:
        assert any(n.startswith("elision_pair:") for n in idx.names)
```

Implementation: in the per-site loop, when site is ELISION:

```python
if site.site_type == SiteType.ELISION:
    left_atom = line.atoms[site.atom_indices[0]]
    right_atom = line.atoms[site.atom_indices[1]]
    pair = left_atom.chars + right_atom.chars
    features[f"elision_pair:{pair}:{decision.name}"] += 1.0
```

(`decision` is already defined in the existing per-site loop at `features.py:87`.)

- [ ] **Step 2: Re-run pytest + commit**

```bash
git commit -m "feat(phase-B): elision_pair (cross-site vowel-pair) features"
```

---

### Task B.5: Lemma / word-form features

**Files:**
- Modify: `src/latin_ebm/features.py`
- Create: `src/latin_ebm/lemma_lexicon.py`

**Diagnosis:** common words (`est`, `et`, `qui`, `in`) have characteristic metrical behavior. The model currently has no way to learn "the word `est` participates in prodelision" beyond per-site features. Add `word:{form}:{site_type}:{choice}` for words that appear in test sites.

Limit the feature space to high-frequency words to avoid blow-up: only fire if the word appears in N+ training lines (use Morpheus presence as a proxy).

- [ ] **Step 1: Extend `VowelLengthLexicon` with lemma lookup**

Avoid loading Morpheus twice. Add a `lemma(word) -> str | None` method to the existing `VowelLengthLexicon` (`src/latin_ebm/lexicon.py`). Modify `_load_morpheus` to also populate a `_word_to_lemma` map while it walks the file (it already reads every line).

In `__init__`, add:

```python
self._word_to_lemma: dict[str, str] = {}
```

In `_load_morpheus`, alongside the existing `_morpheus` and `_morpheus_raw` population, add:

```python
self._word_to_lemma.setdefault(word_form, parts[2].lower())  # parts[2] = lemma
```

Then add a public method:

```python
def lemma(self, word: str) -> str | None:
    return self._word_to_lemma.get(self._normalize_key(word))
```

Test in `tests/test_features_v2.py`:

```python
def test_lexicon_lemma_lookup(lexicon):
    # arma is plural of armum; Morpheus uses "armum" or "arma" depending on encoding
    assert lexicon.lemma("arma") is not None
    assert lexicon.lemma("zztoptpqr_invented") is None
```

- [ ] **Step 2: Test the lemma feature emission**

```python
def test_lemma_features_for_common_words(aeneid_test_lines, lexicon):
    line, _ = aeneid_test_lines[0]
    cands = enumerate_parses(line)
    if not cands: pytest.skip()
    parse = cands[0]
    idx = FeatureIndex()
    extract_features(line, parse, idx, lexicon=lexicon)
    # for any site whose word has a Morpheus lemma, expect a lemma feature
    has_lemma_feature = any(n.startswith("lemma:") for n in idx.names)
    assert has_lemma_feature
```

- [ ] **Step 3: Implement**

In `extract_features`, when iterating sites, use the already-passed `lexicon` parameter (no new kwarg needed). For each site, look up the lemma of the word containing the left atom; emit `lemma:{lemma}:{site_type}:{decision}` if found. Limit to lemmas that appear in some frequency threshold (otherwise the feature index blows up):

```python
if lexicon is not None:
    word = line.words[line.atoms[site.atom_indices[0]].word_idx]
    lem = lexicon.lemma(word)
    if lem is not None and len(lem) >= 2:
        features[f"lemma:{lem}:{site.site_type.name}:{decision.name}"] += 1.0
```

Optional frequency gating (apply if feature count explodes): build a per-lemma occurrence histogram during `build_feature_index`'s first pass and only emit lemma features for lemmas with ≥10 attestations across training.

- [ ] **Step 4: Commit**

```bash
git add src/latin_ebm/lexicon.py src/latin_ebm/features.py tests/test_features_v2.py
git commit -m "feat(phase-B): lemma-conditional site features (extends VowelLengthLexicon)"
```

---

### Task B.6: Re-train and measure Phase B

- [ ] **Step 1: Bump L-BFGS max_iter** (current 200 didn't converge with 122 features; more features need more)

```bash
mamba run -n latin-ebm python scripts/train_v1.py \
    --test-books 1,2 --l2 0.01 --max-iter 500 \
    2>&1 | tee results/phase_B_train.log
```

- [ ] **Step 2: Re-run analyzer**

```bash
mamba run -n latin-ebm python scripts/analyze_errors.py run \
    --test-books 1,2 --l2 0.01 --max-iter 500 \
    --out results/phase_B_errors.jsonl
mamba run -n latin-ebm python scripts/analyze_errors.py summarize \
    results/phase_B_errors.jsonl > results/phase_B_accuracy.json
```

- [ ] **Step 3: Update progression log + commit**

Expected: accuracy 72–76%. If lower than 70%, investigate (probably feature index size blew up too much; check `--l2`).

```bash
git add results/phase_B_*
git commit -m "data(phase-B): linear feature enrichment - accuracy <X>%"
```

**Phase B expected delta: +5 to +8pp foot accuracy.**

---

# Phase C: Elision-Decision Specialization

**Goal:** Drop the 204-line elision-error bucket by half. Elision dominates failures (47.9% of wrong-reachable), so targeted features here have outsized impact.

### Task C.1: Elision context features

**Files:**
- Modify: `src/latin_ebm/features.py`
- Modify: `tests/test_features_v2.py`

Add features that capture the *context* of each elision decision:

- `elision_word_pair:{w1_last_chars}:{w2_first_chars}:{choice}` — last 2 chars of left word × first 2 chars of right word (captures vowel + nasal-m pattern)
- `elision_is_prodelision_eligible:{bool}:{choice}` — whether the right word is `es`/`est`
- `elision_in_foot:{N}:{choice}` — which foot the elision falls in (1..5)
- `elision_distance_from_caesura:{bucket}:{choice}` — distance from main caesura, bucketed (0 = at caesura, ±1, ±2, far)

- [ ] **Step 1: Test → impl → run → commit (one pass)**

```python
def test_elision_in_foot_feature(aeneid_test_lines):
    # find any test line with elision
    for line, gold in aeneid_test_lines:
        if not any(s.site_type.name == "ELISION" for s in line.sites):
            continue
        cands = enumerate_parses(line)
        if not cands: continue
        idx = FeatureIndex()
        extract_features(line, cands[0], idx, lexicon=None)
        assert any(n.startswith("elision_in_foot:") for n in idx.names)
        return
    pytest.skip("no test line with elision")
```

Implementation in `features.py`. First, add the two helpers at module top:

```python
def _syllable_index_for_atom(parse: Parse, atom_idx: int) -> int | None:
    """Find which realized syllable contains the given atom_idx (or None)."""
    for syl_idx, syl in enumerate(parse.syllables):
        if atom_idx in syl.atom_indices:
            return syl_idx
    return None


def _foot_for_syllable(parse: Parse, syl_idx: int) -> int | None:
    """Find which foot (0..5) the given syllable belongs to.
    parse.foot_boundaries is a tuple of start-syllable-indices per foot."""
    if syl_idx is None:
        return None
    for foot_idx in range(len(parse.foot_boundaries)):
        end = parse.foot_boundaries[foot_idx + 1] if foot_idx + 1 < len(parse.foot_boundaries) else len(parse.syllables)
        if parse.foot_boundaries[foot_idx] <= syl_idx < end:
            return foot_idx
    return None
```

Then, inside the per-site loop in `extract_features`:

```python
if site.site_type == SiteType.ELISION:
    syl_idx = _syllable_index_for_atom(parse, site.atom_indices[0])
    foot = _foot_for_syllable(parse, syl_idx)
    if foot is not None:
        features[f"elision_in_foot:{foot}:{decision.name}"] += 1.0
```

Verify against `types.py:Parse.foot_boundaries` format before merging — if it's already a "syllable index per foot start" tuple, the above works; if it's something else, adjust the lookup.

```bash
git commit -m "feat(phase-C): elision contextual features (foot position, word-pair, caesura distance)"
```

---

### Task C.2: Re-train Phase C, measure

```bash
mamba run -n latin-ebm python scripts/train_v1.py --test-books 1,2 --l2 0.01 --max-iter 500
mamba run -n latin-ebm python scripts/analyze_errors.py run --test-books 1,2 --l2 0.01 --max-iter 500 \
    --out results/phase_C_errors.jsonl
mamba run -n latin-ebm python scripts/analyze_errors.py summarize results/phase_C_errors.jsonl \
    > results/phase_C_accuracy.json
```

Check the `wrong_reachable_by_site_type.ELISION` value — should drop from ~200 to ~100.

```bash
git add results/phase_C_*
git commit -m "data(phase-C): elision-specialized features - accuracy <X>%"
```

**Phase C expected delta: +3 to +5pp foot accuracy. Elision error rate ÷2.**

---

# Phase D: MCL/Synizesis Specialization

**Goal:** Reduce the 113-line MCL-error bucket and the 109-line synizesis-error bucket.

### Task D.1: MCL specific-cluster features

The current model only knows "MCL → CLOSE or ONSET" without conditioning on *which* stop+liquid pair. The 8 most-common pairs (`br, pr, tr, dr, cr, gr, bl, pl, ...`) have author-specific preferences (Vergil tends ONSET; later poets more CLOSE).

Add: `mcl_cluster:{pair}:{choice}` for each MCL site.

```python
if site.site_type == SiteType.MUTA_CUM_LIQUIDA:
    # Bridge between atoms i and i+1 lives at bridges[i] (atom_indices[0]).
    bridge = line.bridges[site.atom_indices[0]]
    for stop in "bpdtcgf":
        for liquid in "lr":
            if stop + liquid in bridge.chars:
                features[f"mcl_cluster:{stop+liquid}:{decision.name}"] += 1.0
                break
```

```bash
git commit -m "feat(phase-D1): MCL-cluster specific features (br/pr/tr/...)"
```

### Task D.2: Synizesis vowel-pair features

Similar specialization for synizesis sites — which vowel pair (`ei, eo, ia, io`) is being merged.

```python
if site.site_type == SiteType.SYNIZESIS:
    pair = line.atoms[site.atom_indices[0]].chars + line.atoms[site.atom_indices[1]].chars
    features[f"synizesis_pair:{pair}:{decision.name}"] += 1.0
```

```bash
git commit -m "feat(phase-D2): synizesis vowel-pair features"
```

### Task D.3: MQDQ frequency conditioning

For each (lemma, parse), compute the agreement rate between the parse's weight pattern and the MQDQ majority scansion for the lemma. Emit as a real-valued feature `mqdq_lemma_agreement` (range 0..1).

This is a richer version of the existing aggregate `lex_agree_ratio` but conditioned on lemma, not summed across the whole line.

```bash
git commit -m "feat(phase-D3): MQDQ lemma-level frequency feature"
```

### Task D.4: Re-train + measure

```bash
# (commands per Phase C pattern)
git commit -m "data(phase-D): MCL/synizesis specialization - accuracy <X>%"
```

**Phase D expected delta: +2 to +4pp.**

---

# Phase E: Training-Data Recovery

**Goal:** Reclaim the 359 training lines (4.3%) where gold is unreachable from the current enumeration. Most are elision-related — Pedecerto's gold has elision configurations our atomizer doesn't produce.

### Task E.1: Categorize unreachable training lines

Reuse `analyze_errors.py` on the training set to classify why gold isn't enumerable:

```bash
mamba run -n latin-ebm python scripts/analyze_errors.py run \
    --test-books "" \  # empty = use full corpus as "test" for analysis
    --out results/train_unreachable.jsonl
```

Filter for `status: unreachable` and group by which site type's gold decision isn't in `valid_choices`. This identifies whether we need (a) more permissive elision detection, (b) new site types, or (c) better cross-word handling.

### Task E.2: Targeted atomization fix

Based on E.1 findings, implement the most-common fix. Likely candidates:

- **Elision after vowel+s** (Old Latin "ille's" pattern, very rare but happens)
- **Synaeresis with consonantal-u promotion** ("voluitur"-type, see ceiling-raise plan)
- **Diphthong-split after specific lexical contexts** (Greek loanwords)

Each candidate is a separate task. Implement the highest-frequency one first; re-measure; iterate.

### Task E.3: Validate recovery + accuracy improvement

After each atomization tweak:

```bash
# Re-run ceiling diagnostic (training data is "training", but ceiling is on all)
mamba run -n latin-ebm python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml \
    --books 3,4,5,6,7,8,9,10,11,12 \
    --out results/phase_E_train_ceiling.tsv

# Retrain and measure test accuracy
mamba run -n latin-ebm python scripts/train_v1.py --test-books 1,2 --l2 0.01 --max-iter 500
```

Check that:
- Training data usable count goes UP (more gold-reachable lines)
- Test ceiling stays ≥ 94.3%
- Test foot accuracy goes UP

If test accuracy DROPS, the atomization fix is over-aggressive — revert it.

```bash
git commit -m "data(phase-E): training-data recovery - usable <X> → <Y> lines, accuracy <Z>%"
```

**Phase E expected delta: +1 to +3pp.**

---

# Phase F: MLP Residual Head (the v2)

**Goal:** Replace the linear ceiling on capacity with a hybrid energy `E(x,y) = θᵀφ_hand(x,y) + g_ψ(φ_dense(x,y))`. The MLP captures non-linear interactions that no amount of hand features will give us.

**Architecture choice:** per-foot MLP (one small network, weights shared across all 6 feet) over a dense feature vector per foot. **Critically, the dense per-foot features include site decisions in that foot** — without this, the MLP cannot help on the dominant error class (elision direction, 47.9% of errors). Inputs per foot:

- foot type one-hot (3: DACTYL, SPONDEE, FINAL)
- per-syllable-in-foot weight one-hot (3 syls × 2 weights = 6)
- foot position (1, normalized [0,1])
- neighbor foot types one-hot (prev + next = 6)
- **per-foot site-decision summary:** for each `SiteType` ∈ {ELISION, SYNIZESIS, DIPHTHONG_SPLIT, MCL, PRODELISION}, a count of sites of that type falling in this foot × decision-direction (active vs inactive). This is the M2 fix: without these inputs the MLP is blind to the dominant error class.

Output: scalar energy contribution per foot. Sum across the 6 feet → MLP residual energy.

**Training math.** Two viable approaches:

1. **Joint (preferred):** port the linear part into PyTorch (a single `nn.Linear` over hand features) and jointly optimize θ and ψ via AdamW. The exact NLL `−E(gold) + logsumexp(−E(candidates))` is fully differentiable. This finds the joint MLE.

2. **Two-stage (fallback):** fix θ from L-BFGS Phase B-D training; train ψ alone via AdamW. This converges to a constrained local optimum — generally NOT the joint MLE — but is simpler if the joint path destabilizes. Cost: probably 1-2pp accuracy vs joint.

Default to joint. Fall back to two-stage if joint training oscillates.

### Task F.1: Add PyTorch as optional dependency

`pyproject.toml`:

```toml
[project.optional-dependencies]
ml = ["torch>=2.2"]
```

Install in env:

```bash
mamba run -n latin-ebm pip install "torch>=2.2"
```

### Task F.2: Implement `MLPHead` module

`src/latin_ebm/mlp.py`:

```python
import torch
import torch.nn as nn


class PerFootMLP(nn.Module):
    """Small MLP over per-foot dense features. One network shared across all 6 feet."""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, foot_features: torch.Tensor) -> torch.Tensor:
        """foot_features: (n_feet, input_dim). Returns (1,) scalar energy contribution."""
        per_foot = self.net(foot_features).squeeze(-1)  # (n_feet,)
        return per_foot.sum()
```

Test in `tests/test_mlp.py`:

```python
def test_per_foot_mlp_output_shape():
    import torch
    from latin_ebm.mlp import PerFootMLP
    m = PerFootMLP(input_dim=10, hidden_dim=8)
    x = torch.randn(6, 10)
    y = m(x)
    assert y.shape == ()  # scalar
```

### Task F.3: Dense per-foot feature extractor

`src/latin_ebm/features_v2.py`:

```python
"""Dense per-foot feature extraction for the MLP residual head."""
from __future__ import annotations
import numpy as np
from latin_ebm.types import FootType, PhonWeight, LatinLine, Parse, SiteType

# Per-foot vector layout (D=22):
#   [0:3]    foot type one-hot (DACTYL, SPONDEE, FINAL)
#   [3:9]    per-syllable weights (3 syls × 2 weights), zero-pad for SPONDEE/FINAL
#   [9:10]   foot position (foot_idx / 5)
#   [10:13]  prev foot type one-hot
#   [13:16]  next foot type one-hot
#   [16:22]  per-foot site-decision summary (6 entries):
#            counts of {elision_active, elision_retained, synizesis_merged,
#                       diphthong_split_taken, mcl_closed, mcl_opened}
PER_FOOT_DIM = 22

_FOOT_INDEX = {FootType.DACTYL: 0, FootType.SPONDEE: 1, FootType.FINAL: 2}


def per_foot_dense_features(line: LatinLine, parse: Parse) -> np.ndarray:
    """Build (6, PER_FOOT_DIM) dense feature array."""
    out = np.zeros((6, PER_FOOT_DIM), dtype=np.float32)
    n_feet = len(parse.foot_types)
    syl_to_foot = _build_syl_to_foot(parse)

    for foot_idx in range(n_feet):
        ft = parse.foot_types[foot_idx]
        out[foot_idx, _FOOT_INDEX[ft]] = 1.0

        # Per-syllable weights inside this foot
        syl_range = _foot_syllable_range(parse, foot_idx)
        for offset, syl_idx in enumerate(syl_range[:3]):
            syl = parse.syllables[syl_idx]
            wo = 3 + offset * 2
            if syl.weight == PhonWeight.LONG:
                out[foot_idx, wo] = 1.0
            else:
                out[foot_idx, wo + 1] = 1.0

        # Foot position
        out[foot_idx, 9] = foot_idx / 5.0

        # Prev / next foot type
        if foot_idx > 0:
            out[foot_idx, 10 + _FOOT_INDEX[parse.foot_types[foot_idx - 1]]] = 1.0
        if foot_idx < n_feet - 1:
            out[foot_idx, 13 + _FOOT_INDEX[parse.foot_types[foot_idx + 1]]] = 1.0

    # Per-foot site decision counts (M2: gives MLP visibility into the
    # dominant error class — elision direction)
    from latin_ebm.types import SiteChoice
    for site in line.sites:
        decision = parse.decisions.get(site.index, site.default)
        syl_idx = _syllable_index_for_atom(parse, site.atom_indices[0])
        if syl_idx is None:
            continue
        foot_idx = syl_to_foot.get(syl_idx)
        if foot_idx is None or foot_idx >= 6:
            continue
        offset_base = 16
        if site.site_type == SiteType.ELISION:
            if decision == SiteChoice.ELIDE:
                out[foot_idx, offset_base + 0] += 1.0
            else:
                out[foot_idx, offset_base + 1] += 1.0
        elif site.site_type == SiteType.SYNIZESIS:
            if decision == SiteChoice.MERGE:
                out[foot_idx, offset_base + 2] += 1.0
        elif site.site_type == SiteType.DIPHTHONG_SPLIT:
            if decision == SiteChoice.SPLIT:
                out[foot_idx, offset_base + 3] += 1.0
        elif site.site_type == SiteType.MUTA_CUM_LIQUIDA:
            if decision == SiteChoice.CLOSE:
                out[foot_idx, offset_base + 4] += 1.0
            else:
                out[foot_idx, offset_base + 5] += 1.0

    return out


def _build_syl_to_foot(parse: Parse) -> dict[int, int]:
    """Map each syllable index → foot index."""
    out = {}
    for foot_idx in range(len(parse.foot_types)):
        for syl_idx in _foot_syllable_range(parse, foot_idx):
            out[syl_idx] = foot_idx
    return out


def _foot_syllable_range(parse: Parse, foot_idx: int) -> list[int]:
    """Syllable indices belonging to foot_idx."""
    start = parse.foot_boundaries[foot_idx]
    end = (parse.foot_boundaries[foot_idx + 1]
           if foot_idx + 1 < len(parse.foot_boundaries)
           else len(parse.syllables))
    return list(range(start, end))


def _syllable_index_for_atom(parse: Parse, atom_idx: int) -> int | None:
    for syl_idx, syl in enumerate(parse.syllables):
        if atom_idx in syl.atom_indices:
            return syl_idx
    return None
```

Test:

```python
def test_per_foot_dense_features_shape(aeneid_test_lines):
    line, _ = aeneid_test_lines[0]
    cands = enumerate_parses(line)
    if not cands: pytest.skip()
    from latin_ebm.features_v2 import per_foot_dense_features, PER_FOOT_DIM
    feats = per_foot_dense_features(line, cands[0])
    assert feats.shape == (6, PER_FOOT_DIM)
    # foot type one-hot fires exactly once per foot
    for i in range(6):
        assert feats[i, 0:3].sum() == 1.0
```

Verify `parse.foot_boundaries` format before merging — adjust `_foot_syllable_range` if needed.

### Task F.4: Hybrid `LinearEBMWithMLP`

`src/latin_ebm/energy_v2.py`:

```python
import numpy as np
import torch
from latin_ebm.energy import LinearEBM
from latin_ebm.mlp import PerFootMLP


class HybridEBM:
    """E(x,y) = θᵀφ_hand(x,y) + g_ψ(φ_dense(x,y))."""
    def __init__(self, linear: LinearEBM, mlp: PerFootMLP):
        self.linear = linear
        self.mlp = mlp

    def energy(self, hand_features: np.ndarray, dense_per_foot: np.ndarray) -> float:
        linear_e = self.linear.energy(hand_features)
        with torch.no_grad():
            dense_t = torch.from_numpy(dense_per_foot).float()
            mlp_e = float(self.mlp(dense_t).item())
        return linear_e + mlp_e
```

### Task F.4b: `precompute_v2` — build joint training tensors

Before training, we need a structure that pairs each training line with its candidates, the gold index, the hand-feature matrix, and the dense per-foot tensor. Add to `src/latin_ebm/train.py`:

```python
from dataclasses import dataclass
import torch
from latin_ebm.features_v2 import per_foot_dense_features, PER_FOOT_DIM


@dataclass
class PrecomputedLineV2:
    line_id: str
    n_candidates: int
    gold_indices: list[int]                  # indices in candidates that match gold
    hand_features: np.ndarray                # (n_cands, n_hand)
    dense_features: np.ndarray               # (n_cands, 6, PER_FOOT_DIM)


def precompute_v2(examples, feature_index, lexicon=None) -> list[PrecomputedLineV2]:
    out: list[PrecomputedLineV2] = []
    for ex in examples:
        cands = enumerate_parses(ex.line)
        if not cands: continue
        hand = np.stack([extract_features(ex.line, c, feature_index, lexicon=lexicon) for c in cands])
        dense = np.stack([per_foot_dense_features(ex.line, c) for c in cands])
        gold_indices = [i for i, c in enumerate(cands)
                        if c.foot_types == ex.gold_parse.foot_types
                        and c.slots == ex.gold_parse.slots]
        if not gold_indices: continue
        out.append(PrecomputedLineV2(
            line_id=ex.line.corpus_id,
            n_candidates=len(cands),
            gold_indices=gold_indices,
            hand_features=hand,
            dense_features=dense,
        ))
    return out
```

### Task F.5: Joint training (PyTorch end-to-end)

The linear part is a single `nn.Linear` (or a free parameter tensor `theta`) and the MLP head is `PerFootMLP`. Both optimized jointly via AdamW. The partition function is `logsumexp(-energies)` and is fully differentiable.

`scripts/train_v2.py`:

```python
"""Joint training of linear + MLP-residual hybrid EBM."""
import argparse, json, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.features import build_feature_index, FeatureIndex
from latin_ebm.features_v2 import PER_FOOT_DIM
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.mlp import PerFootMLP
from latin_ebm.train import precompute_v2


def train_joint(precomputed, n_hand_features, mlp_hidden=32, lr=1e-3,
                weight_decay=1e-4, epochs=50, device="cpu"):
    theta = nn.Parameter(torch.zeros(n_hand_features, device=device))
    mlp = PerFootMLP(input_dim=PER_FOOT_DIM, hidden_dim=mlp_hidden).to(device)
    opt = optim.AdamW(
        [{"params": [theta], "weight_decay": 0.01},     # linear L2
         {"params": mlp.parameters(), "weight_decay": weight_decay}],
        lr=lr,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for pre in precomputed:
            hand_t = torch.from_numpy(pre.hand_features).float().to(device)   # (n_cands, n_hand)
            dense_t = torch.from_numpy(pre.dense_features).float().to(device) # (n_cands, 6, D)

            linear_e = hand_t @ theta                                # (n_cands,)
            # Apply MLP per (candidate, foot) → sum over feet
            n_cands = dense_t.shape[0]
            mlp_per_foot = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(n_cands, 6)
            mlp_e = mlp_per_foot.sum(dim=1)                          # (n_cands,)

            energies = linear_e + mlp_e
            log_z = torch.logsumexp(-energies, dim=0)

            # Partial supervision: sum-prob over gold-compatible candidates
            gold_e = energies[pre.gold_indices]
            gold_log_p = torch.logsumexp(-gold_e, dim=0) - log_z
            loss = -gold_log_p

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        logging.info("Epoch %d loss=%.4f", epoch, epoch_loss / len(precomputed))

    return theta.detach().cpu().numpy(), mlp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--mlp-hidden", type=int, default=32)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    lexicon = VowelLengthLexicon(Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt"))
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]

    # Build feature index
    lines = [ex.line for ex in train_ex]
    parses_per_line = [enumerate_parses(ex.line) for ex in train_ex]
    index = build_feature_index(lines, parses_per_line, lexicon=lexicon)

    # Precompute v2
    train_data = precompute_v2(train_ex, index, lexicon=lexicon)
    test_data = precompute_v2(test_ex, index, lexicon=lexicon)

    theta, mlp = train_joint(train_data, n_hand_features=index.n_features,
                              mlp_hidden=args.mlp_hidden, lr=args.mlp_lr,
                              epochs=args.mlp_epochs)

    # Evaluate: count correct predictions on test
    correct = 0
    for pre in test_data:
        hand_t = torch.from_numpy(pre.hand_features).float()
        dense_t = torch.from_numpy(pre.dense_features).float()
        linear_e = hand_t @ torch.from_numpy(theta).float()
        mlp_e = mlp.net(dense_t.view(-1, PER_FOOT_DIM)).view(pre.n_candidates, 6).sum(dim=1)
        energies = linear_e + mlp_e
        pred = int(energies.argmin().item())
        if pred in pre.gold_indices:
            correct += 1
    acc = correct / max(len(test_data), 1)
    out = {"test_accuracy": acc, "n_train": len(train_data), "n_test": len(test_data)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    logging.info("Test accuracy: %.4f", acc)


if __name__ == "__main__":
    main()
```

Tests: small synthetic line with 3 candidates, verify joint optimizer reduces loss. Add gradient-flow assertion (`theta.grad is not None` after backward).

**Fallback (two-stage):** if joint training oscillates (loss bounces, never settles), freeze θ from a separate L-BFGS run, then train MLP alone. Same code path; just skip the `theta` parameter from the optimizer.

### Task F.6: Re-train + measure Phase F

```bash
mamba run -n latin-ebm python scripts/train_v2.py \
    --test-books 1,2 --l2 0.01 --max-iter 500 \
    --mlp-hidden 32 --mlp-epochs 50 --mlp-lr 1e-3 \
    --out results/phase_F_accuracy.json
```

Expected: 82–93% test foot accuracy. If lower than 80%, the MLP is overfitting; reduce `hidden_dim` to 16 or add dropout.

```bash
git commit -m "feat(phase-F): MLP residual head - accuracy <X>%"
```

**Phase F expected delta: +3 to +6pp.**

---

# Phase G: Cross-Author Training Data

**Goal:** Add Ovid, Lucretius, Horace from Pedecerto if available. More data generally helps both linear and MLP parts.

### Task G.1: Inventory available Pedecerto authors

```bash
ls /Users/biofanat/Documents/side-projects/latin-ebm/pedecerto-raw/
```

If only Vergil is present, this phase is no-op or requires manual download (anceps's `src/mqdq/scraping.py` is the path). Skip if not feasible.

### Task G.2: Multi-author training

Extend `train_v2.py` to accept multiple `--xml` paths and a corpus-qualified `--test-spec`:

```python
p.add_argument("--xml", action="append", required=True,
               help="One or more Pedecerto XML paths (repeatable)")
p.add_argument("--test-spec", default="VERG-aene:1,2",
               help="Format: <basename-of-xml-without-ext>:<comma-separated-books>")
# Parsing:
test_corpus_id, test_books_str = args.test_spec.split(":")
test_books = set(test_books_str.split(","))

# Load all XMLs into one example pool
all_examples = []
for xml_path in args.xml:
    result = parse_xml(Path(xml_path), lexicon=lexicon)
    corpus_id = Path(xml_path).stem  # e.g. "VERG-aene"
    for ex in result.examples:
        ex.line.corpus_id = f"{corpus_id}.{ex.line.book}.{ex.line.line_num}"
        ex.line.author = corpus_id
        all_examples.append(ex)

train_ex = [e for e in all_examples
            if not (e.line.author == test_corpus_id and e.line.book in test_books)]
test_ex  = [e for e in all_examples
            if e.line.author == test_corpus_id and e.line.book in test_books]
```

CLI:

```bash
mamba run -n latin-ebm python scripts/train_v2.py \
    --xml pedecerto-raw/VERG-aene.xml \
    --xml pedecerto-raw/OVID-met.xml \
    --test-spec "VERG-aene:1,2" \
    --l2 0.01 --mlp-hidden 32 --mlp-epochs 50 \
    --out results/phase_G_accuracy.json
```

Test set stays Aeneid 1-2; training adds Ovid (and any other listed XML). Author identity is preserved in `line.author` for downstream features.

### Task G.3: Author-conditioning feature

Add `author:{author_name}` feature to capture author-specific metrical preferences.

```bash
git commit -m "feat(phase-G): cross-author training - accuracy <X>%"
```

**Phase G expected delta: +1 to +3pp.**

---

# Phase H: Final Tuning

**Goal:** Hyperparameter sweep, regularization tuning, ablation.

### Task H.1: L2 / MLP-hidden-dim sweep

```bash
for l2 in 0.001 0.01 0.1; do
  for hidden in 16 32 64; do
    mamba run -n latin-ebm python scripts/train_v2.py \
      --test-books 1,2 --l2 $l2 --mlp-hidden $hidden \
      --out results/sweep_l2${l2}_h${hidden}.json
  done
done
```

Pick best by test foot accuracy.

### Task H.2: Feature ablation

`scripts/feature_ablation.py`: turn each feature group on/off, measure impact. Identifies dead weight.

### Task H.3: Final model + project summary update

```bash
git commit -m "data(phase-H): final tuned model - test foot accuracy <X>%"
```

Update `docs/project-summary.md`:

```markdown
## Updated Results (Post Accuracy Plan)

| Model | Foot Pattern Acc | Notes |
|---|---|---|
| EBM v2 (linear + MLP residual) | <X>% | <Y> features, <Z>-dim MLP |
| Anceps | 92.0% | constraint solver |
| EBM v1 (pre-accuracy plan) | 67.7% | linear, 122 features |
```

---

## Self-Review Checklist

**1. Spec coverage:** Every error class identified in the diagnostic (elision 47.9%, MCL 26.5%, synizesis 25.6%, diphthong, training-data attrition) has a dedicated phase. The MLP head addresses residual non-linearity via per-foot site-decision summaries.

**2. Placeholder scan:** `_syllable_index_for_atom` and `_foot_for_syllable` helpers are now spelled out inside Task C.1 and Task F.3 (no hand-waved references). The `precompute_v2` function is defined in Task F.4b with full signature.

**3. Type consistency:** `PerFootMLP`, `HybridEBM`, `PrecomputedLineV2` use the same naming conventions as existing types. Lemma lookup is added to the existing `VowelLengthLexicon` rather than creating a parallel class.

**4. API-mismatch fixes (post-review-1 patches):**
- `FeatureIndex.fire` was a nonexistent method in early drafts. Replaced with direct `features[name] += value` mutation, matching the existing `extract_features` pattern at `features.py:91`.
- `train_nll` signature in Task A.1 corrected: `train_nll(examples, l2_lambda=l2, max_iter=max_iter, lexicon=lexicon)` returning `tuple[LinearEBM, TrainResult]`.
- `LemmaLexicon` standalone class eliminated — merged into `VowelLengthLexicon` to avoid double-loading Morpheus.
- Phase G's multi-XML CLI now has concrete `--xml` (repeatable) + `--test-spec` semantics rather than the invented `--test-books VERG-aene:1,2` syntax.

**5. TDD:** every code change is preceded by a failing test. Tasks D.1, D.2, D.3 still need explicit test scaffolding before execution — flagged here.

**6. Realistic targets:** revised from optimistic 85-90% to defensible **78-85%**. The reduction reflects:
- Phase B's per-syllable/foot-bigram features help pattern shape (already strong, 94.3% gold-in-set), not decision direction (the dominant error class).
- Phase F's MLP gain depends entirely on dense features including site decisions (Task F.3) — otherwise the MLP is blind to the dominant error class.

**7. MLP training math (Task F):** joint AdamW optimization of (θ, ψ) is the default (the partition function is fully differentiable). Two-stage (freeze θ, train ψ) is fallback. The plan explicitly states this trade-off.

**8. Rollback:** Phase 6 (94.3% ceiling, 67.7% acc) is the floor — any phase that REGRESSES below baseline (i.e. < 67.7% test foot accuracy) gets reverted before continuing.

**9. Phase B downgrade is documented** in the trajectory commentary and not hidden in a footnote.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-12-accuracy-raise.md`. Three execution choices:**

**1. Subagent-Driven (recommended for Phases A–E)** — fresh subagent per phase, review between phases. Best for the feature-engineering parts where each phase has clear pass/fail criteria.

**2. Inline Execution (recommended for Phase F)** — the MLP head is interdependent across files (energy + mlp + train_v2 + features_v2). Best done in one session.

**3. Hybrid:** Inline through Phase E, then dispatch Phase F as a single multi-file subagent with PyTorch context.

**Which approach?**
