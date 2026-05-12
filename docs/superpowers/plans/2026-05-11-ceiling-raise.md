# Latin Scansion EBM: Ceiling Raise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the EBM's gold-in-candidate-set ceiling from ~75% to ≥95% on the Aeneid test books, by fixing atomization edge cases, wiring Morpheus natural-length data into the realize/enumerate pipeline, and achieving phonological parity with anceps — **without** absorbing anceps's constraint solver (the EBM must still *learn* metrical topology).

**Architecture:** The codebase already has a working linear EBM (`E_θ(x,y) = θᵀφ(x,y)`) with exact enumeration over hexameter candidates. The ceiling problem is upstream of the energy function: ~12% of test lines produce zero candidates and ~13% have candidates but the gold weight pattern is unreachable. Both are caused by (a) `VocalicAtom.natural_length` being always `None` (the Morpheus alignment adapter exists at `lexicon.py:218` but is never called) and (b) atomization heuristics that mishandle Greek names, intervocalic u/v, and complex vowel chains.

**Tech Stack:** Python 3.11+, pytest, polars (for results tracking), existing `latin_ebm` package, MQDQ + Morpheus dictionaries (already on disk at `data/MqDqMacrons.json` and `data/MorpheusMacrons.txt`).

**Out of scope for this plan:** the v2 MLP, richer features for the 19.5% reachable-but-wrong, cross-author transfer, pentameter/lyric meters. Those are separate plans that depend on the ceiling being raised first.

---

## Background

From the project summary and three exploration passes (atomize/realize/enumerate inventory, anceps phonology rules with file:line citations, and concrete failing Aeneid lines):

| Failure mode | Lines | Root cause | Phase |
|---|---|---|---|
| Gold-unreachable (open syllables can't be inferred long) | ~13% | `VocalicAtom.natural_length` always `None`; `lookup_aligned` never called | Phase 1 |
| Vowel-chain blowup (e.g., "Lauiniaque") | ~5% of no-candidates | Intervocalic consonantal-u heuristic disabled at `atomize.py:106-108` | Phase 2 |
| False enclitic splits on words like "atque" | ~1% | Hardcoded `_NO_STRIP` exception list is incomplete; anceps uses dict-first instead | Phase 3 |
| Wrong syllable closure (qv/sv/gv, MCL, x/z, cross-word) | ~3-5% | EBM's realize.py doesn't mirror anceps's hard-coded clustering | Phase 4 |
| Greek/proper names (Ganymedis, Tyrii) | ~2-3% | No softening for words outside the dictionary | Phase 5 |

Concrete failing test lines (from corpus inspection):
- **Aen. 1.2** `italiam fato profugus lauiniaque uenit` — gold unreachable; "italiam" needs natural-long ī
- **Aen. 1.5** `multa quoque et bello passus dum conderet urbem` — no candidates; vowel chain
- **Aen. 1.28** `et genus inuisum et rapti ganymedis honores` — Greek name
- **Aen. 1.44** `illum exspirantem transfixo pectore flammas` — consonant cluster ambiguity
- **"non Xanthus"** — cross-word long-by-position
- **"patris"** — muta cum liquida (anceps gives `pa*tri*s`; first syllable short)

---

## File Structure

### To create

| Path | Responsibility |
|---|---|
| `scripts/diagnose_ceiling.py` | Per-line failure classifier; emits TSV with `{line_id, status, reason, n_candidates, gold_in_set}` |
| `scripts/summarize_ceiling.py` | Aggregates TSV → summary stats; supports `--compare` against prior baseline |
| `tests/test_diagnostics.py` | Unit tests for the diagnostic classifier |
| `tests/test_lexicon_alignment.py` | Tests for Morpheus alignment edge cases (qu, j/v, diphthong matching) |
| `tests/test_atomize_lexicon.py` | Integration tests verifying `natural_length` propagates from lexicon through atomizer |
| `tests/test_anceps_parity.py` | Tests that pin anceps-equivalent phonology behaviors |
| `tests/test_greek_names.py` | Tests for the `phonologically_uncertain` escape valve |
| `results/baseline_ceiling.tsv` | Phase-0 measurement; committed for diff visibility |
| `results/phase{1..5}_ceiling.tsv` | Per-phase measurements |
| `docs/ceiling-progression.md` | Running log of per-phase deltas |

### To modify

| Path | Why |
|---|---|
| `src/latin_ebm/atomize.py` | Pass lexicon through; populate `natural_length`; replace disabled u/v heuristic; dictionary-first enclitic |
| `src/latin_ebm/lexicon.py` | Fix `_normalize_key` (don't collapse v→u for consonantal-v words); add `is_consonantal_u`, `is_greek_or_proper`, `is_known_form` helpers |
| `src/latin_ebm/realize.py` | Audit MCL default; audit qv/sv/gv; add cross-word long-by-position; handle `phonologically_uncertain` atoms |
| `src/latin_ebm/types.py` | Add `phonologically_uncertain: bool = False` field to `VocalicAtom` |
| `src/latin_ebm/enumerate.py` | Verify weight-compatibility check uses `natural_length` properly |

---

## Conventions

- **Package name is `latin_ebm`** (not `latin_scan_ebm`).
- All tests use pytest. Fixtures live in `tests/conftest.py`.
- Tests assert on real Aeneid lines parsed from `pedecerto-raw/VERG-aene.xml` via the existing `corpus/pedecerto.py` parser; use the `aeneid_test_lines` fixture (you will create it in Phase 0 if not present).
- Each task is one logical change with one test. Each `git commit` covers one task.
- Use TDD strictly: write failing test → run → see the failure → write minimal impl → run → see pass → commit.

**Codebase facts you must know before starting** (verified by direct inspection):

- `parse_xml(path)` returns a `ParseResult` (defined `corpus/pedecerto.py:375`) whose `.examples` field is `list[TrainingExample]`. Each `TrainingExample` has `.line: LatinLine` and `.gold_parse: Parse`.
- `LatinLine` (`types.py:149`) is **NOT frozen** — direct attribute assignment works (`line.author = "Vergil"`). `parse_line_element` already does this at `pedecerto.py:419-423`.
- `VocalicAtom` (`types.py:101`), `ConsonantBridge`, `AmbiguitySite`, `RealizedSyllable`, `Parse` are frozen — use `dataclasses.replace` to copy with changes.
- `corpus_id` produced by the parser is `f"{author}_{title}_{book}_{name}"`. For Aeneid 1.2 the actual string depends on what's in the XML's `<author>`/`<title>` tags. Throughout this plan, the fixture **overrides** `corpus_id` to the synthetic format `f"VERG-aene.{book}.{line_num}"` so test lookups are stable.
- Weight computation is **inlined in `realize()` at `realize.py:327-368`** — there is no `_compute_weight` function. The existing code already consults `natural_length` (line 340-343), but that branch was previously untested because `natural_length` was always `None`. After Phase 1 wires the lexicon, that branch fires.

---

# Phase 0: Diagnostic Harness

**Goal:** Build a classifier that tags every test-book line with its failure mode before changing any logic. This is the measurement instrument for every subsequent phase.

**Why first:** Every later phase has an "expected ceiling delta." Without a precise per-line classifier, we can't tell whether a phase did what it claimed.

### Task 0.1: Add an Aeneid test-books fixture to conftest.py

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_diagnostics.py` (new file):

```python
from pathlib import Path
import pytest

def test_aeneid_test_lines_fixture_loads(aeneid_test_lines):
    """The fixture should yield at least 1000 lines from books 1-2."""
    lines = list(aeneid_test_lines)
    assert len(lines) >= 1000
    assert all(line.book in {"1", "2"} for line in lines)
    assert all(line.atoms for line in lines)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd latin-scan-ebm && pytest tests/test_diagnostics.py::test_aeneid_test_lines_fixture_loads -v
```
Expected: FAIL with "fixture 'aeneid_test_lines' not found"

- [ ] **Step 3: Implement the fixture**

Add to `tests/conftest.py`:

```python
from pathlib import Path
from latin_ebm.corpus.pedecerto import parse_xml


@pytest.fixture(scope="session")
def aeneid_xml_path() -> Path:
    return Path(__file__).parent.parent / "pedecerto-raw" / "VERG-aene.xml"


@pytest.fixture(scope="session")
def aeneid_test_lines(aeneid_xml_path):
    """Aeneid books 1-2: list of (LatinLine, Parse) pairs. Skipped lines excluded.

    Overrides `corpus_id` to a synthetic format `VERG-aene.{book}.{line_num}` so
    test lookups by ID are stable regardless of what the XML's <author>/<title>
    tags contain.
    """
    result = parse_xml(aeneid_xml_path, lexicon=None)
    out = []
    for ex in result.examples:
        if ex.line.book not in {"1", "2"}:
            continue
        ex.line.corpus_id = f"VERG-aene.{ex.line.book}.{ex.line.line_num}"
        out.append((ex.line, ex.gold_parse))
    return out
```

`LatinLine` is NOT a frozen dataclass — direct attribute assignment (the override of `corpus_id`) works. `parse_xml` itself already calls `atomize` and `align_gold_parse` internally; you do not need to call them separately.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_diagnostics.py::test_aeneid_test_lines_fixture_loads -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_diagnostics.py
git commit -m "test: aeneid_test_lines session fixture for books 1-2"
```

---

### Task 0.2: Define `classify_line` failure-mode enum and stub

**Files:**
- Create: `scripts/diagnose_ceiling.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts.diagnose_ceiling import classify_line, LineStatus

def test_classify_line_returns_status_enum(aeneid_test_lines):
    line, gold = aeneid_test_lines[0]
    result = classify_line(line, gold, model=None)
    assert isinstance(result.status, LineStatus)
    assert result.reason is not None or result.status == LineStatus.CORRECT
    assert result.n_candidates >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_diagnostics.py::test_classify_line_returns_status_enum -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts'` or `ImportError`.

- [ ] **Step 3: Implement the stub**

`scripts/diagnose_ceiling.py`:

```python
"""Per-line failure-mode classifier for ceiling diagnostics.

Tags each (line, gold) pair as one of:
  - CORRECT: model's argmin candidate matches gold
  - WRONG_PREDICTION: gold is in candidate set but not argmin
  - GOLD_UNREACHABLE: candidates exist, gold weight pattern absent
  - NO_CANDIDATES: zero candidates pass syllable-count and weight checks

Sub-reasons further classify NO_CANDIDATES and GOLD_UNREACHABLE.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from latin_ebm.types import LatinLine, Parse
from latin_ebm.enumerate import enumerate_parses


class LineStatus(str, Enum):
    CORRECT = "correct"
    WRONG_PREDICTION = "wrong_prediction"
    GOLD_UNREACHABLE = "gold_unreachable"
    NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True)
class DiagnosisRow:
    line_id: str
    status: LineStatus
    reason: Optional[str]
    n_candidates: int
    n_atoms: int
    n_sites: int


def classify_line(line: LatinLine, gold: Parse, model=None) -> DiagnosisRow:
    candidates = enumerate_parses(line, meter=None)
    if not candidates:
        return DiagnosisRow(
            line_id=line.corpus_id,
            status=LineStatus.NO_CANDIDATES,
            reason=_no_candidates_reason(line),
            n_candidates=0,
            n_atoms=len(line.atoms),
            n_sites=len(line.sites),
        )
    gold_in_set = any(_foot_types_match(c, gold) for c in candidates)
    if not gold_in_set:
        return DiagnosisRow(
            line_id=line.corpus_id,
            status=LineStatus.GOLD_UNREACHABLE,
            reason=_gold_unreachable_reason(line, gold, candidates),
            n_candidates=len(candidates),
            n_atoms=len(line.atoms),
            n_sites=len(line.sites),
        )
    if model is None:
        return DiagnosisRow(
            line_id=line.corpus_id,
            status=LineStatus.CORRECT,  # treated as reachable-best if no model
            reason=None,
            n_candidates=len(candidates),
            n_atoms=len(line.atoms),
            n_sites=len(line.sites),
        )
    best = min(candidates, key=lambda c: model.score(line, c))
    if _foot_types_match(best, gold):
        return DiagnosisRow(line.corpus_id, LineStatus.CORRECT, None, len(candidates), len(line.atoms), len(line.sites))
    return DiagnosisRow(line.corpus_id, LineStatus.WRONG_PREDICTION, "argmin_mismatch", len(candidates), len(line.atoms), len(line.sites))


def _foot_types_match(candidate: Parse, gold: Parse) -> bool:
    return candidate.foot_types == gold.foot_types


def _no_candidates_reason(line: LatinLine) -> str:
    """Best-effort sub-reason for why no candidates exist."""
    # filled in by Task 0.3
    return "unknown"


def _gold_unreachable_reason(line, gold, candidates) -> str:
    """Best-effort sub-reason for why gold weight pattern is missing."""
    # filled in by Task 0.4
    return "unknown"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_diagnostics.py::test_classify_line_returns_status_enum -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_ceiling.py tests/test_diagnostics.py
git commit -m "feat: diagnose_ceiling skeleton with LineStatus enum"
```

---

### Task 0.3: Implement `_no_candidates_reason` sub-classifier

**Files:**
- Modify: `scripts/diagnose_ceiling.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_no_candidates_reason_vowel_chain(aeneid_test_lines):
    """Aen 1.5 fails with a vowel-chain reason."""
    pairs = {l.corpus_id: (l, g) for l, g in aeneid_test_lines}
    line, gold = pairs["VERG-aene.1.5"]
    row = classify_line(line, gold)
    assert row.status == LineStatus.NO_CANDIDATES
    assert "vowel_chain" in row.reason

def test_no_candidates_reason_syllable_count(aeneid_test_lines):
    """A line where every bundle produces M < 12 or M > 17 gets syllable_count reason."""
    # Find any no-candidate line and assert reason is one of {vowel_chain, syllable_count, weight_filter}
    for line, gold in aeneid_test_lines:
        row = classify_line(line, gold)
        if row.status == LineStatus.NO_CANDIDATES:
            assert row.reason in {"vowel_chain", "syllable_count_oob", "weight_filter"}
            return
    pytest.skip("no no_candidates line found")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_diagnostics.py::test_no_candidates_reason_vowel_chain -v
```
Expected: FAIL with assertion on `"vowel_chain" not in "unknown"`.

- [ ] **Step 3: Implement the sub-classifier**

Replace `_no_candidates_reason` in `scripts/diagnose_ceiling.py`:

```python
from latin_ebm.realize import syllable_count

def _no_candidates_reason(line: LatinLine) -> str:
    # Heuristic 1: vowel chain — any word has ≥4 consecutive vocalic atoms
    by_word: dict[int, int] = {}
    for atom in line.atoms:
        by_word[atom.word_idx] = by_word.get(atom.word_idx, 0) + 1
    if max(by_word.values(), default=0) >= 4:
        return "vowel_chain"

    # Heuristic 2: even with all defaults, syllable count out of [12,17]
    default_decisions = {s.index: s.default for s in line.sites}
    m = syllable_count(line, default_decisions)
    if m < 12 or m > 17:
        return "syllable_count_oob"

    # Otherwise, weight filter rejected all
    return "weight_filter"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_diagnostics.py::test_no_candidates_reason_vowel_chain tests/test_diagnostics.py::test_no_candidates_reason_syllable_count -v
```
Expected: PASS (or pytest.skip on the second if no such line exists in books 1-2).

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_ceiling.py tests/test_diagnostics.py
git commit -m "feat: no_candidates sub-reason classifier (vowel_chain, syllable_count_oob, weight_filter)"
```

---

### Task 0.4: Implement `_gold_unreachable_reason` sub-classifier

**Files:**
- Modify: `scripts/diagnose_ceiling.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gold_unreachable_reason_open_syllable(aeneid_test_lines):
    """Aen 1.2 'italiam fato...' fails because open-syllable natural length is missing."""
    pairs = {l.corpus_id: (l, g) for l, g in aeneid_test_lines}
    line, gold = pairs["VERG-aene.1.2"]
    row = classify_line(line, gold)
    assert row.status == LineStatus.GOLD_UNREACHABLE
    assert row.reason in {"open_syllable_length", "diphthong_mismatch", "elision_mismatch"}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL with reason still being `"unknown"`.

- [ ] **Step 3: Implement the sub-classifier**

```python
from latin_ebm.types import PhonWeight, MetricalSlot

def _gold_unreachable_reason(line, gold, candidates) -> str:
    """Pick the closest candidate; identify the first syllable where it disagrees with gold."""
    if not candidates:
        return "no_candidates"
    # Closest candidate = same foot count, max-shared-prefix of slot pattern
    def shared_prefix(c):
        n = 0
        for cs, gs in zip(c.slots, gold.slots):
            if cs != gs:
                break
            n += 1
        return n
    closest = max(candidates, key=shared_prefix)
    idx = shared_prefix(closest)
    if idx >= len(gold.slots):
        return "decisions_mismatch"
    cand_syl = closest.syllables[idx] if idx < len(closest.syllables) else None
    gold_syl = gold.syllables[idx] if idx < len(gold.syllables) else None
    if cand_syl is None or gold_syl is None:
        return "syllable_count_mismatch"
    if cand_syl.is_open and gold_syl.weight == PhonWeight.LONG and cand_syl.weight == PhonWeight.SHORT:
        return "open_syllable_length"
    # Diphthong mismatch: candidate's syllable has a diphthong-flagged atom but gold's syllable doesn't,
    # or vice versa. Check the actual atoms in each.
    cand_has_diph = any(line.atoms[i].in_diphthong for i in cand_syl.atom_indices)
    gold_has_diph = any(line.atoms[i].in_diphthong for i in gold_syl.atom_indices)
    if cand_has_diph != gold_has_diph:
        return "diphthong_mismatch"
    # Elision mismatch: the count of active atoms differs between candidate and gold
    if len(closest.syllables) != len(gold.syllables):
        return "elision_mismatch"
    return "other"
```

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_ceiling.py tests/test_diagnostics.py
git commit -m "feat: gold_unreachable sub-reason classifier"
```

---

### Task 0.5: Add CLI entry point and TSV output

**Files:**
- Modify: `scripts/diagnose_ceiling.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import tempfile
from pathlib import Path

def test_cli_writes_tsv():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.tsv"
        result = subprocess.run(
            ["python", "scripts/diagnose_ceiling.py",
             "pedecerto-raw/VERG-aene.xml",
             "--books", "1,2",
             "--out", str(out)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        rows = out.read_text().splitlines()
        header = rows[0].split("\t")
        assert header == ["line_id", "status", "reason", "n_candidates", "n_atoms", "n_sites"]
        assert len(rows) > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — no CLI yet.

- [ ] **Step 3: Implement the CLI**

Append to `scripts/diagnose_ceiling.py`:

```python
def main():
    import argparse, csv
    from pathlib import Path
    from latin_ebm.corpus.pedecerto import parse_xml
    from latin_ebm.lexicon import VowelLengthLexicon

    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("--books", default="1,2")
    p.add_argument("--out", required=True)
    p.add_argument("--no-lexicon", action="store_true",
                   help="run without lexicon (baseline pre-Phase-1 mode)")
    args = p.parse_args()
    books = set(args.books.split(","))

    lexicon = None
    if not args.no_lexicon:
        data = Path("data")
        lexicon = VowelLengthLexicon(
            mqdq_path=data / "MqDqMacrons.json",
            morpheus_path=data / "MorpheusMacrons.txt",
        )

    result = parse_xml(args.xml, lexicon=lexicon)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["line_id", "status", "reason", "n_candidates", "n_atoms", "n_sites"])
        for ex in result.examples:
            if ex.line.book not in books:
                continue
            ex.line.corpus_id = f"VERG-aene.{ex.line.book}.{ex.line.line_num}"
            row = classify_line(ex.line, ex.gold_parse, model=None)
            w.writerow([row.line_id, row.status.value, row.reason or "",
                        row.n_candidates, row.n_atoms, row.n_sites])


if __name__ == "__main__":
    main()
```

Baseline (Phase 0) measurement is captured with `--no-lexicon` to match the pre-fix world; later phases drop the flag.

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnose_ceiling.py tests/test_diagnostics.py
git commit -m "feat: diagnose_ceiling CLI with TSV output"
```

---

### Task 0.6: Implement `summarize_ceiling.py` aggregator

**Files:**
- Create: `scripts/summarize_ceiling.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_summarize_produces_counts(tmp_path):
    tsv = tmp_path / "in.tsv"
    tsv.write_text(
        "line_id\tstatus\treason\tn_candidates\tn_atoms\tn_sites\n"
        "a\tcorrect\t\t10\t14\t1\n"
        "b\tno_candidates\tvowel_chain\t0\t18\t3\n"
        "c\tgold_unreachable\topen_syllable_length\t5\t14\t2\n"
    )
    from scripts.summarize_ceiling import summarize
    summary = summarize(tsv)
    assert summary["total"] == 3
    assert summary["by_status"]["correct"] == 1
    assert summary["ceiling"] == pytest.approx(2/3)  # correct + reachable-but-wrong... actually just non-NO_CANDIDATES & gold-in-set
```

Refine the assertion to match your final semantics: **ceiling** = fraction where gold IS in the candidate set = `1 - (no_candidates + gold_unreachable) / total`.

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — script missing.

- [ ] **Step 3: Implement**

```python
"""Aggregate diagnose_ceiling TSV → summary stats."""
from __future__ import annotations
from pathlib import Path
import csv
from collections import Counter


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
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("tsv")
    p.add_argument("--compare", default=None)
    args = p.parse_args()
    a = summarize(Path(args.tsv))
    print(json.dumps(a, indent=2))
    if args.compare:
        b = summarize(Path(args.compare))
        print(f"\nDelta ceiling: {a['ceiling'] - b['ceiling']:+.3%}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/summarize_ceiling.py tests/test_diagnostics.py
git commit -m "feat: summarize_ceiling aggregator with --compare"
```

---

### Task 0.7: Capture baseline measurement

**Files:**
- Create: `results/baseline_ceiling.tsv`
- Create: `docs/ceiling-progression.md`

- [ ] **Step 1: Run the baseline**

```bash
python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml --books 1,2 --out results/baseline_ceiling.tsv
python scripts/summarize_ceiling.py results/baseline_ceiling.tsv > results/baseline_summary.json
```

- [ ] **Step 2: Write progression log**

`docs/ceiling-progression.md`:

```markdown
# Ceiling Progression Log

Per-phase ceiling = fraction of test-book lines (Aeneid 1-2) where the gold
foot pattern is in the EBM's candidate set.

| Phase | Date | Ceiling | Δ vs prev | no_candidates | gold_unreachable | Notes |
|---|---|---|---|---|---|---|
| baseline | 2026-05-11 | TBD | — | TBD | TBD | Pre-change state |
```

Fill `TBD` from `baseline_summary.json`.

- [ ] **Step 3: Commit**

```bash
git add results/baseline_ceiling.tsv results/baseline_summary.json docs/ceiling-progression.md
git commit -m "data: phase-0 baseline ceiling measurement"
```

**Phase 0 expected ceiling delta:** 0% (instrumentation only).

---

# Phase 1: Wire Lexicon Into Atomization

**Goal:** Make `VocalicAtom.natural_length` actually populate from Morpheus (and MQDQ, where layered) so the realize/enumerate pipeline can prune candidates that contradict known long/short vowels — and admit candidates whose gold pattern requires natural-long open syllables.

**Why this is Phase 1:** It directly attacks the ~13% gold-unreachable bucket and is a prerequisite for Phases 2–5, which all assume the lexicon is reachable from the atomizer.

### Task 1.1: Add test that `lookup_aligned` returns correct lengths for a known word

**Files:**
- Create: `tests/test_lexicon_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.types import PhonWeight


@pytest.fixture(scope="session")
def lexicon():
    data = Path(__file__).parent.parent / "data"
    return VowelLengthLexicon(
        mqdq_path=data / "MqDqMacrons.json",
        morpheus_path=data / "MorpheusMacrons.txt",
    )


def test_lookup_aligned_patris(lexicon):
    """patris (gen. of pater) — first 'a' short by nature, 'i' short."""
    result = lexicon.lookup_aligned("patris", atom_vowels=["a", "i"], author="Vergil")
    assert result[0] == PhonWeight.SHORT
    assert result[1] == PhonWeight.SHORT


def test_lookup_aligned_italiam(lexicon):
    """italiam — first 'i' naturally long (ī-talia)."""
    result = lexicon.lookup_aligned("italiam", atom_vowels=["i", "a", "i", "a"], author="Vergil")
    assert result[0] == PhonWeight.LONG
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_lexicon_alignment.py -v
```
Expected: One or both PASS if `lookup_aligned` works as designed; FAIL otherwise.

If they fail, document the actual return values in a code comment so the implementation in Task 1.3 has a target.

- [ ] **Step 3: If failures, write minimal scaffolding** (skip if pass)

If tests fail, the issue is most likely in `_normalize_key` (line 106) collapsing `v→u` too aggressively, or `_align_to_atoms` (line 268) mis-walking the form. Add print debugging temporarily; capture in a comment.

- [ ] **Step 4: Re-run to confirm green**

- [ ] **Step 5: Commit**

```bash
git add tests/test_lexicon_alignment.py
git commit -m "test: lookup_aligned baseline for patris and italiam"
```

---

### Task 1.2: Fix `_normalize_key` to preserve consonantal-v

**Files:**
- Modify: `src/latin_ebm/lexicon.py:106`
- Modify: `tests/test_lexicon_alignment.py`

**Diagnosis:** `_normalize_key` currently does `v→u` and `j→i` *everywhere*. This breaks Morpheus lookup for words like "novam" where Morpheus stores the form with `v` preserved (since the 'v' is a consonant), and the atomizer's word is `nouam` (which normalizes to `nouam`). The lookup key `novam → nouam` matches the atomizer's view; the problem is symmetric on Morpheus side. The correct fix is to canonicalize **both** sides to a single convention: keys are `[a-z]` only with `v→u, j→i`. This is already what it does, **but** the alignment adapter must remember which positions Morpheus marked as v/j (consonants) and skip them during atom-walk.

- [ ] **Step 1: Write the failing test**

```python
def test_normalize_key_collapses_v_to_u(lexicon):
    assert lexicon._normalize_key("novam") == "nouam"
    assert lexicon._normalize_key("jam") == "iam"


def test_lookup_aligned_novam_skips_consonantal_v(lexicon):
    """For 'nouam' (= novam): atomizer atoms are [o, a] (the 'u' is consonantal in Latin
    after we get Phase 2; here it's still vocalic but Morpheus has 'novam' with v).
    
    Morpheus form 'no_va*m' has vowels at positions: o (long), a (short).
    Atom vowels we'll pass: ["o", "a"].
    Expected: [LONG, SHORT].
    """
    result = lexicon.lookup_aligned("nouam", atom_vowels=["o", "a"], author="Vergil")
    assert result == [PhonWeight.LONG, PhonWeight.SHORT]
```

- [ ] **Step 2: Run test**

Expected: FAIL — the current `_align_to_atoms` treats Morpheus's 'v' as a vowel slot, producing length 3, then mismatches.

- [ ] **Step 3: Patch `_align_to_atoms`**

In `src/latin_ebm/lexicon.py`, update `_align_to_atoms` (around line 268):

```python
def _align_to_atoms(self, macro_form: str, atom_vowels: list[str]) -> list[PhonWeight | None]:
    """Walk macro_form. Yield one length per atom vowel. Skip consonantal v/j (they are
    NOT vowels in Latin even though Morpheus spells them with v/j)."""
    CONSONANTAL = {"v", "j"}
    VOWELS = set("aeiouy")
    DIPHTHONG_OPEN = "["
    result: list[PhonWeight | None] = []
    i = 0  # cursor into macro_form
    a = 0  # cursor into atom_vowels
    while i < len(macro_form) and a < len(atom_vowels):
        ch = macro_form[i]
        if ch in CONSONANTAL:
            i += 1
            continue
        if ch == DIPHTHONG_OPEN:
            # e.g. [ae] — try to consume one or two atoms (diphthong vs split)
            end = macro_form.index("]", i)
            diph = macro_form[i+1:end]
            i = end + 1
            # diphthong always long
            result.append(PhonWeight.LONG)
            a += 1
            # if atom_vowels has the second atom of the diphthong as separate, consume it too
            if a < len(atom_vowels) and atom_vowels[a] == diph[1]:
                result.append(PhonWeight.LONG)
                a += 1
            continue
        if ch in VOWELS:
            if ch != atom_vowels[a]:
                # misalignment — skip this morpheus vowel
                i += 1
                continue
            # peek for length marker
            mark = macro_form[i+1] if i+1 < len(macro_form) else ""
            length = {"_": PhonWeight.LONG, "^": PhonWeight.SHORT}.get(mark, None)
            result.append(length)
            i += 1 + (1 if mark in "_^*" else 0)
            a += 1
            continue
        i += 1
    while a < len(atom_vowels):
        result.append(None)
        a += 1
    return result
```

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/lexicon.py tests/test_lexicon_alignment.py
git commit -m "fix(lexicon): skip consonantal v/j during atom alignment"
```

---

### Task 1.3: Handle `qu` digraph in alignment

**Files:**
- Modify: `src/latin_ebm/lexicon.py:_align_to_atoms`
- Modify: `tests/test_lexicon_alignment.py`

**Diagnosis:** Morpheus stores "qui" with vowel positions for both `u` and `i`. The atomizer treats `qu` as a single consonant unit, so atom_vowels = `["i"]` for "qui". The alignment must skip Morpheus's `u` when it directly follows `q`.

- [ ] **Step 1: Write the failing test**

```python
def test_lookup_aligned_qui(lexicon):
    """qui — atomizer treats 'qu' as consonant; atom_vowels = ['i']."""
    result = lexicon.lookup_aligned("qui", atom_vowels=["i"], author="Vergil")
    assert result == [PhonWeight.LONG]


def test_lookup_aligned_quoque(lexicon):
    """quoque — 'qu' twice; atom_vowels = ['o', 'e']."""
    result = lexicon.lookup_aligned("quoque", atom_vowels=["o", "e"], author="Vergil")
    assert result[0] is not None
```

- [ ] **Step 2: Run tests**

Expected: FAIL — `_align_to_atoms` treats `u` after `q` as a vowel and mis-aligns.

- [ ] **Step 3: Patch `_align_to_atoms`**

Add a `qu`-skip rule:

```python
        if ch == "q" and i + 1 < len(macro_form) and macro_form[i+1] == "u":
            # 'qu' digraph — Morpheus spells the u, but atomizer treats it as consonant.
            # Skip both characters; do NOT consume an atom.
            i += 2
            # also skip any quantity marker on the u
            if i < len(macro_form) and macro_form[i] in "_^*":
                i += 1
            continue
```

Place this branch *before* the general `ch in VOWELS` branch.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/lexicon.py tests/test_lexicon_alignment.py
git commit -m "fix(lexicon): treat qu as a single consonant unit in alignment"
```

---

### Task 1.4: Add `lexicon` parameter to `atomize()` and propagate

**Files:**
- Modify: `src/latin_ebm/atomize.py`
- Modify: `tests/test_atomize_lexicon.py` (new)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from latin_ebm.atomize import atomize
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.types import PhonWeight


@pytest.fixture(scope="session")
def lexicon():
    data = Path(__file__).parent.parent / "data"
    return VowelLengthLexicon(
        mqdq_path=data / "MqDqMacrons.json",
        morpheus_path=data / "MorpheusMacrons.txt",
    )


def test_atomize_populates_natural_length_for_patris(lexicon):
    line = atomize("patris", lexicon=lexicon)
    a_atom = next(a for a in line.atoms if a.chars == "a")
    assert a_atom.natural_length == PhonWeight.SHORT


def test_atomize_populates_natural_length_for_italiam(lexicon):
    line = atomize("italiam", lexicon=lexicon)
    first_i = next(a for a in line.atoms if a.chars == "i")
    assert first_i.natural_length == PhonWeight.LONG
```

- [ ] **Step 2: Run tests**

Expected: FAIL — `natural_length` is `None`.

- [ ] **Step 3: Implement**

In `src/latin_ebm/atomize.py`, locate where `VocalicAtom` instances are constructed (around line 317 per inventory). Refactor to call `lexicon.lookup_aligned` per word once all atoms are built, then rebuild atoms with the new `natural_length` field:

```python
def _populate_natural_lengths(
    atoms: list[VocalicAtom],
    words: tuple[str, ...],
    lexicon,
    author: str = "",
) -> list[VocalicAtom]:
    if lexicon is None:
        return atoms
    by_word: dict[int, list[int]] = {}
    for i, a in enumerate(atoms):
        by_word.setdefault(a.word_idx, []).append(i)
    out = list(atoms)
    for wi, atom_indices in by_word.items():
        word = words[wi]
        atom_vowels = [atoms[i].chars for i in atom_indices]
        lengths = lexicon.lookup_aligned(word, atom_vowels, author=author)
        for atom_i, length in zip(atom_indices, lengths):
            if length is not None:
                out[atom_i] = dataclasses.replace(out[atom_i], natural_length=length)
    return out
```

Call this at the end of `atomize()`. Make sure `atomize`'s signature accepts `lexicon=None` and `author=""`.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/atomize.py tests/test_atomize_lexicon.py
git commit -m "feat(atomize): populate VocalicAtom.natural_length from lexicon"
```

---

### Task 1.5: Wire `lexicon` through the `aeneid_test_lines` fixture

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_atomize_lexicon.py`

- [ ] **Step 1: Write the failing test**

```python
def test_aeneid_lines_have_lengths_populated(aeneid_test_lines):
    """At least some atoms in books 1-2 should have natural_length set after wiring."""
    n_with_length = 0
    n_total = 0
    for line, _gold in aeneid_test_lines:
        for a in line.atoms:
            n_total += 1
            if a.natural_length is not None:
                n_with_length += 1
    frac = n_with_length / n_total
    assert frac > 0.5, f"only {frac:.1%} of atoms have lengths"
```

- [ ] **Step 2: Run test**

Expected: FAIL — fixture passes `lexicon=None`.

- [ ] **Step 3: Update fixture**

```python
@pytest.fixture(scope="session")
def lexicon():
    data = Path(__file__).parent.parent / "data"
    return VowelLengthLexicon(
        mqdq_path=data / "MqDqMacrons.json",
        morpheus_path=data / "MorpheusMacrons.txt",
    )


@pytest.fixture(scope="session")
def aeneid_test_lines(aeneid_xml_path, lexicon):
    raw_lines = parse_xml(aeneid_xml_path)
    out = []
    for raw in raw_lines:
        if raw.book not in {"1", "2"}:
            continue
        line = atomize(raw.text, lexicon=lexicon, author="Vergil")
        # ... (set metadata, align gold, append) — same as before
    return out
```

- [ ] **Step 4: Run test**

Expected: PASS (>50% of atoms have lengths).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_atomize_lexicon.py
git commit -m "test: aeneid fixture uses real lexicon"
```

---

### Task 1.6: Regression guard — `realize` consults `natural_length`

**Files:**
- Modify: `tests/test_realize.py`

**Diagnosis:** Inspection of `realize.py:327-368` (weight computation is **inlined**, not a separate function) confirms the existing code already consults `natural_length` at line 340: `has_long_vowel = any(line.atoms[i].natural_length == PhonWeight.LONG for i in group)`. Before Phase 1, `natural_length` was always `None` so this branch never fired. After Task 1.4, it does. **No code change is required** in `realize.py` for this case. We only need a regression-guard test to lock the behavior in.

The one gap in the existing code is that a known-`SHORT` natural length is NOT explicitly checked — but the `else` branch at line 359 already returns `SHORT`, so a known-SHORT open syllable correctly resolves to SHORT by fallthrough. There is also nothing to do for closed syllables (line 351 returns `LONG` regardless of natural length, which is correct: a positionally-long syllable IS long, even if its underlying vowel is naturally short).

- [ ] **Step 1: Write the regression-guard test**

In `tests/test_realize.py`:

```python
import dataclasses
from latin_ebm.types import PhonWeight
from latin_ebm.realize import realize


def test_open_syllable_with_natural_long_resolves_long(sample_line):
    """An open syllable whose nucleus atom has natural_length=LONG must realize as LONG."""
    new_atom = dataclasses.replace(sample_line.atoms[0], natural_length=PhonWeight.LONG)
    line = sample_line
    line.atoms = (new_atom,) + sample_line.atoms[1:]
    syllables = realize(line, decisions={})
    # find the syllable containing the first atom
    syl = next(s for s in syllables if 0 in s.atom_indices)
    assert syl.weight == PhonWeight.LONG


def test_open_syllable_with_natural_short_resolves_short(sample_line):
    """An open syllable whose nucleus atom has natural_length=SHORT must realize as SHORT
    (no coda, no diphthong, no other long signal)."""
    new_atom = dataclasses.replace(sample_line.atoms[0], natural_length=PhonWeight.SHORT)
    line = sample_line
    line.atoms = (new_atom,) + sample_line.atoms[1:]
    syllables = realize(line, decisions={})
    syl = next(s for s in syllables if 0 in s.atom_indices)
    # If this syllable happens to be closed in the fixture, skip — only assert on open case
    if syl.is_open:
        assert syl.weight == PhonWeight.SHORT
```

(Note: `LatinLine.atoms` is the tuple of frozen `VocalicAtom`s. `LatinLine` itself is mutable, so we replace its `atoms` field by direct assignment. The `sample_line` fixture from `tests/conftest.py:115` has 3 atoms total.)

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_realize.py::test_open_syllable_with_natural_long_resolves_long tests/test_realize.py::test_open_syllable_with_natural_short_resolves_short -v
```
Expected: PASS (existing code already correct; this is a regression guard).

- [ ] **Step 3: If tests FAIL**

Inspect `realize.py:327-368`. The `has_long_vowel` check at line 340 must come before the final `else: weight = SHORT` at line 359. If not, reorder. If the test fails for some other reason, debug the fixture first — the most common cause is that `sample_line.atoms[0]` is closed by a coda from `sample_bridges[0] = "rm"`, in which case it resolves LONG by position and the test passes anyway. The test above guards with `if syl.is_open`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_realize.py
git commit -m "test(realize): regression guard for natural_length-driven weight"
```

---

### Task 1.7: Verify `enumerate.py` weight-compatibility uses `natural_length`

**Files:**
- Modify: `src/latin_ebm/enumerate.py:25` (`_weight_compatible`)
- Modify: `tests/test_enumerate.py`

**Diagnosis:** `_weight_compatible` (per inventory, line 40-56) currently treats open syllables with unknown vowel length as compatible with any slot. With `natural_length` now populated, we want to *exclude* candidates whose realized weight contradicts the slot.

But — too-aggressive pruning broke the project before (see "Lexicon as hard constraint (failed)" in project-summary.md). The safer approach: trust `natural_length` only where it's set, and remain permissive where it's `None`. The realize-side weight already encodes this (Task 1.6); enumerate just needs to do hard-equal between `syllable.weight` and `slot`.

- [ ] **Step 1: Write the failing test**

In `tests/test_enumerate.py`:

```python
def test_italiam_fato_gold_in_candidates(aeneid_test_lines):
    pairs = {l.corpus_id: (l, g) for l, g in aeneid_test_lines}
    line, gold = pairs["VERG-aene.1.2"]
    from latin_ebm.enumerate import enumerate_parses
    candidates = enumerate_parses(line, meter=None)
    assert any(c.foot_types == gold.foot_types for c in candidates), \
        "Aen 1.2 gold foot pattern should be reachable after lexicon wiring"
```

- [ ] **Step 2: Run test**

Expected: PASS if Task 1.4 + 1.6 already opened the candidate set enough; FAIL otherwise.

If FAIL, dump `[c.foot_types for c in candidates]` and compare to `gold.foot_types` to identify the gap.

- [ ] **Step 3: If FAIL, tighten `_weight_compatible`**

The function should return `True` iff `syl.weight` and `slot` are not contradictory. Refresher: an unresolved-short open syllable should be admissible against `LONGUM` *only if* its atom's `natural_length` is not `SHORT`. Conversely, a known-short nucleus should NOT match `LONGUM`.

```python
def _weight_compatible(syl, slot, line) -> bool:
    if slot == MetricalSlot.ANCEPS:
        return True
    if slot == MetricalSlot.LONGUM:
        if syl.weight == PhonWeight.LONG:
            return True
        # admissible only if no atom in syllable is known-short
        return not any(line.atoms[i].natural_length == PhonWeight.SHORT for i in syl.atom_indices)
    if slot == MetricalSlot.BREVE:
        if syl.weight == PhonWeight.SHORT:
            return True
        return not any(line.atoms[i].natural_length == PhonWeight.LONG for i in syl.atom_indices) \
               and not syl.coda  # closed syllables can't be short
    return False
```

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/enumerate.py tests/test_enumerate.py
git commit -m "fix(enumerate): _weight_compatible consults natural_length"
```

---

### Task 1.8: Re-measure ceiling after Phase 1

- [ ] **Step 1: Run diagnostic**

```bash
python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml --books 1,2 --out results/phase1_ceiling.tsv
python scripts/summarize_ceiling.py results/phase1_ceiling.tsv --compare results/baseline_ceiling.tsv > results/phase1_summary.txt
```

- [ ] **Step 2: Update progression log**

Append to `docs/ceiling-progression.md`:

```markdown
| phase-1 | <date> | <%> | <delta> | <count> | <count> | Wired lexicon; aligned qu and consonantal v/j |
```

- [ ] **Step 3: Commit results**

```bash
git add results/phase1_ceiling.tsv results/phase1_summary.txt docs/ceiling-progression.md
git commit -m "data: phase-1 ceiling measurement"
```

**Phase 1 expected ceiling delta:** +5 to +10 pp (largest single phase; targets gold-unreachable open syllables).

---

# Phase 2: Dictionary-Driven Consonantal u/v

**Goal:** Re-enable intervocalic consonantal-u detection, gated on a lexicon lookup so that "suus"/"deus" stay vocalic and "uenit"/"lauinia" get the `u` reclassified as the consonant `v`.

### Task 2.1: Add `lexicon.is_consonantal_u` API and test

**Files:**
- Modify: `src/latin_ebm/lexicon.py`
- Modify: `tests/test_lexicon_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
def test_is_consonantal_u_for_uenit(lexicon):
    """uenit (= venit, 'comes') — initial u is consonantal."""
    assert lexicon.is_consonantal_u("uenit", position=0) is True


def test_is_consonantal_u_for_suus(lexicon):
    """suus — intervocalic u is vocalic (no v in Morpheus)."""
    assert lexicon.is_consonantal_u("suus", position=1) is False


def test_is_consonantal_u_for_lauinia(lexicon):
    """lauinia — intervocalic u is consonantal (Morpheus has 'lavinia')."""
    assert lexicon.is_consonantal_u("lauinia", position=2) is True
```

- [ ] **Step 2: Run tests**

Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Add a parallel raw-forms storage in `_load_morpheus`**

Currently `_load_morpheus` (lexicon.py:111-128) parses the macronized string into a list of per-vowel lengths and discards the original string. We need to keep both: the parsed lengths (for `lookup_aligned`) AND the original macronized form (for v/j position lookups).

In `__init__`, add a parallel dict:

```python
self._morpheus_raw: dict[str, set[str]] = defaultdict(set)
```

In `_load_morpheus`, after parsing each line, also store the raw form:

```python
def _load_morpheus(self, path):
    # ... existing parsing of "word\tmorph\tlemma\tmacro_form"
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        word, _morph, _lemma, macro_form = parts[0], parts[1], parts[2], parts[3]
        key = self._normalize_key(word)
        # Existing parsed-lengths storage
        self._morpheus[key].append(self._parse_macron_form(macro_form))
        # NEW: also store the raw macronized form (preserves v/j, [ae], [oe])
        self._morpheus_raw[key].add(macro_form.lower())
```

- [ ] **Step 4: Implement `is_consonantal_u`**

```python
def is_consonantal_u(self, word: str, position: int) -> bool:
    """Return True iff at least one Morpheus form for `word` has 'v' at `position`.

    `position` is the 0-indexed character offset in the *atomizer's* spelling
    (which uses u/i for both vowel and consonant). Morpheus stores v/j for
    consonantal versions, so we compare position-by-position after stripping
    macron marks and diphthong brackets.
    """
    import re
    key = self._normalize_key(word)
    raw_forms = self._morpheus_raw.get(key, set())
    if not raw_forms:
        return False
    for form in raw_forms:
        # Strip quantity marks (_ ^ *) and diphthong brackets
        clean = re.sub(r"[\^_\*]", "", form)
        clean = clean.replace("[", "").replace("]", "")
        if position < len(clean) and clean[position] == "v":
            return True
    return False
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_lexicon_alignment.py::test_is_consonantal_u_for_uenit tests/test_lexicon_alignment.py::test_is_consonantal_u_for_suus tests/test_lexicon_alignment.py::test_is_consonantal_u_for_lauinia -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/latin_ebm/lexicon.py tests/test_lexicon_alignment.py
git commit -m "feat(lexicon): is_consonantal_u backed by Morpheus v-marks"
```

---

### Task 2.2: Re-enable intervocalic consonantal-u in atomizer, gated by lexicon

**Files:**
- Modify: `src/latin_ebm/atomize.py:78-108`
- Modify: `tests/test_atomize_lexicon.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_uenit_atomizes_with_consonantal_u(lexicon):
    line = atomize("uenit", lexicon=lexicon)
    # 'u' at position 0 should not be a vocalic atom
    assert line.atoms[0].chars != "u"
    # there should be exactly 2 vocalic atoms (e, i)
    assert len(line.atoms) == 2
    assert [a.chars for a in line.atoms] == ["e", "i"]


def test_suus_atomizes_with_vocalic_u(lexicon):
    line = atomize("suus", lexicon=lexicon)
    assert [a.chars for a in line.atoms] == ["u", "u"]


def test_lauinia_atomizes_with_consonantal_u(lexicon):
    line = atomize("lauinia", lexicon=lexicon)
    # Expected vocalic atoms: a, i, i, a (the second 'u' is consonantal 'v')
    assert [a.chars for a in line.atoms] == ["a", "i", "i", "a"]
```

- [ ] **Step 2: Run tests**

Expected: FAIL (intervocalic u still disabled).

- [ ] **Step 3: Replace the disabled heuristic**

In `src/latin_ebm/atomize.py`, modify `_is_consonantal_u`:

```python
def _is_consonantal_u(word: str, pos: int, lexicon=None) -> bool:
    """Dictionary-first; deterministic-rule fallback."""
    # 1. After 'q' or 'g' before vowel — always consonantal (digraph-like)
    if pos > 0 and word[pos-1] in "qg" and pos + 1 < len(word) and word[pos+1] in "aeiouy":
        return True
    # 2. Word-initial before vowel — always consonantal
    if pos == 0 and pos + 1 < len(word) and word[pos+1] in "aeiouy":
        # but only if lexicon agrees (or we have no lexicon, fall back to True)
        if lexicon is None:
            return True
        return lexicon.is_consonantal_u(word, pos)
    # 3. Intervocalic before vowel — consult lexicon
    if (pos > 0 and word[pos-1] in "aeiouy"
        and pos + 1 < len(word) and word[pos+1] in "aeiouy"):
        if lexicon is None:
            return False  # safe default for "suus", "deus"
        return lexicon.is_consonantal_u(word, pos)
    return False
```

Update all callers of `_is_consonantal_u` to pass `lexicon`.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/atomize.py tests/test_atomize_lexicon.py
git commit -m "feat(atomize): dictionary-gated intervocalic consonantal-u"
```

---

### Task 2.3: Regression-guard the Aen. 1.5 vowel-chain failure

**Files:**
- Modify: `tests/test_atomize_lexicon.py`

- [ ] **Step 1: Write the test**

```python
def test_aen_1_5_produces_candidates(aeneid_test_lines):
    """multa quoque et bello passus dum conderet urbem — was no-candidates."""
    pairs = {l.corpus_id: (l, g) for l, g in aeneid_test_lines}
    line, gold = pairs["VERG-aene.1.5"]
    from latin_ebm.enumerate import enumerate_parses
    candidates = enumerate_parses(line, meter=None)
    assert len(candidates) > 0
```

- [ ] **Step 2: Run test**

If PASS, this phase already handled it. If FAIL, investigate which word's atomization still blows up (likely "quoque" or "passus").

- [ ] **Step 3: Triage if needed**

Run `scripts/diagnose_ceiling.py` for just this line in a debug script, dump atom/site counts.

- [ ] **Step 4: Commit**

```bash
git add tests/test_atomize_lexicon.py
git commit -m "test: regression guard Aen 1.5 vowel-chain"
```

---

### Task 2.4: Re-measure ceiling after Phase 2

- [ ] **Step 1: Run**

```bash
python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml --books 1,2 --out results/phase2_ceiling.tsv
python scripts/summarize_ceiling.py results/phase2_ceiling.tsv --compare results/phase1_ceiling.tsv > results/phase2_summary.txt
```

- [ ] **Step 2: Append to progression log**

- [ ] **Step 3: Commit**

```bash
git add results/phase2_ceiling.tsv results/phase2_summary.txt docs/ceiling-progression.md
git commit -m "data: phase-2 ceiling measurement"
```

**Phase 2 expected ceiling delta:** +3 to +5 pp.

---

# Phase 3: Dictionary-First Enclitic Check

**Goal:** Replace the `_NO_STRIP` hardcoded exception list (`atomize.py:124-135`) with a dictionary check: only strip `-que/-ne/-ve/-ue` if the *whole* form is unknown but the *stripped* form is known.

**Precondition check.** The existing `_NO_STRIP` set already covers `neque, atque, quoque, usque, undique, ubique, cumque, namque, denique, itaque, ...` (~25 words). Phase 3 is only worth doing if the **uncovered tail** (rare/late-Latin compounds, hapax forms in Pedecerto) accounts for a measurable slice of failures. **Before starting Phase 3 tasks, run:**

```bash
grep -E "(que|ne|ue|ve)$" results/phase2_ceiling.tsv | head -50
# Inspect: are any failures plausibly caused by false enclitic splits of words
# NOT in _NO_STRIP? If yes, proceed. If no failure pattern matches, skip Phase 3
# and jump to Phase 4.
```

If the precondition fails (no uncovered failures), commit a note to `docs/ceiling-progression.md`:

```markdown
| phase-3 | <date> | <same as phase-2> | 0.0% | (skipped — `_NO_STRIP` covers observed failures) |
```

and skip to Phase 4. Otherwise proceed.

### Task 3.1: Add `lexicon.is_known_form` API

**Files:**
- Modify: `src/latin_ebm/lexicon.py`
- Modify: `tests/test_lexicon_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
def test_is_known_form(lexicon):
    assert lexicon.is_known_form("atque") is True
    assert lexicon.is_known_form("neque") is True
    assert lexicon.is_known_form("quoque") is True
    assert lexicon.is_known_form("zorpumque") is False  # invented word
    assert lexicon.is_known_form("uirum") is True  # bare stem of "uirumque"
```

- [ ] **Step 2: Run tests**

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def is_known_form(self, word: str) -> bool:
    key = self._normalize_key(word)
    return key in self._morpheus or key in self._mqdq
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/lexicon.py tests/test_lexicon_alignment.py
git commit -m "feat(lexicon): is_known_form API"
```

---

### Task 3.2: Replace `_NO_STRIP` with dictionary-first check

**Files:**
- Modify: `src/latin_ebm/atomize.py:124-135`
- Modify: `tests/test_atomize_lexicon.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("word", ["atque", "neque", "quoque", "usque", "itaque"])
def test_dictionary_first_keeps_whole_word(word, lexicon):
    """Words that are themselves dictionary entries must not get enclitic-stripped."""
    line = atomize(word, lexicon=lexicon)
    # Single word = single word_idx
    assert max(a.word_idx for a in line.atoms) == 0


def test_unknown_compound_still_strips(lexicon):
    line = atomize("uirumque", lexicon=lexicon)
    # "uirumque" is unknown as whole; "uirum" is known → strip
    # After strip, should have two word_idx values (uirum + que)
    assert max(a.word_idx for a in line.atoms) >= 1
```

- [ ] **Step 2: Run tests**

Expected: FAIL (depending on `_NO_STRIP` coverage).

- [ ] **Step 3: Refactor `_strip_enclitic`**

Replace `_NO_STRIP` set with lexicon check. Signature:

```python
def _strip_enclitic(word: str, lexicon=None) -> tuple[str, str | None]:
    ENCLITICS = ("que", "ne", "ue", "ve")
    for enc in ENCLITICS:
        if len(word) > len(enc) + 1 and word.endswith(enc):
            stem = word[:-len(enc)]
            # Dictionary-first rule:
            #   - If whole word is known → don't strip
            #   - Else if stem is known → strip
            #   - Else don't strip (safer default for unknown words)
            if lexicon is None:
                # legacy behavior: strip unconditionally (matches pre-fix)
                return (stem, enc)
            if lexicon.is_known_form(word):
                return (word, None)
            if lexicon.is_known_form(stem):
                return (stem, enc)
            return (word, None)
    return (word, None)
```

Wire `lexicon` through to all callsites.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/atomize.py tests/test_atomize_lexicon.py
git commit -m "feat(atomize): dictionary-first enclitic stripping; deprecate _NO_STRIP"
```

---

### Task 3.3: Delete `_NO_STRIP` and verify

**Files:**
- Modify: `src/latin_ebm/atomize.py`
- Modify: `tests/test_atomize_lexicon.py`

- [ ] **Step 1: Write the failing test**

```python
def test_no_strip_list_removed():
    from latin_ebm import atomize as atomize_mod
    assert not hasattr(atomize_mod, "_NO_STRIP")
```

- [ ] **Step 2: Run test**

Expected: FAIL.

- [ ] **Step 3: Delete the list**

Remove `_NO_STRIP = {...}` from `src/latin_ebm/atomize.py` lines 124-135.

- [ ] **Step 4: Run all atomize/lexicon tests**

```bash
pytest tests/test_atomize.py tests/test_atomize_lexicon.py tests/test_lexicon_alignment.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/atomize.py tests/test_atomize_lexicon.py
git commit -m "refactor(atomize): remove _NO_STRIP hardcoded exception list"
```

---

### Task 3.4: Re-measure ceiling after Phase 3

- [ ] **Step 1: Run**

```bash
python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml --books 1,2 --out results/phase3_ceiling.tsv
python scripts/summarize_ceiling.py results/phase3_ceiling.tsv --compare results/phase2_ceiling.tsv > results/phase3_summary.txt
```

- [ ] **Step 2: Update progression log + commit**

```bash
git add results/phase3_ceiling.tsv results/phase3_summary.txt docs/ceiling-progression.md
git commit -m "data: phase-3 ceiling measurement"
```

**Phase 3 expected ceiling delta:** +0.5 to +1.5 pp.

---

# Phase 4: Anceps Phonology Parity

**Goal:** Mirror anceps's hard-coded clustering decisions so the EBM treats identical inputs identically: `qv/sv/gv` don't close syllables; muta cum liquida defaults to onset (syllable stays open); `x`/`z` are biconsonantal; long-by-position applies across word boundaries.

### Task 4.1: Verify `qv/sv/gv` don't close preceding syllable

**Files:**
- Modify: `src/latin_ebm/realize.py:_max_onset_split` (line ~40)
- Create: `tests/test_anceps_parity.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from latin_ebm.atomize import atomize
from latin_ebm.realize import realize
from latin_ebm.types import PhonWeight


def test_qv_cluster_does_not_close_previous_syllable(lexicon):
    """In 'antiqua', the 'i' before 'qu' should remain open (no long-by-position)."""
    line = atomize("antiqua", lexicon=lexicon)
    syllables = realize(line, decisions={})
    # find the syllable whose nucleus contains 'i' adjacent to 'qu' onset
    i_syl = next(s for s in syllables if "i" in s.nucleus)
    assert i_syl.is_open, "syllable before qu should be open"


def test_gv_cluster_does_not_close(lexicon):
    """In 'lingua' (l-i-ng-v-a per Allen & Greenough), 'i' before 'gv' is short."""
    line = atomize("lingua", lexicon=lexicon)
    syllables = realize(line, decisions={})
    # ... assert no extra closure
```

- [ ] **Step 2: Run tests**

Expected: depends on current behavior. Check.

- [ ] **Step 3: If FAIL, extend `_VALID_ONSET_PAIRS`**

The fix is one-line: `realize.py:28-37` constructs `_VALID_ONSET_PAIRS` from stop+liquid, s+stop, s+liquid combinations. `qu/su/gu` are NOT included, so `_max_onset_split("qu")` currently returns `("q", "u")` — splitting the digraph and putting `q` in the previous syllable's coda (wrong).

Verification: `qu` is tokenized as a digraph by `_tokenize_word` (`atomize.py:159`, "Handles digraphs (ch, ph, th, rh, qu)") and joined verbatim into `ConsonantBridge.chars` (`atomize.py:345`: `cons_str = "".join(cons_units)`). So bridge.chars contains literal `"qu"` as a 2-char substring — extending `_VALID_ONSET_PAIRS` is sufficient.

Patch `realize.py` after the existing `_VALID_ONSET_PAIRS` construction (around line 37):

```python
# u-as-v digraphs: qu/su/gu — the 'u' is consonantal, so the whole 2-char unit
# stays together as a single onset and does NOT close the preceding syllable.
# Mirrors anceps's SHORT_COMBINATIONS (utils.py:17).
_VALID_ONSET_PAIRS.update({"qu", "su", "gu"})
```

No change to `_max_onset_split` is needed — the existing maximal-onset loop now picks `qu` as a valid 2-char onset and returns `("", "qu")` for a bridge containing just `qu`.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/realize.py tests/test_anceps_parity.py
git commit -m "feat(realize): qv/sv/gv stay in onset (matches anceps)"
```

---

### Task 4.2: Pin MCL default behavior — short before stop+liquid

**Files:**
- Modify: `tests/test_anceps_parity.py`
- Possibly modify: `src/latin_ebm/realize.py` or `atomize.py:_detect_sites`

- [ ] **Step 1: Write the failing test**

```python
def test_patris_first_syllable_open_short_by_default(lexicon):
    """patris under default MCL choice (ONSET) → first syllable 'pa' is open and short."""
    line = atomize("patris", lexicon=lexicon)
    # The MCL site between 'a' and 'i' should default to ONSET (anceps's behavior).
    syllables = realize(line, decisions={})
    pa_syl = syllables[0]
    assert pa_syl.is_open, "first syllable of patris should be open under default MCL"
    assert pa_syl.weight == PhonWeight.SHORT, "first syllable of patris should be SHORT"
```

- [ ] **Step 2: Run test**

Expected: PASS if the existing default is ONSET, FAIL if it's CLOSE.

- [ ] **Step 3: If FAIL, update default in atomizer**

In `src/latin_ebm/atomize.py:_detect_sites`, ensure MCL sites are created with `default=SiteChoice.ONSET`:

```python
AmbiguitySite(
    index=..., site_type=SiteType.MUTA_CUM_LIQUIDA,
    atom_indices=(left_atom_idx, right_atom_idx),
    valid_choices=(SiteChoice.ONSET, SiteChoice.CLOSE),
    default=SiteChoice.ONSET,  # anceps default
)
```

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/atomize.py tests/test_anceps_parity.py
git commit -m "feat(atomize): MCL default = ONSET (anceps parity)"
```

---

### Task 4.3: Cross-word long-by-position

**Files:**
- Modify: `src/latin_ebm/realize.py`
- Modify: `tests/test_anceps_parity.py`

**Diagnosis:** Anceps appends the next word's consonantal prefix to the current word before applying long-by-position, then strips it. The EBM's `realize.py` should do the equivalent: when computing the coda of the line's *last* syllable of a word, the next word's onset can extend the coda for weight purposes.

- [ ] **Step 1: Write the failing test**

```python
def test_non_before_xanthus_is_long_by_position(lexicon):
    """'non Xanthus' — 'non' final-o is long because nX is a closing cluster."""
    line = atomize("non Xanthus", lexicon=lexicon)
    syllables = realize(line, decisions={})
    non_syl = next(s for s in syllables if "o" in s.nucleus)
    assert non_syl.weight == PhonWeight.LONG
```

- [ ] **Step 2: Run test**

Expected: FAIL — `realize.py` may not consider next-word onset.

- [ ] **Step 3: Implement cross-word coda extension**

In `src/latin_ebm/realize.py`, augment the weight-computation block (lines 327-368, where syllable weights are decided) to consult the next word's onset when the current syllable is the word-final syllable of its word.

The structure: a word-final syllable with no within-word coda might still be heavy if the next word's onset is a closing cluster (e.g., "non Xanthus" → 'o' is closed by `nX`). The existing code at line 351 (`elif syl.coda: weight = LONG`) only sees the within-word coda. Add a check before the diphthong branch:

```python
# After computing `syl.coda` but before deciding final weight, check cross-word:
def _next_word_initial_cluster(line: LatinLine, syl_atom_indices: tuple[int, ...]) -> str:
    """Return the consonant cluster from the start of the NEXT word, if this syllable
    is word-final. Otherwise return empty string.

    The atomizer encodes word boundaries on bridges (`ConsonantBridge.has_word_boundary`).
    A word-final atom is followed by a bridge whose `has_word_boundary` is True. We
    return that bridge's `chars` (which contains both the current word's trailing consonants
    AND the next word's leading consonants concatenated — see atomize.py:222 _build_line_units).
    """
    last_atom_idx = max(syl_atom_indices)
    if last_atom_idx >= len(line.bridges):
        return ""  # end of line
    bridge = line.bridges[last_atom_idx]
    if not bridge.has_word_boundary:
        return ""
    # `bridge.chars` is the full inter-atom material; we want only the part of it
    # that is in the NEXT word. The atomizer concatenates the current word's tail
    # with the next word's head, separated by a space. Split on whitespace; take
    # the part after the space (or whichever side corresponds to the next word).
    parts = bridge.chars.split(" ", 1)
    return parts[1] if len(parts) > 1 else ""


def _is_closing_cluster(cluster: str) -> bool:
    """Return True iff a vowel followed by this consonant cluster is long-by-position.

    Mirrors anceps's CLOSE_SYLLABLE (utils.py:21-23): any consonant pair NOT in
    SHORT_COMBINATIONS, PLUS the biconsonantals x and z (handled in Task 4.4).
    """
    if not cluster:
        return False
    # Strip leading whitespace
    cluster = cluster.lstrip()
    if len(cluster) >= 1 and cluster[0] in "xz":
        return True  # x/z are biconsonantal
    if len(cluster) < 2:
        return False
    pair = cluster[:2]
    # SHORT_COMBINATIONS (don't close): stop+liquid, qu/su/gu, *+h
    STOPS = "bpdtcgf"
    LIQUIDS = "rl"
    if pair[0] in STOPS and pair[1] in LIQUIDS:
        return False  # muta cum liquida
    if pair in ("qu", "su", "gu"):
        return False  # u-as-v digraphs
    if pair[1] == "h":
        return False  # *+h
    return True
```

Then modify the weight-decision block (around line 351). Replace:

```python
elif syl.coda:
    weight = PhonWeight.LONG
```

with:

```python
elif syl.coda:
    weight = PhonWeight.LONG
elif _is_closing_cluster(_next_word_initial_cluster(line, syl.atom_indices)):
    weight = PhonWeight.LONG  # long by position across word boundary
```

**Important:** Verify the assumption about `bridge.chars` containing space-separated current-and-next-word consonants by reading `atomize.py:_build_line_units` (line 222). If the actual format differs, adjust `_next_word_initial_cluster` accordingly. If bridges don't preserve next-word consonants at all, add tracking in `atomize.py` first (extra Step 3a in this task).

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/realize.py tests/test_anceps_parity.py
git commit -m "feat(realize): cross-word long-by-position (anceps parity)"
```

---

### Task 4.4: `x` and `z` are biconsonantal

**Files:**
- Modify: `src/latin_ebm/realize.py`
- Modify: `tests/test_anceps_parity.py`

- [ ] **Step 1: Write the failing test**

```python
def test_x_in_coda_makes_syllable_long(lexicon):
    """rex — 'e' followed by 'x' is long by position."""
    line = atomize("rex", lexicon=lexicon)
    syllables = realize(line, decisions={})
    assert syllables[0].weight == PhonWeight.LONG


def test_x_as_onset_does_not_close_previous_word(lexicon):
    """ad Xerxen — the 'a' before 'd X' becomes long: dX closes via biconsonantal x."""
    line = atomize("ad Xerxen", lexicon=lexicon)
    syllables = realize(line, decisions={})
    ad_syl = syllables[0]
    assert ad_syl.weight == PhonWeight.LONG
```

- [ ] **Step 2: Run tests**

Expected: depends. Most likely the first passes, the second fails.

- [ ] **Step 3: Patch `_is_closing_cluster`**

```python
_BICONSONANTAL = {"x", "z"}

def _is_closing_cluster(s: str) -> bool:
    if any(c in _BICONSONANTAL for c in s):
        return True
    # ... existing checks (excluding stop+liquid, qu/su/gu, *h)
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/realize.py tests/test_anceps_parity.py
git commit -m "feat(realize): x and z are biconsonantal (anceps parity)"
```

---

### Task 4.5: Re-measure ceiling after Phase 4

- [ ] **Step 1: Run + commit**

```bash
python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml --books 1,2 --out results/phase4_ceiling.tsv
python scripts/summarize_ceiling.py results/phase4_ceiling.tsv --compare results/phase3_ceiling.tsv > results/phase4_summary.txt
git add results/phase4_ceiling.tsv results/phase4_summary.txt docs/ceiling-progression.md
git commit -m "data: phase-4 ceiling measurement"
```

**Phase 4 expected ceiling delta:** +2 to +4 pp.

---

# Phase 5: Greek / Proper-Name Escape Valve

**Goal:** Mark atoms in words that are absent from both dictionaries (most Greek proper names) as `phonologically_uncertain`, and have the weight-compatibility check apply a softer constraint — admit the candidate rather than rejecting it.

### Task 5.1: Add `phonologically_uncertain` field to `VocalicAtom`

**Files:**
- Modify: `src/latin_ebm/types.py`
- Modify: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
from latin_ebm.types import VocalicAtom

def test_vocalic_atom_has_phonologically_uncertain():
    a = VocalicAtom(index=0, chars="a", word_idx=0, natural_length=None,
                    in_diphthong=False, diphthong_role=None,
                    is_word_final=False, is_word_initial=True)
    assert hasattr(a, "phonologically_uncertain")
    assert a.phonologically_uncertain is False
```

- [ ] **Step 2: Run test**

Expected: FAIL.

- [ ] **Step 3: Add the field**

In `src/latin_ebm/types.py`, add to `VocalicAtom`:

```python
@dataclasses.dataclass(frozen=True)
class VocalicAtom:
    index: int
    chars: str
    word_idx: int
    natural_length: PhonWeight | None
    in_diphthong: bool
    diphthong_role: DiphthongRole | None
    is_word_final: bool
    is_word_initial: bool
    phonologically_uncertain: bool = False  # NEW
```

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Update conftest fixtures**

Add `phonologically_uncertain=False` to all `VocalicAtom(...)` calls in `tests/conftest.py` (or just let the default handle it if your test suite uses keyword args).

- [ ] **Step 6: Run full test suite**

```bash
pytest -x
```

- [ ] **Step 7: Commit**

```bash
git add src/latin_ebm/types.py tests/test_types.py tests/conftest.py
git commit -m "feat(types): phonologically_uncertain field on VocalicAtom"
```

---

### Task 5.2: Add `lexicon.is_unknown_proper_name` detector

**Files:**
- Modify: `src/latin_ebm/lexicon.py`
- Modify: `tests/test_lexicon_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
def test_is_unknown_proper_name(lexicon):
    # Ganymedis: not in MQDQ/Morpheus (Greek proper name); starts uppercase in source
    assert lexicon.is_unknown_proper_name("ganymedis", raw_was_capitalized=True) is True
    # pectore: in Morpheus, not a proper name
    assert lexicon.is_unknown_proper_name("pectore", raw_was_capitalized=False) is False
    # italiam: proper name but well-attested
    assert lexicon.is_unknown_proper_name("italiam", raw_was_capitalized=True) is False
```

- [ ] **Step 2: Run tests**

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def is_unknown_proper_name(self, word: str, raw_was_capitalized: bool) -> bool:
    if not raw_was_capitalized:
        return False
    return not self.is_known_form(word)
```

This is a conservative heuristic — only words that were capitalized in source AND absent from both dictionaries are flagged.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/lexicon.py tests/test_lexicon_alignment.py
git commit -m "feat(lexicon): is_unknown_proper_name heuristic"
```

---

### Task 5.3: Track capitalization through normalization

**Files:**
- Modify: `src/latin_ebm/normalize.py`
- Modify: `src/latin_ebm/atomize.py`
- Modify: `tests/test_normalize.py`

**Diagnosis:** The current normalizer lowercases on the way in; we lose the capitalization signal we need for proper-name detection. Add a parallel output that tracks per-word capitalization.

- [ ] **Step 1: Write the failing test**

```python
def test_normalize_returns_capitalization_mask():
    from latin_ebm.normalize import normalize_with_caps
    norm, caps = normalize_with_caps("Arma virumque Troiae")
    assert norm == "arma virumque troiae"
    assert caps == (True, False, True)  # per-word
```

- [ ] **Step 2: Run test**

Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def normalize_with_caps(raw: str) -> tuple[str, tuple[bool, ...]]:
    norm = normalize(raw)
    caps = tuple(w[0].isupper() for w in raw.split() if w)
    return norm, caps
```

- [ ] **Step 4: Run test, commit**

```bash
git add src/latin_ebm/normalize.py tests/test_normalize.py
git commit -m "feat(normalize): normalize_with_caps preserves word-level capitalization"
```

---

### Task 5.4: Wire capitalization → `phonologically_uncertain` flag

**Files:**
- Modify: `src/latin_ebm/atomize.py`
- Modify: `tests/test_greek_names.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
def test_ganymedis_atoms_flagged_uncertain(lexicon):
    line = atomize("Ganymedis", lexicon=lexicon)
    assert all(a.phonologically_uncertain for a in line.atoms)


def test_italiam_atoms_not_flagged_uncertain(lexicon):
    """italiam — capitalized but known; not flagged."""
    line = atomize("Italiam", lexicon=lexicon)
    assert all(not a.phonologically_uncertain for a in line.atoms)


def test_pectore_atoms_not_flagged_uncertain(lexicon):
    line = atomize("pectore", lexicon=lexicon)
    assert all(not a.phonologically_uncertain for a in line.atoms)
```

- [ ] **Step 2: Run tests**

Expected: FAIL.

- [ ] **Step 3: Implement caps-tracking that survives enclitic expansion**

**Problem:** Enclitic stripping (`_strip_enclitic` in `atomize.py:138`) expands one input word like `"Uirumque"` into two atomizer words `("uirum", "que")`. After expansion, `len(words) != len(caps_per_word)` and `atom.word_idx` indexes into the post-expansion list. We must propagate the capitalization through the expansion.

The cleanest fix: build a parallel `was_capitalized: tuple[bool, ...]` aligned to the **post-expansion** `words` tuple. The enclitic split inherits the stem's capitalization; the enclitic itself is always treated as lowercase.

In `src/latin_ebm/atomize.py`, modify the word-building section (look for where `_strip_enclitic` is called, around line 138-151):

```python
def _split_and_track_caps(raw_words: list[str], caps: tuple[bool, ...]) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    """Apply enclitic stripping and propagate capitalization to the expanded word list."""
    out_words: list[str] = []
    out_caps: list[bool] = []
    for raw_word, was_cap in zip(raw_words, caps):
        stem, enc = _strip_enclitic(raw_word.lower())
        out_words.append(stem)
        out_caps.append(was_cap)
        if enc is not None:
            out_words.append(enc)
            out_caps.append(False)  # enclitic itself is never a proper name
    return tuple(out_words), tuple(out_caps)
```

Then in `atomize`:

```python
def atomize(raw: str, lexicon=None, author: str = "") -> LatinLine:
    normalized, caps_per_raw_word = normalize_with_caps(raw)
    raw_words = normalized.split()
    words, caps_per_word = _split_and_track_caps(raw_words, caps_per_raw_word)
    # ... existing atom construction using `words` ...
    # After atoms are built, flag uncertain ones:
    new_atoms = []
    for atom in atoms:
        word = words[atom.word_idx]
        was_capitalized = caps_per_word[atom.word_idx]
        if lexicon is not None and lexicon.is_unknown_proper_name(word, was_capitalized):
            atom = dataclasses.replace(atom, phonologically_uncertain=True)
        new_atoms.append(atom)
    # ... assemble LatinLine with new_atoms
```

Verify that `atom.word_idx` indexes into the expanded `words` tuple — this should match the existing atomizer behavior since `_strip_enclitic` is already called before atom construction at `atomize.py:138`. If the existing code currently splits enclitics inline and assigns `word_idx` to the expanded position, this aligns naturally.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/atomize.py tests/test_greek_names.py
git commit -m "feat(atomize): flag unknown proper-name atoms as phonologically_uncertain"
```

---

### Task 5.5: Soften weight-compatibility for uncertain atoms

**Files:**
- Modify: `src/latin_ebm/enumerate.py:_weight_compatible`
- Modify: `tests/test_greek_names.py`

- [ ] **Step 1: Write the failing test**

```python
def test_aen_1_28_ganymedis_gold_in_candidates(aeneid_test_lines):
    pairs = {l.corpus_id: (l, g) for l, g in aeneid_test_lines}
    line, gold = pairs["VERG-aene.1.28"]
    from latin_ebm.enumerate import enumerate_parses
    candidates = enumerate_parses(line, meter=None)
    assert any(c.foot_types == gold.foot_types for c in candidates)
```

- [ ] **Step 2: Run test**

Expected: FAIL.

- [ ] **Step 3: Update `_weight_compatible`**

```python
def _weight_compatible(syl, slot, line) -> bool:
    # If any atom in syllable is phonologically_uncertain, accept any slot
    if any(line.atoms[i].phonologically_uncertain for i in syl.atom_indices):
        return True
    # ... existing logic from Task 1.7
```

- [ ] **Step 4: Run test**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latin_ebm/enumerate.py tests/test_greek_names.py
git commit -m "feat(enumerate): phonologically_uncertain atoms admit any slot"
```

---

### Task 5.6: Re-measure ceiling after Phase 5

- [ ] **Step 1: Run + commit**

```bash
python scripts/diagnose_ceiling.py pedecerto-raw/VERG-aene.xml --books 1,2 --out results/phase5_ceiling.tsv
python scripts/summarize_ceiling.py results/phase5_ceiling.tsv --compare results/phase4_ceiling.tsv > results/phase5_summary.txt
git add results/phase5_ceiling.tsv results/phase5_summary.txt docs/ceiling-progression.md
git commit -m "data: phase-5 ceiling measurement"
```

**Phase 5 expected ceiling delta:** +2 to +3 pp.

---

# Phase 6: End-to-End Verification

**Goal:** Confirm cumulative ceiling ≥ 95% and that no regression has occurred in the existing test suite or training pipeline.

### Task 6.1: Run full pytest

- [ ] **Step 1: Run**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 2: If failures, diagnose and fix before moving on**

---

### Task 6.2: Retrain the v1 model with new atomizer outputs

**Files:**
- No code changes; just a run.

- [ ] **Step 1: Run training**

```bash
python scripts/train_v1.py --test-books 1,2 --l2 0.01 --max-iter 200 > results/phase6_train.log 2>&1
```

- [ ] **Step 2: Compare results**

Expected outputs from `train_v1.py`: foot pattern accuracy, syllable accuracy, per-phenomenon F1. Compare against the baseline numbers in `docs/project-summary.md` (67.3% foot accuracy). The new run should be substantially higher because the ceiling is higher.

- [ ] **Step 3: Update `docs/project-summary.md` with new results**

Add a "Post-ceiling-raise" row to the main results table.

- [ ] **Step 4: Commit**

```bash
git add results/phase6_train.log docs/project-summary.md
git commit -m "data: full retrain after ceiling raise; foot acc improves"
```

---

### Task 6.3: Final ceiling progression report

**Files:**
- Modify: `docs/ceiling-progression.md`

- [ ] **Step 1: Write a "summary" section**

Append to `docs/ceiling-progression.md`:

```markdown
## Summary

| Phase | Ceiling | Cumulative Δ | Key change |
|---|---|---|---|
| baseline | <%> | — | — |
| phase-1 | <%> | <%> | Lexicon wired into atomization; natural_length populated |
| phase-2 | <%> | <%> | Dictionary-driven consonantal u/v |
| phase-3 | <%> | <%> | Dictionary-first enclitic handling |
| phase-4 | <%> | <%> | Anceps phonology parity (qv/sv/gv, MCL, x/z, cross-word) |
| phase-5 | <%> | <%> | Greek/proper-name escape valve |

**Goal:** ≥ 95%. **Achieved:** <actual>%.

If the goal is not met, the remaining gap is most likely:
- Words with multiple Morpheus entries (alternate macronizations) where the chosen one disagrees with this author's usage → may need MQDQ-supplement layering (a follow-up plan).
- Rare phenomena (correption, prodelision, brevis in longo) not yet exhaustively handled.
- Edge cases in `_align_to_atoms` that fail silently — instrument with a coverage counter.
```

- [ ] **Step 2: Commit**

```bash
git add docs/ceiling-progression.md
git commit -m "docs: final ceiling progression summary"
```

---

## Self-Review Checklist (run before handing off)

**1. Spec coverage:**
- [x] Atomization edge cases (consonantal u/v, enclitics, vowel chains) — covered in Phases 2, 3
- [x] Morpheus alignment (`qu`, j/v, vowel-count mismatches) — Phase 1
- [x] Anceps phonology parity (qv/sv/gv, MCL, x/z, cross-word LBP) — Phase 4
- [x] Greek/proper-name handling — Phase 5
- [x] Per-phase measurement infrastructure — Phase 0
- [x] Final retrain & verification — Phase 6

**2. Placeholder scan:**
- No "TBD"/"implement later" in code blocks. Verification: every implementation step has executable code.
- A few diagnostic outputs are marked `<%>` in the progression-log template — those are filled in by the engineer after running the script (not code placeholders, output placeholders).
- Task 4.3 includes a verification step ("Verify the assumption about `bridge.chars`...") that may require an extra 3a sub-step if the format differs from what's described. This is intentional — the plan author could not inspect runtime `bridge.chars` without running code.

**3. Type consistency:**
- `lexicon` parameter is added consistently to `atomize`, `_is_consonantal_u`, `_strip_enclitic`.
- `phonologically_uncertain` field is added once to `VocalicAtom` and propagates via `dataclasses.replace`.
- `LineStatus` enum is used consistently throughout `diagnose_ceiling.py`.
- `is_known_form`, `is_consonantal_u`, `is_unknown_proper_name` are all on `VowelLengthLexicon` and named consistently.

**4. Bite-sized check:** every task has ≤ 7 steps, each step ≤ 5 min.

**5. Tests-first check:** every code change is preceded by a failing test in the same task.

**6. API verification (post-review patches applied):**
- `parse_xml(path, lexicon=...)` returns `ParseResult` with `.examples: list[TrainingExample]`, each having `.line` and `.gold_parse`. Confirmed at `pedecerto.py:443-470`.
- `LatinLine` is mutable (`types.py:149` is `@dataclass` not `@dataclass(frozen=True)`); fixture/CLI both rely on direct `line.corpus_id = ...` assignment.
- Weight computation is **inlined** in `realize()` at `realize.py:327-368`; existing code already consults `natural_length` at line 340. Task 1.6 is a regression guard, not a patch.
- `_collect_coda_for_last` at `realize.py:406`, `_max_onset_split` at `realize.py:40`, `_normalize_key` at `lexicon.py:106`, `lookup_aligned` at `lexicon.py:218`, `_align_to_atoms` at `lexicon.py:268` — all citations verified.
- `_NO_STRIP` is at `atomize.py:124-135` (35 was a typo earlier).
- Phase 3 has a precondition gate: if the existing `_NO_STRIP` set covers all observed failures in `phase2_ceiling.tsv`, Phase 3 is skipped (its expected delta is conditional on observed gaps).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-11-ceiling-raise.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because phases are dependency-ordered and you want a checkpoint after each measurement task.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints. Faster but less safe given the size.

**Which approach?**
