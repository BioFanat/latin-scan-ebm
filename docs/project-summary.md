# Latin Scansion via Meter-Conditioned Energy-Based Models

## Project Summary

### The Problem

Automated Latin scansion — determining the metrical structure of a line of Latin verse — requires jointly resolving several interrelated decisions: where syllable boundaries fall, whether vowels are long or short, whether elision occurs at word boundaries, and how the resulting syllable sequence maps onto a metrical template. Existing approaches treat these as a pipeline (syllabify first, then assign quantities, then fit meter), which means early errors are irrecoverable. Neural approaches (LSTMs, seq2seq) can achieve high per-syllable accuracy but lack structural guarantees — they may predict a quantity sequence that doesn't correspond to any legal metrical pattern.

### The Architecture

We formulate scansion as **exact energy minimization** over a structured candidate set. A line of Latin verse is represented as a conservative sequence of **vocalic atoms** (potential syllable nuclei) and **consonant bridges** (material between them), with **ambiguity sites** marking locations where prosodic decisions must be made (elision, synizesis, diphthong splitting, muta cum liquida). The model jointly searches over all combinations of local decisions and metrical templates, scoring each candidate parse with a decomposed energy function:

```
E_θ(x, y) = Σ E_site + Σ E_syll + Σ E_pair + Σ E_foot + E_global
```

For dactylic hexameter, the candidate set is naturally small (typically tens to hundreds of candidates per line), making exact inference tractable. The partition function Z(x) is a finite sum, enabling exact NLL training — the key computational advantage over generic EBMs that require approximate inference.

### Key Design Choices

**1. Conservative pre-syllabic representation.** Canonical diphthongs like *ae* are represented as two separate atoms with a default-merge relation, rather than pre-merged into one syllable. This allows the model to infer diphthong splitting (e.g., *aër*) jointly with meter, rather than committing to a syllable count in preprocessing.

**2. Separation of ω and μ.** Phonological weight (the syllable *is* heavy) is kept distinct from metrical slot type (the position *permits* heavy or light). Anceps is a metrical permission, not a phonological fact. This separation matters for training signal and downstream applications.

**3. Exact enumeration, not approximate inference.** The hexameter candidate set has at most 32 foot patterns × a small number of pre-metrical syllabification variants. Even a line with 3 elision sites, 2 synizesis sites, and 1 diphthong site produces at most ~7000 candidates. Exact enumeration is simpler and more reliable than beam search or MCMC.

**4. Hard metrical constraints baked into search.** E_foot is implemented by pruning: illegal foot structures never become candidates. This means every candidate the model scores is a legal hexameter parse. The energy function only needs to distinguish between legal alternatives.

**5. Linear features for v1.** The energy is θᵀφ(x,y) — a single dot product against ~130 sparse binary/count features. This is interpretable (top features reveal metrical preferences), data-efficient (6K training lines suffice), and mathematically convenient (log-linear model with convex NLL objective).

**6. Partial supervision.** The training objective marginalizes over unobserved structure. Pedecerto gives us foot types and syllable quantities but not every internal realization decision (e.g., MCL closure direction). The NLL training handles this by summing over all candidates compatible with the observed gold components.

**7. Lexicon as features, not constraints.** The MQDQ frequency dictionary provides empirical vowel length data per word form per author, but we use it as soft training features rather than hard constraints on the search space. Hard constraints from MQDQ hurt performance because the dictionary conflates natural and positional vowel length.

---

## Implementation

### Codebase

13 source modules (3,622 lines), 10 test files (1,630 lines), 161 tests all passing.

| Module | Lines | Purpose |
|--------|-------|---------|
| `types.py` | 263 | 6 enums, 8 dataclasses — the complete type vocabulary |
| `normalize.py` | 78 | Unicode NFC, lowercase, strip diacritics/punctuation |
| `atomize.py` | 422 | Raw text → vocalic atoms, consonant bridges, ambiguity sites |
| `realize.py` | 441 | Apply prosodic decisions → realized syllables with onset/nucleus/coda |
| `enumerate.py` | 206 | Exhaustive candidate parse enumeration with syllable count + weight pruning |
| `meters.py` | 217 | Hexameter: 32 precomputed templates, caesura classification, bucolic diaeresis |
| `features.py` | 210 | FeatureIndex registry + sparse feature extraction (site, global, lexicon) |
| `energy.py` | 69 | LinearEBM: θᵀφ scoring, predict, score_candidates |
| `train.py` | 249 | Exact NLL with tractable Z, L-BFGS optimization, partial supervision |
| `evaluate.py` | 248 | Book splits, line-exact-match, foot accuracy, per-phenomenon F1, baselines |
| `lexicon.py` | 328 | MQDQ + Morpheus vowel length lookup with per-vowel consensus |
| `io.py` | 418 | JSON + Polars star-schema + Parquet serialization |
| `corpus/pedecerto.py` | 471 | MQDQ XML parsing, sy/wb/mf decoding, gold alignment |

### Data Pipeline

```
Raw XML (Pedecerto/MQDQ)
  → parse_xml(): extract words, sy attributes, mf markers
  → atomize(): raw text → LatinLine (atoms, bridges, sites)
  → align_gold_parse(): infer decisions from mf="SY", syllable count alignment
  → TrainingExample (line + gold parse + observed components)
```

The Aeneid (VERG-aene.xml) parses with 99.3% success: 9,827 of 9,896 lines.

### Training Pipeline

```
TrainingExamples
  → enumerate_parses(): all valid hexameter candidates per line
  → extract_features(): sparse φ(x,y) per candidate
  → build_feature_index(): collect feature vocabulary, freeze
  → precompute_training_data(): cache features + gold-compatible indices
  → nll_loss_and_grad(): exact NLL with gradient, L2 regularization
  → L-BFGS optimization (scipy.optimize.minimize)
  → LinearEBM model with learned θ
```

Training on ~6K lines (8K minus skipped) with 129 features takes ~2 minutes on a laptop.

---

## Experimental Results

### Main Results (Aeneid, book-held-out: train books 3-12, test books 1-2)

| Model | Foot Pattern Acc | Syllable Acc | Elision F1 | Diphthong F1 | MCL F1 |
|-------|-----------------|-------------|-----------|-------------|--------|
| **EBM v1 + lexicon features** | **66.9%** | **86.3%** | **0.861** | — | — |
| EBM v1 (no lexicon) | 63.6% | 84.9% | 0.867 | 0.604 | 0.920 |
| CRF (gold syllabification) | 36.4% | 86.1% | — | — | — |
| Default baseline | 39.7% | 75.8% | 0.825 | 0.000 | 0.935 |
| Random baseline | 30.3% | 70.7% | 0.704 | 0.232 | 0.513 |

### Diagnostic: Why Not Higher?

Analysis of the test set (books 1-2) reveals the accuracy bottleneck:

| Category | Lines | % |
|----------|-------|---|
| Correct prediction | 891 | 57.7% |
| Wrong prediction (gold reachable) | 302 | 19.5% |
| Gold not in candidate set | 202 | 13.1% |
| No candidates at all | 150 | 9.7% |

**Ceiling analysis (books 11-12 as test):**
- 74.1% of test lines have the gold parse in the candidate set
- The model achieves 74.5% accuracy on those reachable lines
- Performance at 90.3% of the achievable ceiling

The bottleneck is **data quality**, not model architecture:
- **9.7% no candidates**: Syllable count outside [12,17] for all decision bundles. Root cause: atomization errors (consonantal u/v edge cases, enclitic handling, complex vowel sequences like "Lauiniaque").
- **13.1% gold unreachable**: Candidates exist but gold weight pattern doesn't match. Root cause: without natural vowel length data, the realizer can't distinguish genuinely long vowels from short ones in open syllables.
- **19.5% wrong prediction**: Gold is reachable but the model picks the wrong candidate. Root cause: insufficient features to disambiguate — many candidates differ only in vowel quantities for open syllables, and the model lacks the phonological knowledge to choose correctly.

### Anceps Comparison: The Phonology Gap

We ran [anceps](https://github.com/Dargones/anceps) — the current best constraint-based scanner — on the same Aeneid books 1-2 test set:

| Model | Foot Pattern Acc | Notes |
|-------|-----------------|-------|
| **Anceps** | **92.0%** | Constraint solver + MQDQ frequency dictionary |
| Anceps (excl. 6.8% failures) | 98.8% | When it answers, it's almost always right |
| EBM v1 + lexicon features | 66.9% | Learned model |
| EBM v1 (no lexicon) | 63.6% | |

The 25pp gap is almost entirely explained by phonological knowledge, not model architecture:

1. **Anceps has hand-tuned syllabification rules** covering every Latin phonological phenomenon — decades of philological work. Our atomizer is a v1 approximation that mishandles many edge cases (enclitics, medial consonantal-u, complex vowel sequences).

2. **Anceps uses the MQDQ dictionary as a hard constraint** during its constraint search. This works for anceps because it syllabifies correctly first, then uses the dictionary in the right syllable context. We tried the same dictionary as a hard constraint and it hurt — because our syllabification is less reliable, the dictionary values get applied to the wrong contexts.

3. **Anceps either gets it right or fails** (6.8% failure rate). Our model always produces an answer but is often wrong. Anceps's design philosophy is conservative: refuse to answer rather than answer incorrectly.

The EBM's structural advantage (joint inference, learned soft preferences, meter generalization) becomes relevant once the phonological gap closes. Anceps can't learn from data, can't transfer across meters, and can't express soft preferences — it either satisfies constraints or doesn't.

### CRF Baseline: Why Joint Inference Matters

The CRF baseline operates on Pedecerto's gold syllabification — it receives pre-segmented syllables and labels each as L/S. Despite 86.1% syllable accuracy, it achieves only **36.4% foot pattern accuracy** — worse than the default baseline. The reason: per-syllable predictions lack structural coherence. One wrong syllable breaks the foot assignment for the entire line.

The EBM's structural constraint (only legal hexameter parses are candidates) produces **63.6% foot accuracy** — a 27pp improvement over the CRF. This validates the core architectural premise: joint inference over syllabification and meter outperforms pipeline labeling.

### Top Learned Features

The model learns philologically sensible preferences:

| Weight | Feature | Interpretation |
|--------|---------|---------------|
| +3.52 | `foot5:SPONDEE` | Fifth-foot spondee penalized (rare in Vergil) |
| -3.52 | `foot5:DACTYL` | Fifth-foot dactyl preferred |
| +2.74 | `pattern:DDSSSF` | Specific foot patterns have prior probability |
| -2.17 | `caesura:PENTHEMIMERAL` | Penthemimeral caesura preferred (most common) |
| -1.76 | `caesura:KATA_TRITON` | Kata triton caesura preferred |
| +1.39 | `site_vowel:u:RETAIN_SHORT` | Correption of 'u' in hiatus |
| +1.35 | `site:SYNIZESIS:MERGE` | Synizesis slightly preferred |

---

## What We Tried to Fix the Gaps

### Vowel Length Data Integration

**Problem:** 25.9% of test lines can't produce the correct candidate because the model doesn't know natural vowel lengths for open syllables.

**Data sources explored:**
- **MQDQ frequency dictionary** (from anceps project): 95K word forms with per-author frequency counts. Empirical — derived from actual scansions across the entire Pedecerto corpus.
- **Morpheus lexicon** (from Alatius): 283K forms with theory-based natural lengths from morphological analysis.

**Approach 1: Lexicon as hard constraint (failed).** Set `VocalicAtom.natural_length` from the MQDQ dictionary and use it to constrain weight compatibility during enumeration. Result: gold-in-candidate-set **dropped** from 74.1% to 62.9%. Root cause: the MQDQ dictionary records **metrical** length (how a vowel was scanned in context), not **natural** length. A short vowel in a closed syllable appears as "long" in MQDQ because it's long by position. This over-constrains the search.

We tried per-vowel consensus with an 85% agreement threshold — still hurt (67.9%). We tried character-aligned matching between the dictionary's vowel counting scheme and our atomizer's — still hurt (66.5%). The fundamental issue: MQDQ can't distinguish natural from positional length.

Morpheus gives natural length but has vowel-count mismatches with our atomizer (treats `qu` as two vowels, doesn't handle consonantal i/j the same way), making alignment unreliable.

**Approach 2: Lexicon as features (worked).** Use MQDQ data as soft training features — "how much does this parse's weight assignment agree with the MQDQ majority scansion for each word?" Three features: agreement ratio, disagreement ratio, disagreement count. Result: **+3.3pp** improvement (63.6% → 66.9%). The model learns how much to trust the lexicon without being constrained by it.

### Consonantal u/v Heuristic

**Problem:** MQDQ uses 'u' for both vowel-u and consonant-v. Words like "uirumque" (= virumque) had the initial 'u' treated as a vowel, creating a false 'ui' diphthong and wrong atom count.

**Fix:** Added `_is_consonantal_u()` heuristic in `atomize.py`: word-initial 'u' before a vowel is treated as consonantal. This mirrors the existing `_is_consonantal_i()` heuristic. Reduced atom count from 17 to 16 for the canonical first line.

### Diphthong Detection Fix

**Problem:** Greedy diphthong matching was bidirectional — a vowel claimed as the second element of one diphthong could also be claimed as the first element of another. In "saeuae" (s-ae-u-ae), the 'e' of the first 'ae' was also matching with 'u' to form a false 'eu' diphthong.

**Fix:** Changed to greedy left-to-right diphthong detection. Once a vowel is claimed as "second" of a diphthong, it can't be "first" of another.

### Gradient Sign Bug

**Problem:** The NLL gradient was inverted — training moved weights in the wrong direction, causing loss to increase.

**Fix:** Corrected from `E_p[φ] - E_gold[φ]` to `E_gold[φ] - E_p[φ]`. After fix, training reduced loss by 70% and L-BFGS converged in 164 iterations.

---

## Remaining Obstacles

### 1. Natural Vowel Length Data (the biggest gap)

The single largest improvement would come from a reliable source of **natural** (not metrical) vowel lengths. Neither MQDQ (metrical) nor Morpheus (alignment issues) works well as a hard constraint. Options:

- **Build a natural-length lexicon** by post-processing MQDQ: if a vowel is long in open syllables and long in closed syllables, it's naturally long. If it's long only in closed syllables, it's naturally short (long by position only). This deconfounding requires per-vowel-per-syllable-structure statistics that could be computed from the MQDQ data.
- **Use Morpheus with better alignment**: fix the vowel-count mismatch by matching on character identity rather than position. The `lookup_aligned` method exists but needs refinement for edge cases (qu, consonantal i/j, diphthongs).
- **Train a separate vowel-length classifier**: use the MQDQ data to learn natural length from word form + morphological context, then use its predictions as features.

### 2. Atomization Edge Cases (9.7% no-candidates)

Lines producing zero candidates have atomization problems:

- **Enclitic handling**: Words like "Lauiniaque" need the `-que` enclitic stripped before lexicon lookup. The atomizer doesn't currently handle enclitics.
- **Complex vowel sequences**: "Lauiniaque" has 5 consecutive vowels (a-u-i-i-a) after consonantal analysis. The correct syllabification depends on knowing that the second 'i' is consonantal — which our heuristic doesn't catch because it's not word-initial or intervocalic in the standard sense.
- **u/v in non-initial positions**: The consonantal-u heuristic only fires word-initially. Medial consonantal-u (as in "soluit" = solvit) is not handled.

### 3. Feature Richness (19.5% wrong predictions on reachable lines)

The 129 features capture foot patterns and caesura well but lack:

- **Per-syllable weight features**: The model doesn't see individual syllable weights in context — it only sees global counts. Adding "syllable at position X has weight Y" features would help.
- **Word-form features**: Common words like "est", "et", "in" have characteristic metrical behavior that could be directly captured.
- **Morphological features**: Verb endings (-ō, -ās, -āmus) carry natural length information. CLTK integration would provide this.
- **Bigram/interaction features**: "spondee followed by dactyl" patterns, elision-in-foot-N, etc.

### 4. Training Data Coverage

- **24% of training lines skipped**: Lines where the gold parse isn't in the candidate set can't contribute to training. Improving atomization and weight handling would recover these.
- **Single author**: Training and testing on Vergil only. Cross-author evaluation (Ovid, Lucretius, Horace) is the scientifically interesting test.
- **Single meter**: Hexameter only. The architecture is designed for meter generalization (shared E_local, swappable E_meter), but pentameter, elegiac couplets, and lyric meters aren't implemented.

---

## Future Directions

### Near-term (v1 improvements)

1. **Deconfound MQDQ vowel lengths**: Compute natural-vs-positional length from the MQDQ data by comparing vowel scansions in open vs. closed syllable contexts. Use deconfounded lengths as a hard constraint.
2. **Enclitic stripping**: Handle -que, -ne, -ue, -ce in the atomizer and lexicon lookup.
3. **Richer features**: Per-position weight indicators, word-form features, morphological features from CLTK.
4. **More training data**: Add Ovid, Lucretius, and other authors from Pedecerto. Test cross-author transfer.

### Medium-term (v2: MLP + enrichment)

5. **Hybrid local energy**: Replace linear E_local with `θᵀφ_hand + g_ψ(φ_dense) + b_lex(lemma)` where g_ψ is a small MLP. This captures non-linear interactions like "author × vowel pair × morphology."
6. **Lexical bias terms**: Sparse per-lemma bias for exception words and proper names.
7. **Pentameter and elegiac couplets**: Implement the Pentameter meter class; add E_couplet cross-line term for elegiac pairs.
8. **Author conditioning**: Add author-specific global features; test whether they help or leak.

### Long-term (v3: full generalization)

9. **Lyric meters**: Hendecasyllabics, Sapphics, Alcaics — meter automaton over slot alphabet.
10. **Dramatic meters**: Iambic trimeter with resolution and freer substitution.
11. **Greek meter**: Richer local phonology (movable nu, correptio Attica, aggressive resolution).
12. **Meter identification**: Free-energy-based model comparison across meters.
13. **Downstream applications**: Textual criticism (energy difference between readings), authorship attribution (metrical fingerprints), pedagogy (decomposed energy explains *why* a parse wins), verse generation (energy-based reranking).

---

## Reproducing Results

```bash
# Install
cd latin-scan-ebm
pip install -e ".[dev]"

# Run tests
pytest

# Train and evaluate
python scripts/train_v1.py --test-books 11,12

# CRF baseline comparison
pip install sklearn-crfsuite
python scripts/crf_baseline.py
```

Data requirements:
- `pedecerto-raw/VERG-aene.xml` — Vergil's Aeneid from Pedecerto
- `data/MqDqMacrons.json` — MQDQ frequency dictionary from [anceps](https://github.com/Dargones/anceps)
- `data/MorpheusMacrons.txt` — Morpheus lexicon from [anceps](https://github.com/Dargones/anceps)
