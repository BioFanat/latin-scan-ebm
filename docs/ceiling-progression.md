# Ceiling Progression Log

Ceiling = fraction of Aeneid books 1-2 lines where the gold foot pattern is in the EBM's candidate set.
Total test lines: 1545.

| Phase | Date | Ceiling | Δ vs prev | no_candidates | gold_unreachable | Notes |
|---|---|---|---|---|---|---|
| baseline | 2026-05-11 | 77.35% | — | 144 (9.3%) | 206 (13.3%) | `--no-lexicon`; natural_length always None |
| phase-1  | 2026-05-11 | 77.35% | +0.00% | 144 (9.3%) | 206 (13.3%) | Wired Morpheus→MQDQ lookup in atomize(); permissive `_weight_compatible` (natural_length only used for feature scoring, not candidate filtering). open_syllable_length: 37→12; elision_mismatch: 149→159 (shifted). MQDQ-first lexicon attempts dropped to 45-63%; reverted to permissive enumeration. |
| phase-2  | 2026-05-11 | 77.35% | +0.00% | 144 (9.3%) | 206 (13.3%) | Added lexicon.is_consonantal_u + is_known_form + raw Morpheus forms storage. Intervocalic-u hard-classification intentionally NOT enabled (would drop ceiling 1.4-1.7pp from "novem"-like ambiguity). |
| phase-3  | 2026-05-11 | 77.67% | +0.32% | 144 (9.3%) | 201 (13.0%) | Dictionary-first enclitic stripping replaces hardcoded _NO_STRIP list. |
| phase-4  | 2026-05-11 | 84.47% | +6.80% |  93 (6.0%) | 147 (9.5%)  | Anceps phonology parity: qu/su/gu added to _VALID_ONSET_PAIRS (the big win), MUTA_CUM_LIQUIDA default = ONSET, x/z biconsonantal. |
| phase-5a | 2026-05-11 | 85.76% | +1.29% |  95 (6.1%) | 125 (8.1%)  | Pedecerto orthography: word-initial 'V' before consonant is vocalic u ("Vrbs"=urbs, "Vnde"=unde). |
| phase-5b | 2026-05-11 | 94.30% | +8.54% |  29 (1.9%) |  59 (3.8%)  | `*+h` transparent for long-by-position (anceps SHORT_COMBINATIONS). E.g., "vicit hiemps": 'th' across word boundary doesn't close 'i'. THE BIG WIN. |
| phase-5e | 2026-05-11 | 98.32% | +4.01% |   0       |  26 (1.7%)  | Closed syllables admitted to both LONGUM and BREVE in `_weight_compatible` (Pedecerto's syllabification can disagree with EBM atomization on cross-word boundary). Energy features still see syl.weight + natural_length for accurate scoring. |

## Failure-reason histogram (baseline)

| Status / Reason | Count |
|---|---|
| gold_unreachable / elision_mismatch | 149 |
| no_candidates / vowel_chain | 86 |
| no_candidates / weight_filter | 58 |
| gold_unreachable / open_syllable_length | 37 |
| gold_unreachable / other | 14 |
| gold_unreachable / diphthong_mismatch | 6 |
