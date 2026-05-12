# Ceiling Progression Log

Ceiling = fraction of Aeneid books 1-2 lines where the gold foot pattern is in the EBM's candidate set.
Total test lines: 1545.

| Phase | Date | Ceiling | Δ vs prev | no_candidates | gold_unreachable | Notes |
|---|---|---|---|---|---|---|
| baseline | 2026-05-11 | 77.35% | — | 144 (9.3%) | 206 (13.3%) | `--no-lexicon`; natural_length always None |

## Failure-reason histogram (baseline)

| Status / Reason | Count |
|---|---|
| gold_unreachable / elision_mismatch | 149 |
| no_candidates / vowel_chain | 86 |
| no_candidates / weight_filter | 58 |
| gold_unreachable / open_syllable_length | 37 |
| gold_unreachable / other | 14 |
| gold_unreachable / diphthong_mismatch | 6 |
