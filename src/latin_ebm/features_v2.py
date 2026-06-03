"""Dense per-foot feature extraction for the MLP residual head."""
from __future__ import annotations

import numpy as np

from latin_ebm.types import FootType, LatinLine, Parse, PhonWeight, SiteChoice, SiteType


# Per-foot vector layout (D=22):
#   [0:3]    foot type one-hot (DACTYL, SPONDEE, FINAL)
#   [3:9]    per-syllable weights (3 syls × 2 weights), zero-pad for SPONDEE/FINAL
#   [9:10]   foot position (foot_idx / 5)
#   [10:13]  prev foot type one-hot
#   [13:16]  next foot type one-hot
#   [16:22]  per-foot site-decision summary (6 entries):
#            elision_active, elision_retained, synizesis_merged,
#            diphthong_split_taken, mcl_closed, mcl_opened
PER_FOOT_DIM = 22

_FOOT_INDEX = {FootType.DACTYL: 0, FootType.SPONDEE: 1, FootType.FINAL: 2}


def _foot_syllable_range(parse: Parse, foot_idx: int) -> list[int]:
    bnds = parse.foot_boundaries
    start = bnds[foot_idx]
    end = bnds[foot_idx + 1] if foot_idx + 1 < len(bnds) else len(parse.syllables)
    return list(range(start, end))


def _build_syl_to_foot(parse: Parse) -> dict[int, int]:
    out: dict[int, int] = {}
    for foot_idx in range(len(parse.foot_types)):
        for syl_idx in _foot_syllable_range(parse, foot_idx):
            out[syl_idx] = foot_idx
    return out


def _syllable_index_for_atom(parse: Parse, atom_idx: int) -> int | None:
    for syl_idx, syl in enumerate(parse.syllables):
        if atom_idx in syl.atom_indices:
            return syl_idx
    return None


def per_foot_dense_features(line: LatinLine, parse: Parse) -> np.ndarray:
    out = np.zeros((6, PER_FOOT_DIM), dtype=np.float32)
    n_feet = len(parse.foot_types)
    syl_to_foot = _build_syl_to_foot(parse)

    for foot_idx in range(n_feet):
        ft = parse.foot_types[foot_idx]
        out[foot_idx, _FOOT_INDEX[ft]] = 1.0

        syl_range = _foot_syllable_range(parse, foot_idx)
        for offset, syl_idx in enumerate(syl_range[:3]):
            syl = parse.syllables[syl_idx]
            wo = 3 + offset * 2
            if syl.weight == PhonWeight.LONG:
                out[foot_idx, wo] = 1.0
            else:
                out[foot_idx, wo + 1] = 1.0

        out[foot_idx, 9] = foot_idx / 5.0

        if foot_idx > 0:
            out[foot_idx, 10 + _FOOT_INDEX[parse.foot_types[foot_idx - 1]]] = 1.0
        if foot_idx < n_feet - 1:
            out[foot_idx, 13 + _FOOT_INDEX[parse.foot_types[foot_idx + 1]]] = 1.0

    for site in line.sites:
        if not site.atom_indices:
            continue
        decision = parse.decisions.get(site.index, site.default)
        syl_idx = _syllable_index_for_atom(parse, site.atom_indices[0])
        if syl_idx is None:
            continue
        foot_idx = syl_to_foot.get(syl_idx)
        if foot_idx is None or foot_idx >= 6:
            continue
        ob = 16
        if site.site_type == SiteType.ELISION:
            if decision == SiteChoice.ELIDE:
                out[foot_idx, ob + 0] += 1.0
            else:
                out[foot_idx, ob + 1] += 1.0
        elif site.site_type == SiteType.SYNIZESIS:
            if decision == SiteChoice.MERGE:
                out[foot_idx, ob + 2] += 1.0
        elif site.site_type == SiteType.DIPHTHONG_SPLIT:
            if decision == SiteChoice.SPLIT:
                out[foot_idx, ob + 3] += 1.0
        elif site.site_type == SiteType.MUTA_CUM_LIQUIDA:
            if decision == SiteChoice.CLOSE:
                out[foot_idx, ob + 4] += 1.0
            else:
                out[foot_idx, ob + 5] += 1.0

    return out
