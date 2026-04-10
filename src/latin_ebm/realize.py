"""Realization: apply prosodic decisions (ρ) to atoms → realized syllables.

Given a LatinLine and a decision bundle, produce the realized syllable
sequence g_x(ρ) = (s_1, ..., s_M) with phonological weights assigned
and onset/nucleus/coda fully populated.
"""

from __future__ import annotations

from latin_ebm.atomize import STOPS, LIQUIDS
from latin_ebm.types import (
    AmbiguitySite,
    LatinLine,
    PhonWeight,
    RealizedSyllable,
    SiteChoice,
    SiteType,
)


# ---------------------------------------------------------------------------
# Maximal onset: which consonant clusters can begin a Latin syllable
# ---------------------------------------------------------------------------

# Valid Latin onset clusters (simplified for v1).
# A single consonant is always a valid onset.
# Two-consonant onsets: stop+liquid, s+stop, s+liquid, and a few others.
_VALID_ONSET_PAIRS = set()
for _s in STOPS:
    for _lq in LIQUIDS:
        _VALID_ONSET_PAIRS.add(_s + _lq)
# s + stop
for _s2 in STOPS:
    _VALID_ONSET_PAIRS.add("s" + _s2)
# s + liquid
for _lq2 in LIQUIDS:
    _VALID_ONSET_PAIRS.add("s" + _lq2)


def _max_onset_split(consonants: str) -> tuple[str, str]:
    """Split a consonant cluster into (coda, onset) using maximal onset.

    Assigns as many consonants as possible to the onset of the following
    syllable, subject to the constraint that the onset must be a valid
    Latin onset cluster.
    """
    if not consonants:
        return ("", "")

    # Try taking all consonants as onset, then progressively fewer
    for i in range(len(consonants)):
        onset = consonants[i:]
        coda = consonants[:i]

        if len(onset) == 1:
            return (coda, onset)

        if len(onset) == 2 and onset in _VALID_ONSET_PAIRS:
            return (coda, onset)

        # Three-consonant onsets: s + stop + liquid (e.g., "str")
        if (len(onset) == 3
                and onset[0] == "s"
                and onset[1] in STOPS
                and onset[2] in LIQUIDS):
            return (coda, onset)

    # Fallback: all consonants go to coda (shouldn't happen in Latin)
    return (consonants, "")


# ---------------------------------------------------------------------------
# Core realization
# ---------------------------------------------------------------------------


def _build_decision_map(
    sites: tuple[AmbiguitySite, ...],
    decisions: dict[int, SiteChoice],
) -> dict[int, SiteChoice]:
    """Build a complete decision map with defaults for unspecified sites."""
    result: dict[int, SiteChoice] = {}
    for site in sites:
        if site.index in decisions:
            result[site.index] = decisions[site.index]
        else:
            result[site.index] = site.default
    return result


def _get_site_for_atoms(
    sites: tuple[AmbiguitySite, ...],
    left_idx: int,
    right_idx: int,
) -> AmbiguitySite | None:
    """Find the ambiguity site involving atoms at left_idx and right_idx."""
    for site in sites:
        if site.atom_indices == (left_idx, right_idx):
            return site
    return None


def syllable_count(line: LatinLine, decisions: dict[int, SiteChoice]) -> int:
    """Fast path: count realized syllables without building full syllable objects.

    This counts "active" atoms — atoms that survive after applying
    elision, merge, and split decisions.
    """
    all_decisions = _build_decision_map(line.sites, decisions)

    active = [True] * len(line.atoms)

    for site in line.sites:
        choice = all_decisions[site.index]
        left_idx, right_idx = site.atom_indices

        if site.site_type == SiteType.ELISION:
            if choice == SiteChoice.ELIDE:
                active[left_idx] = False
            # RETAIN, RETAIN_SHORT: both stay active

        elif site.site_type == SiteType.PRODELISION:
            if choice == SiteChoice.ELIDE:
                active[left_idx] = False
            elif choice == SiteChoice.ELIDE_RIGHT:
                active[right_idx] = False

        elif site.site_type == SiteType.SYNIZESIS:
            if choice == SiteChoice.MERGE:
                active[right_idx] = False
            # DEFAULT: both stay active

        elif site.site_type == SiteType.DIPHTHONG_SPLIT:
            if choice == SiteChoice.DEFAULT:
                # Diphthong stays merged → one nucleus
                active[right_idx] = False
            elif choice == SiteChoice.SPLIT:
                # Split → two nuclei (both stay active)
                pass

        # MCL doesn't change active count
        # elif site.site_type == SiteType.MUTA_CUM_LIQUIDA: pass

    return sum(active)


def realize(
    line: LatinLine,
    decisions: dict[int, SiteChoice],
) -> tuple[RealizedSyllable, ...]:
    """Apply realization decisions to atoms and produce syllables.

    Returns a tuple of RealizedSyllable with all fields populated:
    atom_indices, onset, nucleus, coda, is_open, weight.
    """
    if not line.atoms:
        return ()

    all_decisions = _build_decision_map(line.sites, decisions)

    # Phase 1: Determine which atoms are active and build nucleus groups.
    # A nucleus group is a list of atom indices that merge into one syllable.
    active = [True] * len(line.atoms)
    # merged_into[i] = j means atom i's nucleus merges into atom j's
    merged_into: dict[int, int] = {}

    # Track MCL decisions for weight computation
    mcl_decisions: dict[int, SiteChoice] = {}  # bridge_index → choice

    # Track correption sites
    correption_atoms: set[int] = set()

    for site in line.sites:
        choice = all_decisions[site.index]
        left_idx, right_idx = site.atom_indices

        if site.site_type == SiteType.ELISION:
            if choice == SiteChoice.ELIDE:
                active[left_idx] = False
            elif choice == SiteChoice.RETAIN_SHORT:
                correption_atoms.add(left_idx)

        elif site.site_type == SiteType.PRODELISION:
            if choice == SiteChoice.ELIDE:
                active[left_idx] = False
            elif choice == SiteChoice.ELIDE_RIGHT:
                active[right_idx] = False

        elif site.site_type == SiteType.SYNIZESIS:
            if choice == SiteChoice.MERGE:
                active[right_idx] = False
                merged_into[right_idx] = left_idx

        elif site.site_type == SiteType.DIPHTHONG_SPLIT:
            if choice == SiteChoice.DEFAULT:
                # Merge: diphthong stays as one nucleus
                active[right_idx] = False
                merged_into[right_idx] = left_idx
            # SPLIT: both stay active (separate nuclei)

        elif site.site_type == SiteType.MUTA_CUM_LIQUIDA:
            # The bridge index between atoms left_idx and right_idx
            # is left_idx (bridge[i] is between atom[i] and atom[i+1])
            mcl_decisions[left_idx] = choice

    # Phase 2: Build nucleus groups (which atom indices form each syllable)
    nucleus_groups: list[list[int]] = []
    for i, atom in enumerate(line.atoms):
        if not active[i]:
            # Check if merged into a previous atom
            if i in merged_into:
                target = merged_into[i]
                for group in nucleus_groups:
                    if target in group:
                        group.append(i)
                        break
            continue
        nucleus_groups.append([i])

    # Phase 3: Assign consonant material to onset and coda.
    syllables: list[RealizedSyllable] = []

    for syl_idx, group in enumerate(nucleus_groups):
        # Build nucleus from atom chars
        nucleus = "".join(line.atoms[i].chars for i in group)
        atom_indices = tuple(group)

        # Determine if this is a diphthong nucleus (merged canonical pair)
        is_diphthong = (
            len(group) == 2
            and line.atoms[group[0]].in_diphthong
            and line.atoms[group[1]].in_diphthong
        )

        # Get consonant material BEFORE this syllable's first atom
        first_atom_idx = group[0]
        if first_atom_idx > 0:
            bridge_before = line.bridges[first_atom_idx - 1]
            cons_before = bridge_before.chars
        else:
            cons_before = ""

        # Onset: from the bridge before this syllable.
        # For the first syllable, all initial consonants are onset.
        if syl_idx == 0:
            onset = cons_before  # everything before first vowel
            # Don't modify the bridge — no coda to assign to a predecessor
        else:
            # Split consonants between previous syllable's coda and this onset
            # using maximal onset principle
            # But we already assigned the previous coda... we need a different approach.
            # Let's collect all consonants and split them after.
            onset = ""  # will be set in the split phase

        coda = ""  # will be set in the split phase

        syllables.append(RealizedSyllable(
            atom_indices=atom_indices,
            onset=onset,
            nucleus=nucleus,
            coda=coda,
            is_open=True,  # placeholder
            weight=PhonWeight.SHORT,  # placeholder
        ))

    # Phase 4: Split consonants between syllables using maximal onset.
    # Recompute onset/coda properly by looking at inter-syllable consonants.
    final_syllables: list[RealizedSyllable] = []

    for syl_idx in range(len(syllables)):
        syl = syllables[syl_idx]
        group = nucleus_groups[syl_idx]
        first_atom = group[0]
        last_atom = group[-1]

        # Onset: consonants coming from the left
        if syl_idx == 0:
            onset = _collect_onset_for_first(line, first_atom)
        else:
            # Get consonants between previous syllable's last atom and this atom
            prev_last = nucleus_groups[syl_idx - 1][-1]
            between = _collect_consonants_between(line, prev_last, first_atom, active)

            # Check for MCL decision affecting this split
            mcl_choice = mcl_decisions.get(prev_last)
            if mcl_choice == SiteChoice.ONSET:
                # MCL cluster goes to onset → all consonants to onset
                coda_prev = ""
                onset = between
            elif mcl_choice == SiteChoice.CLOSE:
                # MCL cluster closes preceding → split so stop goes to coda
                coda_prev, onset = _max_onset_split(between)
                # But force at least the stop into the coda
                if between and not coda_prev:
                    coda_prev = between[0]
                    onset = between[1:]
            else:
                coda_prev, onset = _max_onset_split(between)

            # Update previous syllable with its coda
            if final_syllables:
                prev = final_syllables[-1]
                final_syllables[-1] = RealizedSyllable(
                    atom_indices=prev.atom_indices,
                    onset=prev.onset,
                    nucleus=prev.nucleus,
                    coda=coda_prev,
                    is_open=(coda_prev == ""),
                    weight=prev.weight,  # will be recomputed
                )

        # Coda for last syllable: consonants after the last atom of the line
        if syl_idx == len(syllables) - 1:
            coda = _collect_coda_for_last(line, last_atom)
        else:
            coda = ""  # will be set when processing next syllable

        final_syllables.append(RealizedSyllable(
            atom_indices=syl.atom_indices,
            onset=onset,
            nucleus=syl.nucleus,
            coda=coda,
            is_open=(coda == ""),
            weight=PhonWeight.SHORT,  # placeholder, computed next
        ))

    # Phase 5: Compute weights.
    result: list[RealizedSyllable] = []
    for syl_idx, syl in enumerate(final_syllables):
        group = nucleus_groups[syl_idx]

        # Determine if the nucleus is inherently long
        is_diphthong = (
            len(group) == 2
            and line.atoms[group[0]].in_diphthong
            and line.atoms[group[1]].in_diphthong
        )

        # Check if any atom in the group has known long natural length
        has_long_vowel = any(
            line.atoms[i].natural_length == PhonWeight.LONG
            for i in group
        )

        # Check correption
        is_correption = any(i in correption_atoms for i in group)

        # Weight determination
        if is_correption:
            weight = PhonWeight.SHORT
        elif syl.coda:
            # Closed syllable → long (heavy by position)
            weight = PhonWeight.LONG
        elif is_diphthong:
            weight = PhonWeight.LONG
        elif has_long_vowel:
            weight = PhonWeight.LONG
        else:
            weight = PhonWeight.SHORT

        result.append(RealizedSyllable(
            atom_indices=syl.atom_indices,
            onset=syl.onset,
            nucleus=syl.nucleus,
            coda=syl.coda,
            is_open=syl.is_open,
            weight=weight,
        ))

    return tuple(result)


# ---------------------------------------------------------------------------
# Helpers for consonant collection
# ---------------------------------------------------------------------------


def _collect_onset_for_first(line: LatinLine, first_atom_idx: int) -> str:
    """Collect all consonant material before the first active atom."""
    # There are no bridges before atom 0, but there might be leading
    # consonants encoded in the word before the first vowel.
    # Actually, in our representation, the bridges are between atoms.
    # The consonants before atom 0 in the word aren't in any bridge.
    # They're implicit in the normalized text. For simplicity, we
    # extract them from the word text.
    if first_atom_idx == 0:
        # No consonants before the very first atom — or rather,
        # the word-initial consonants before the first vowel.
        word = line.words[line.atoms[0].word_idx]
        onset = ""
        for ch in word:
            if ch in "aeiouy":
                break
            onset += ch
        return onset

    # If atoms before first_atom_idx are all inactive, collect
    # bridge consonants between them
    parts: list[str] = []
    for i in range(first_atom_idx):
        if i < len(line.bridges):
            parts.append(line.bridges[i].chars)
    return "".join(parts)


def _collect_coda_for_last(line: LatinLine, last_atom_idx: int) -> str:
    """Collect consonant material after the last active atom."""
    if last_atom_idx >= len(line.bridges):
        # Last atom in the line — get trailing consonants from word
        word = line.words[line.atoms[last_atom_idx].word_idx]
        # Find the last vowel position and return everything after
        last_v = -1
        for i, ch in enumerate(word):
            if ch in "aeiouy":
                last_v = i
        if last_v >= 0 and last_v < len(word) - 1:
            return word[last_v + 1:]
        return ""

    # Collect bridges from last_atom_idx to end
    parts: list[str] = []
    for i in range(last_atom_idx, len(line.bridges)):
        parts.append(line.bridges[i].chars)
    return "".join(parts)


def _collect_consonants_between(
    line: LatinLine,
    left_atom_idx: int,
    right_atom_idx: int,
    active: list[bool],
) -> str:
    """Collect all consonant material between two active atoms.

    This includes bridge consonants and any bridges spanning inactive atoms.
    """
    parts: list[str] = []
    for i in range(left_atom_idx, right_atom_idx):
        if i < len(line.bridges):
            parts.append(line.bridges[i].chars)
    return "".join(parts)
