"""Candidate enumeration: produce all valid parses Y_m(x) for a line.

Enumerates local decision bundles, materializes syllable sequences,
prunes by syllable count, pairs with meter templates, and yields
the full candidate set for scoring. For hexameter, this is exact
and typically produces tens to hundreds of candidates per line.
"""

from __future__ import annotations

from itertools import product

from latin_ebm.meters import Hexameter
from latin_ebm.realize import realize, syllable_count
from latin_ebm.types import (
    LatinLine,
    MetricalSlot,
    Parse,
    PhonWeight,
    RealizedSyllable,
    SiteChoice,
)


def _weight_compatible(
    syl: RealizedSyllable,
    slot: MetricalSlot,
    line: LatinLine,
) -> bool:
    """Check if a syllable weight satisfies a metrical slot requirement.

    Design philosophy: this is the CANDIDATE-SET MEMBERSHIP filter, not
    the scoring function. We are permissive for ambiguous cases (open
    syllables) and let the EBM's features over `atom.natural_length`
    learn to prefer the correct weight assignment from the candidate set.

    Hard constraints retained (these would never be "wrong" to enforce):
      - Closed syllable → LONG by position
      - Diphthong nucleus → LONG (unless under correption, where realize
        already sets weight=SHORT, allowing BREVE here)

    Open-syllable ambiguity:
      - Treated as length-unknown for filtering purposes, regardless of
        what `atom.natural_length` claims. Both LONGUM and BREVE admit.
      - Rationale: natural_length data conflates natural with positional
        length (MQDQ) or has alignment gaps (Morpheus). Hard-filtering
        on it drops the ceiling without commensurate accuracy gain.
        The realized weight (in syl.weight) still reflects natural_length
        and is visible to the energy features.
    """
    if slot == MetricalSlot.ANCEPS:
        return True

    # Open non-diphthong syllable → permissive on both LONGUM and BREVE.
    # Rationale: lexicon-derived natural_length is unreliable as a hard
    # filter (MQDQ conflates natural+positional, Morpheus has alignment
    # gaps). Treat it as feature evidence (visible via syl.weight) rather
    # than a hard constraint. The energy model learns to disambiguate.
    if syl.is_open:
        is_diphthong = (
            len(syl.atom_indices) >= 2
            and all(
                line.atoms[i].in_diphthong
                for i in syl.atom_indices
                if i < len(line.atoms)
            )
        )
        if not is_diphthong:
            return slot in (MetricalSlot.LONGUM, MetricalSlot.BREVE)

    # Diphthong nucleus or closed syllable: trust syl.weight strictly.
    if slot == MetricalSlot.LONGUM:
        return syl.weight == PhonWeight.LONG
    if slot == MetricalSlot.BREVE:
        return syl.weight == PhonWeight.SHORT
    return False


def _find_word_boundaries(line: LatinLine, active_atoms: list[bool]) -> list[int]:
    """Find syllable indices where word boundaries occur.

    Returns a list of syllable indices where a new word begins.
    Used for caesura and bucolic diaeresis classification.
    """
    boundaries: list[int] = []
    syl_idx = 0
    prev_word = -1

    for atom_idx, atom in enumerate(line.atoms):
        if not active_atoms[atom_idx]:
            continue
        if atom.word_idx != prev_word and prev_word >= 0:
            boundaries.append(syl_idx)
        prev_word = atom.word_idx
        syl_idx += 1

    return boundaries


def enumerate_parses(
    line: LatinLine,
    meter: Hexameter | None = None,
) -> list[Parse]:
    """Enumerate all valid hexameter parses for a line.

    For each combination of local decisions × compatible meter templates,
    produces a Parse object. Returns all valid candidates.
    """
    if meter is None:
        meter = Hexameter()

    valid_range = meter.valid_syllable_counts()

    # Build decision domains
    if not line.sites:
        domains: list[list[tuple[int, SiteChoice]]] = []
    else:
        domains = [
            [(site.index, choice) for choice in site.valid_choices]
            for site in line.sites
        ]

    # If no sites, single decision bundle (empty)
    if not domains:
        bundles: list[dict[int, SiteChoice]] = [{}]
    else:
        bundles = [
            dict(combo) for combo in product(*domains)
        ]

    results: list[Parse] = []

    for decisions in bundles:
        # Fast syllable count check
        m = syllable_count(line, decisions)
        if m not in valid_range:
            continue

        # Full realization
        syllables = realize(line, decisions)

        # Enumerate compatible templates
        templates = meter.enumerate_templates(m)

        for foot_types, slot_sequence in templates:
            # Check weight compatibility
            compatible = all(
                _weight_compatible(syl, slot, line)
                for syl, slot in zip(syllables, slot_sequence)
            )
            if not compatible:
                continue

            # Compute foot boundaries
            foot_boundaries = meter.foot_boundaries_from_template(foot_types)

            # Determine which atoms are active for word boundary detection
            active = [True] * len(line.atoms)
            for site in line.sites:
                choice = decisions.get(site.index, site.default)
                left_idx, right_idx = site.atom_indices
                from latin_ebm.types import SiteType
                if site.site_type == SiteType.ELISION and choice == SiteChoice.ELIDE:
                    active[left_idx] = False
                elif site.site_type == SiteType.PRODELISION:
                    if choice == SiteChoice.ELIDE:
                        active[left_idx] = False
                    elif choice == SiteChoice.ELIDE_RIGHT:
                        active[right_idx] = False
                elif site.site_type == SiteType.SYNIZESIS and choice == SiteChoice.MERGE:
                    active[right_idx] = False
                elif site.site_type == SiteType.DIPHTHONG_SPLIT and choice == SiteChoice.DEFAULT:
                    active[right_idx] = False

            word_boundaries = _find_word_boundaries(line, active)

            # Classify caesura and bucolic diaeresis
            caesura = meter.classify_caesura(
                syllables, foot_boundaries, word_boundaries,
            )
            bucolic = meter.check_bucolic_diaeresis(
                syllables, foot_boundaries, word_boundaries,
            )

            results.append(Parse(
                decisions=decisions,
                syllables=syllables,
                slots=slot_sequence,
                foot_boundaries=foot_boundaries,
                foot_types=foot_types,
                caesura=caesura,
                bucolic_diaeresis=bucolic,
                meter=meter.name,
            ))

    return results


def enumerate_compatible(
    line: LatinLine,
    gold_partial: Parse,
    meter: Hexameter | None = None,
) -> list[Parse]:
    """Enumerate parses compatible with a partially observed gold parse.

    A candidate is compatible if its observed components match the gold.
    Used for partial supervision during training.
    """
    all_parses = enumerate_parses(line, meter)

    compatible: list[Parse] = []
    for parse in all_parses:
        if parse.foot_types != gold_partial.foot_types:
            continue
        if parse.slots != gold_partial.slots:
            continue
        compatible.append(parse)

    return compatible
