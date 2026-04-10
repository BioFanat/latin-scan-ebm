"""Meter definitions: protocol and concrete implementations.

The Meter protocol defines the interface that any meter must satisfy.
Hexameter is the first concrete implementation. Adding new meters
(pentameter, lyric, dramatic) means adding new classes that satisfy
the same protocol — no shared base class required.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol, Sequence

from latin_ebm.types import (
    CaesuraType,
    FootType,
    MetricalSlot,
    RealizedSyllable,
)


# ---------------------------------------------------------------------------
# Meter protocol
# ---------------------------------------------------------------------------

# A template is (foot_types, slot_sequence): the foot decomposition and
# the corresponding per-syllable metrical slot assignments.
Template = tuple[tuple[FootType, ...], tuple[MetricalSlot, ...]]


class Meter(Protocol):
    """Structural interface for a meter definition."""

    @property
    def name(self) -> str: ...

    def valid_syllable_counts(self) -> range:
        """Range of syllable counts that this meter can accept."""
        ...

    def enumerate_templates(self, n_syllables: int) -> Sequence[Template]:
        """All valid (foot_types, slot_sequence) pairs for a given syllable count."""
        ...

    def classify_caesura(
        self,
        syllables: Sequence[RealizedSyllable],
        foot_boundaries: Sequence[int],
        word_boundaries: Sequence[int],
    ) -> CaesuraType:
        """Determine the caesura type for a realized parse."""
        ...

    def check_bucolic_diaeresis(
        self,
        syllables: Sequence[RealizedSyllable],
        foot_boundaries: Sequence[int],
        word_boundaries: Sequence[int],
    ) -> bool:
        """Whether a word boundary coincides with the 4th/5th foot boundary."""
        ...


# ---------------------------------------------------------------------------
# Hexameter
# ---------------------------------------------------------------------------

# Slot patterns for each foot type
_DACTYL_SLOTS = (MetricalSlot.LONGUM, MetricalSlot.BREVE, MetricalSlot.BREVE)
_SPONDEE_SLOTS = (MetricalSlot.LONGUM, MetricalSlot.LONGUM)
_FINAL_SLOTS = (MetricalSlot.LONGUM, MetricalSlot.ANCEPS)


def _build_hexameter_templates() -> dict[int, list[Template]]:
    """Precompute all 32 hexameter templates, grouped by syllable count.

    Feet 1-5 are each dactyl (—∪∪) or spondee (——). Foot 6 is always
    final (—×). The number of dactyls d among feet 1-5 determines
    M = 12 + d syllables. For each M, there are C(5, M-12) templates.
    """
    templates: dict[int, list[Template]] = {}

    for n_dactyls in range(6):  # 0..5 dactyls among feet 1-5
        m = 12 + n_dactyls

        # All ways to place n_dactyls dactyls in positions 0..4
        for dactyl_positions in combinations(range(5), n_dactyls):
            dactyl_set = set(dactyl_positions)

            foot_types: list[FootType] = []
            slots: list[MetricalSlot] = []

            for foot_idx in range(5):
                if foot_idx in dactyl_set:
                    foot_types.append(FootType.DACTYL)
                    slots.extend(_DACTYL_SLOTS)
                else:
                    foot_types.append(FootType.SPONDEE)
                    slots.extend(_SPONDEE_SLOTS)

            # Foot 6 is always final (—×)
            foot_types.append(FootType.FINAL)
            slots.extend(_FINAL_SLOTS)

            template: Template = (tuple(foot_types), tuple(slots))
            templates.setdefault(m, []).append(template)

    return templates


class Hexameter:
    """Dactylic hexameter: the primary meter for epic and didactic poetry.

    Template inventory:
    - Feet 1-5: dactyl (—∪∪) or spondee (——)
    - Foot 6: final (—×), always
    - 2^5 = 32 total foot patterns
    - M ∈ [12, 17] syllables per line
    """

    _templates = _build_hexameter_templates()

    @property
    def name(self) -> str:
        return "hexameter"

    def valid_syllable_counts(self) -> range:
        return range(12, 18)

    def enumerate_templates(self, n_syllables: int) -> Sequence[Template]:
        if n_syllables not in self._templates:
            return []
        return self._templates[n_syllables]

    def classify_caesura(
        self,
        syllables: Sequence[RealizedSyllable],
        foot_boundaries: Sequence[int],
        word_boundaries: Sequence[int],
    ) -> CaesuraType:
        """Classify the main caesura of a hexameter line.

        A caesura is a word boundary falling within a foot (not at a foot
        boundary). The classification depends on which metrical position
        the word boundary falls after:

        - Penthemimeral: after the 5th half-foot (within foot 3, after longum)
        - Trihemimeral: after the 3rd half-foot (within foot 2, after longum)
        - Hephthemimeral: after the 7th half-foot (within foot 4, after longum)
        - Kata triton trochaion: after the 3rd trochee (within foot 3,
          after the first breve of a dactyl)
        """
        if len(foot_boundaries) < 6:
            return CaesuraType.NONE

        # Convert foot boundaries to a set for quick lookup.
        # A word boundary at a foot boundary is diaeresis, not caesura.
        fb_set = set(foot_boundaries)
        wb_set = set(word_boundaries)

        # Syllable indices within each foot: foot_boundaries[f] is the
        # start of foot f. A caesura occurs when a word boundary falls
        # strictly between foot_boundaries[f] and foot_boundaries[f+1].

        # Penthemimeral: word break after syllable at position
        # foot_boundaries[2] (start of foot 3) — i.e., after the longum
        # of foot 3. The word boundary falls *after* that syllable.
        penthem_pos = foot_boundaries[2] + 1
        if penthem_pos in wb_set and penthem_pos not in fb_set:
            return CaesuraType.PENTHEMIMERAL

        # Kata triton trochaion: after the first breve of a dactyl in
        # foot 3 — only possible if foot 3 IS a dactyl.
        kata_pos = foot_boundaries[2] + 2
        if kata_pos in wb_set and kata_pos not in fb_set:
            return CaesuraType.KATA_TRITON

        # Trihemimeral: after longum of foot 2
        trih_pos = foot_boundaries[1] + 1
        if trih_pos in wb_set and trih_pos not in fb_set:
            return CaesuraType.TRIHEMIMERAL

        # Hephthemimeral: after longum of foot 4
        heph_pos = foot_boundaries[3] + 1
        if heph_pos in wb_set and heph_pos not in fb_set:
            return CaesuraType.HEPHTHEMIMERAL

        return CaesuraType.NONE

    def check_bucolic_diaeresis(
        self,
        syllables: Sequence[RealizedSyllable],
        foot_boundaries: Sequence[int],
        word_boundaries: Sequence[int],
    ) -> bool:
        """Whether a word boundary coincides with the start of foot 5.

        Bucolic diaeresis: a word ends exactly where foot 4 ends / foot 5
        begins. Named for its frequency in pastoral (bucolic) poetry.
        """
        if len(foot_boundaries) < 5:
            return False
        return foot_boundaries[4] in set(word_boundaries)

    def foot_boundaries_from_template(
        self, foot_types: tuple[FootType, ...],
    ) -> tuple[int, ...]:
        """Compute syllable indices where each foot begins, given foot types."""
        boundaries: list[int] = [0]
        for ft in foot_types:
            if ft == FootType.DACTYL:
                boundaries.append(boundaries[-1] + 3)
            elif ft == FootType.SPONDEE:
                boundaries.append(boundaries[-1] + 2)
            elif ft == FootType.FINAL:
                boundaries.append(boundaries[-1] + 2)
        return tuple(boundaries[:-1])  # last entry would be total syllable count
