"""Tests for meter definitions: template enumeration, slot sequences, caesura."""

from math import comb

from latin_ebm.meters import Hexameter
from latin_ebm.types import (
    CaesuraType,
    FootType,
    MetricalSlot,
    PhonWeight,
    RealizedSyllable,
)


class TestHexameterTemplates:
    def setup_method(self):
        self.hex = Hexameter()

    def test_valid_syllable_range(self):
        assert self.hex.valid_syllable_counts() == range(12, 18)

    def test_total_template_count(self):
        """2^5 = 32 total foot patterns across all syllable counts."""
        total = sum(
            len(self.hex.enumerate_templates(m))
            for m in self.hex.valid_syllable_counts()
        )
        assert total == 32

    def test_template_count_per_syllable_count(self):
        """For M syllables, there are C(5, M-12) templates."""
        for m in range(12, 18):
            expected = comb(5, m - 12)
            actual = len(self.hex.enumerate_templates(m))
            assert actual == expected, f"M={m}: expected {expected}, got {actual}"

    def test_all_spondee_template(self):
        """M=12: all spondees (feet 1-5), exactly 1 template."""
        templates = self.hex.enumerate_templates(12)
        assert len(templates) == 1
        foot_types, slots = templates[0]
        # Feet 1-5 should all be SPONDEE, foot 6 is FINAL
        assert foot_types == (
            FootType.SPONDEE, FootType.SPONDEE, FootType.SPONDEE,
            FootType.SPONDEE, FootType.SPONDEE, FootType.FINAL,
        )
        # 5 spondees (2 slots each) + 1 final (2 slots) = 12 slots
        assert len(slots) == 12

    def test_all_dactyl_template(self):
        """M=17: all dactyls (feet 1-5), exactly 1 template."""
        templates = self.hex.enumerate_templates(17)
        assert len(templates) == 1
        foot_types, slots = templates[0]
        assert foot_types == (
            FootType.DACTYL, FootType.DACTYL, FootType.DACTYL,
            FootType.DACTYL, FootType.DACTYL, FootType.FINAL,
        )
        assert len(slots) == 17

    def test_slot_sequence_length_matches_syllable_count(self):
        """Every template's slot sequence must have exactly M elements."""
        for m in self.hex.valid_syllable_counts():
            for foot_types, slots in self.hex.enumerate_templates(m):
                assert len(slots) == m, (
                    f"M={m}, feet={foot_types}: slot length {len(slots)} != {m}"
                )

    def test_foot_6_always_final(self):
        """Every template must have FootType.FINAL as the last foot."""
        for m in self.hex.valid_syllable_counts():
            for foot_types, _ in self.hex.enumerate_templates(m):
                assert foot_types[-1] == FootType.FINAL

    def test_foot_types_length_always_six(self):
        """Every template must have exactly 6 feet."""
        for m in self.hex.valid_syllable_counts():
            for foot_types, _ in self.hex.enumerate_templates(m):
                assert len(foot_types) == 6

    def test_slot_patterns_per_foot(self):
        """Verify the slot pattern for each foot type."""
        for m in self.hex.valid_syllable_counts():
            for foot_types, slots in self.hex.enumerate_templates(m):
                pos = 0
                for ft in foot_types:
                    if ft == FootType.DACTYL:
                        assert slots[pos] == MetricalSlot.LONGUM
                        assert slots[pos + 1] == MetricalSlot.BREVE
                        assert slots[pos + 2] == MetricalSlot.BREVE
                        pos += 3
                    elif ft == FootType.SPONDEE:
                        assert slots[pos] == MetricalSlot.LONGUM
                        assert slots[pos + 1] == MetricalSlot.LONGUM
                        pos += 2
                    elif ft == FootType.FINAL:
                        assert slots[pos] == MetricalSlot.LONGUM
                        assert slots[pos + 1] == MetricalSlot.ANCEPS
                        pos += 2

    def test_invalid_syllable_count_returns_empty(self):
        assert self.hex.enumerate_templates(11) == []
        assert self.hex.enumerate_templates(18) == []
        assert self.hex.enumerate_templates(0) == []

    def test_name(self):
        assert self.hex.name == "hexameter"


class TestHexameterFootBoundaries:
    def setup_method(self):
        self.hex = Hexameter()

    def test_all_spondee_boundaries(self):
        """All spondees: boundaries at 0, 2, 4, 6, 8, 10."""
        feet = (
            FootType.SPONDEE, FootType.SPONDEE, FootType.SPONDEE,
            FootType.SPONDEE, FootType.SPONDEE, FootType.FINAL,
        )
        boundaries = self.hex.foot_boundaries_from_template(feet)
        assert boundaries == (0, 2, 4, 6, 8, 10)

    def test_all_dactyl_boundaries(self):
        """All dactyls: boundaries at 0, 3, 6, 9, 12, 15."""
        feet = (
            FootType.DACTYL, FootType.DACTYL, FootType.DACTYL,
            FootType.DACTYL, FootType.DACTYL, FootType.FINAL,
        )
        boundaries = self.hex.foot_boundaries_from_template(feet)
        assert boundaries == (0, 3, 6, 9, 12, 15)

    def test_mixed_boundaries(self):
        """DDSSD F: boundaries at 0, 3, 6, 8, 10, 13."""
        feet = (
            FootType.DACTYL, FootType.DACTYL, FootType.SPONDEE,
            FootType.SPONDEE, FootType.DACTYL, FootType.FINAL,
        )
        boundaries = self.hex.foot_boundaries_from_template(feet)
        assert boundaries == (0, 3, 6, 8, 10, 13)


class TestHexameterCaesura:
    def setup_method(self):
        self.hex = Hexameter()

    def _make_dummy_syllables(self, n: int) -> list[RealizedSyllable]:
        """Create n dummy syllables for testing caesura logic."""
        return [
            RealizedSyllable(
                atom_indices=(i,), onset="", nucleus="a", coda="",
                is_open=True, weight=PhonWeight.SHORT,
            )
            for i in range(n)
        ]

    def test_penthemimeral_caesura(self):
        """Word boundary after longum of foot 3 → penthemimeral."""
        # All spondees: feet start at 0,2,4,6,8,10
        # Foot 3 starts at position 4. Longum is at position 4.
        # Word boundary after position 4 → at position 5.
        sylls = self._make_dummy_syllables(12)
        foot_boundaries = (0, 2, 4, 6, 8, 10)
        word_boundaries = [5]  # after longum of foot 3
        result = self.hex.classify_caesura(sylls, foot_boundaries, word_boundaries)
        assert result == CaesuraType.PENTHEMIMERAL

    def test_trihemimeral_caesura(self):
        """Word boundary after longum of foot 2 → trihemimeral."""
        sylls = self._make_dummy_syllables(12)
        foot_boundaries = (0, 2, 4, 6, 8, 10)
        word_boundaries = [3]  # after longum of foot 2
        result = self.hex.classify_caesura(sylls, foot_boundaries, word_boundaries)
        assert result == CaesuraType.TRIHEMIMERAL

    def test_no_caesura(self):
        """No word boundaries within feet → NONE."""
        sylls = self._make_dummy_syllables(12)
        foot_boundaries = (0, 2, 4, 6, 8, 10)
        word_boundaries = [0, 2, 4, 6, 8, 10]  # all at foot boundaries (= diaereses)
        result = self.hex.classify_caesura(sylls, foot_boundaries, word_boundaries)
        assert result == CaesuraType.NONE


class TestHexameterBucolicDiaeresis:
    def setup_method(self):
        self.hex = Hexameter()

    def _make_dummy_syllables(self, n: int) -> list[RealizedSyllable]:
        return [
            RealizedSyllable(
                atom_indices=(i,), onset="", nucleus="a", coda="",
                is_open=True, weight=PhonWeight.SHORT,
            )
            for i in range(n)
        ]

    def test_bucolic_diaeresis_present(self):
        """Word boundary at start of foot 5 → bucolic diaeresis."""
        # All spondees: foot 5 starts at position 8
        foot_boundaries = (0, 2, 4, 6, 8, 10)
        word_boundaries = [8]
        sylls = self._make_dummy_syllables(12)
        assert self.hex.check_bucolic_diaeresis(sylls, foot_boundaries, word_boundaries)

    def test_bucolic_diaeresis_absent(self):
        """No word boundary at start of foot 5 → no bucolic diaeresis."""
        foot_boundaries = (0, 2, 4, 6, 8, 10)
        word_boundaries = [3, 5, 7]  # none at position 8
        sylls = self._make_dummy_syllables(12)
        assert not self.hex.check_bucolic_diaeresis(sylls, foot_boundaries, word_boundaries)
