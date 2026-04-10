"""Tests for candidate parse enumeration."""

from latin_ebm.atomize import atomize
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.types import FootType


class TestEnumerateBasic:
    def test_simple_line_has_candidates(self):
        """A typical hexameter line should produce at least one candidate."""
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        assert len(candidates) > 0

    def test_all_candidates_have_six_feet(self):
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        for c in candidates:
            assert len(c.foot_types) == 6
            assert c.foot_types[-1] == FootType.FINAL

    def test_all_candidates_have_valid_syllable_count(self):
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        for c in candidates:
            assert 12 <= len(c.syllables) <= 17

    def test_slots_match_syllable_count(self):
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        for c in candidates:
            assert len(c.slots) == len(c.syllables)

    def test_foot_boundaries_consistent(self):
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        for c in candidates:
            assert len(c.foot_boundaries) == 6
            assert c.foot_boundaries[0] == 0

    def test_meter_label(self):
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        for c in candidates:
            assert c.meter == "hexameter"


class TestEnumerateCounts:
    def test_no_ambiguity_line(self):
        """A line with no ambiguity sites: candidates = number of compatible templates."""
        line = atomize("arma")
        # 'arma' has 2 atoms, no sites, so 2 syllables.
        # But 2 syllables is way below hexameter range [12,17].
        # So no candidates.
        candidates = enumerate_parses(line)
        assert len(candidates) == 0

    def test_candidate_count_bounded(self):
        """The plan says typical lines have tens to hundreds of candidates."""
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)
        # Should be manageable — not millions
        assert len(candidates) < 10000


class TestEnumerateCorrectness:
    def test_known_scansion_in_candidates(self):
        """The known scansion of Aeneid 1.1 should be among candidates.

        Known: DDSSD F (feet 1-5: D,D,S,S,D + F)
        """
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        candidates = enumerate_parses(line)

        expected_feet = (
            FootType.DACTYL, FootType.DACTYL, FootType.SPONDEE,
            FootType.SPONDEE, FootType.DACTYL, FootType.FINAL,
        )
        matching = [c for c in candidates if c.foot_types == expected_feet]
        assert len(matching) > 0, (
            f"Expected DDSSD F among {len(candidates)} candidates, "
            f"found foot patterns: {set(c.foot_types for c in candidates)}"
        )
