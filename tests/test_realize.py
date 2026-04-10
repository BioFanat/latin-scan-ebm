"""Tests for syllable realization."""

from latin_ebm.atomize import atomize
from latin_ebm.realize import realize, syllable_count
from latin_ebm.types import PhonWeight, SiteChoice, SiteType


class TestSyllableCount:
    def test_arma_no_sites(self):
        """'arma' has no ambiguity sites → 2 syllables."""
        line = atomize("arma")
        assert syllable_count(line, {}) == 2

    def test_elision_reduces_count(self):
        """Elision at a site reduces syllable count by 1."""
        line = atomize("multum ille")
        # Find the elision site
        elision_sites = [s for s in line.sites if s.site_type == SiteType.ELISION]
        assert len(elision_sites) == 1
        site = elision_sites[0]

        # Default (ELIDE) should reduce count
        count_elided = syllable_count(line, {site.index: SiteChoice.ELIDE})
        count_retained = syllable_count(line, {site.index: SiteChoice.RETAIN})
        assert count_elided == count_retained - 1

    def test_diphthong_default_merges(self):
        """Default at diphthong site → merge (one fewer syllable)."""
        line = atomize("troiae")
        diph_sites = [s for s in line.sites if s.site_type == SiteType.DIPHTHONG_SPLIT]
        assert len(diph_sites) >= 1
        site = diph_sites[0]

        count_merged = syllable_count(line, {site.index: SiteChoice.DEFAULT})
        count_split = syllable_count(line, {site.index: SiteChoice.SPLIT})
        assert count_split == count_merged + 1

    def test_synizesis_merge(self):
        """Synizesis MERGE reduces syllable count by 1."""
        line = atomize("meo")
        syn_sites = [s for s in line.sites if s.site_type == SiteType.SYNIZESIS]
        assert len(syn_sites) == 1
        site = syn_sites[0]

        count_default = syllable_count(line, {site.index: SiteChoice.DEFAULT})
        count_merged = syllable_count(line, {site.index: SiteChoice.MERGE})
        assert count_merged == count_default - 1


class TestRealize:
    def test_arma_syllables(self):
        """'arma' → 2 syllables: 'ar' (closed, long) + 'ma' (open, short)."""
        line = atomize("arma")
        sylls = realize(line, {})
        assert len(sylls) == 2

        # First syllable: 'a' with coda 'r' (or 'rm' split)
        assert sylls[0].nucleus == "a"
        assert sylls[0].weight == PhonWeight.LONG  # closed by consonant

        # Second syllable: 'a' (open)
        assert sylls[1].nucleus == "a"

    def test_atom_indices_populated(self):
        """Realized syllables should have non-empty atom_indices."""
        line = atomize("arma")
        sylls = realize(line, {})
        for syl in sylls:
            assert len(syl.atom_indices) > 0

    def test_elision_removes_syllable(self):
        """Elision should remove a syllable and adjust consonant material."""
        line = atomize("multum ille")
        elision_site = [s for s in line.sites if s.site_type == SiteType.ELISION][0]

        sylls_elided = realize(line, {elision_site.index: SiteChoice.ELIDE})
        sylls_retained = realize(line, {elision_site.index: SiteChoice.RETAIN})

        assert len(sylls_elided) == len(sylls_retained) - 1

    def test_diphthong_merge(self):
        """Default at diphthong site → ae becomes one nucleus."""
        line = atomize("troiae")
        diph_sites = [s for s in line.sites if s.site_type == SiteType.DIPHTHONG_SPLIT]
        site = diph_sites[0]

        sylls_merged = realize(line, {site.index: SiteChoice.DEFAULT})
        sylls_split = realize(line, {site.index: SiteChoice.SPLIT})

        # Merged: ae is one nucleus
        merged_nuclei = [s.nucleus for s in sylls_merged]
        assert "ae" in merged_nuclei

        # Split: a and e are separate
        assert len(sylls_split) == len(sylls_merged) + 1

    def test_closed_syllable_is_long(self):
        """A syllable closed by a consonant should have LONG weight."""
        line = atomize("arma")
        sylls = realize(line, {})
        # First syllable 'ar-' is closed
        closed_sylls = [s for s in sylls if not s.is_open]
        for s in closed_sylls:
            assert s.weight == PhonWeight.LONG

    def test_mcl_onset_keeps_light(self):
        """MCL with ONSET choice: preceding syllable stays open/light."""
        line = atomize("patris")
        mcl_sites = [s for s in line.sites if s.site_type == SiteType.MUTA_CUM_LIQUIDA]
        assert len(mcl_sites) == 1
        site = mcl_sites[0]

        sylls_onset = realize(line, {site.index: SiteChoice.ONSET})
        # First syllable 'pa-' should be open when MCL goes to onset
        assert sylls_onset[0].is_open

    def test_mcl_close_makes_heavy(self):
        """MCL with CLOSE choice: preceding syllable gets closed/heavy."""
        line = atomize("patris")
        mcl_sites = [s for s in line.sites if s.site_type == SiteType.MUTA_CUM_LIQUIDA]
        site = mcl_sites[0]

        sylls_closed = realize(line, {site.index: SiteChoice.CLOSE})
        # First syllable 'pat-' should have weight LONG
        assert sylls_closed[0].weight == PhonWeight.LONG

    def test_full_line_syllable_count(self):
        """Full line with default decisions should produce reasonable count."""
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        sylls = realize(line, {})
        # With defaults, should be in the hexameter range
        assert 12 <= len(sylls) <= 20  # loose bound for now
