"""Tests for core types: enum coverage, dataclass construction, frozen semantics."""

import pytest

from latin_ebm.types import (
    AmbiguitySite,
    CaesuraType,
    ConsonantBridge,
    FootType,
    MetricalSlot,
    PhonWeight,
    RealizedSyllable,
    SiteChoice,
    SiteType,
    TrainingExample,
    VocalicAtom,
)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestEnums:
    def test_phon_weight_members(self):
        assert set(PhonWeight) == {PhonWeight.LONG, PhonWeight.SHORT}

    def test_metrical_slot_members(self):
        assert set(MetricalSlot) == {
            MetricalSlot.LONGUM, MetricalSlot.BREVE, MetricalSlot.ANCEPS,
        }

    def test_foot_type_members(self):
        assert set(FootType) == {
            FootType.DACTYL, FootType.SPONDEE, FootType.FINAL,
        }

    def test_site_type_members(self):
        assert len(SiteType) == 5

    def test_site_choice_members(self):
        assert len(SiteChoice) == 9

    def test_caesura_type_members(self):
        assert len(CaesuraType) == 5


# ---------------------------------------------------------------------------
# Frozen dataclass semantics
# ---------------------------------------------------------------------------


class TestFrozenSemantics:
    def test_vocalic_atom_is_frozen(self):
        atom = VocalicAtom(
            index=0, chars="a", word_idx=0,
            natural_length=PhonWeight.LONG,
            in_diphthong=False, diphthong_role=None,
            is_word_final=False, is_word_initial=True,
        )
        with pytest.raises(AttributeError):
            atom.chars = "e"  # type: ignore[misc]

    def test_consonant_bridge_is_frozen(self):
        bridge = ConsonantBridge(chars="rm", has_word_boundary=False, is_muta_cum_liquida=False)
        with pytest.raises(AttributeError):
            bridge.chars = "st"  # type: ignore[misc]

    def test_ambiguity_site_is_frozen(self):
        site = AmbiguitySite(
            index=0, site_type=SiteType.ELISION,
            atom_indices=(0, 1),
            valid_choices=(SiteChoice.ELIDE, SiteChoice.RETAIN),
            default=SiteChoice.ELIDE,
        )
        with pytest.raises(AttributeError):
            site.site_type = SiteType.SYNIZESIS  # type: ignore[misc]

    def test_realized_syllable_is_frozen(self):
        syll = RealizedSyllable(
            atom_indices=(0,), onset="", nucleus="a", coda="rm",
            is_open=False, weight=PhonWeight.LONG,
        )
        with pytest.raises(AttributeError):
            syll.weight = PhonWeight.SHORT  # type: ignore[misc]

    def test_parse_is_frozen(self, sample_parse):
        with pytest.raises(AttributeError):
            sample_parse.meter = "pentameter"  # type: ignore[misc]

    def test_scored_parse_is_frozen(self, sample_scored_parse):
        with pytest.raises(AttributeError):
            sample_scored_parse.e_total = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LatinLine is mutable (metadata attachment)
# ---------------------------------------------------------------------------


class TestLatinLineMutability:
    def test_metadata_can_be_set(self, sample_line):
        sample_line.author = "vergil"
        assert sample_line.author == "vergil"

    def test_corpus_id_can_be_set(self, sample_line):
        sample_line.corpus_id = "verg_aen_1_1"
        assert sample_line.corpus_id == "verg_aen_1_1"


# ---------------------------------------------------------------------------
# Construction and field access
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_atom_fields(self, sample_atoms):
        atom = sample_atoms[0]
        assert atom.index == 0
        assert atom.chars == "a"
        assert atom.word_idx == 0
        assert atom.natural_length == PhonWeight.LONG
        assert atom.in_diphthong is False
        assert atom.diphthong_role is None
        assert atom.is_word_final is False
        assert atom.is_word_initial is True

    def test_atom_unknown_length(self, sample_atoms):
        assert sample_atoms[2].natural_length is None

    def test_bridge_fields(self, sample_bridges):
        bridge = sample_bridges[0]
        assert bridge.chars == "rm"
        assert bridge.has_word_boundary is False
        assert bridge.is_muta_cum_liquida is False

    def test_bridge_with_word_boundary(self, sample_bridges):
        assert sample_bridges[1].has_word_boundary is True

    def test_site_fields(self, sample_site):
        assert sample_site.site_type == SiteType.ELISION
        assert sample_site.atom_indices == (1, 2)
        assert SiteChoice.ELIDE in sample_site.valid_choices
        assert SiteChoice.RETAIN in sample_site.valid_choices

    def test_line_structural_counts(self, sample_line):
        assert len(sample_line.atoms) == 3
        assert len(sample_line.bridges) == 2  # len(atoms) - 1
        assert len(sample_line.sites) == 1

    def test_line_words_are_tuple(self, sample_line):
        assert isinstance(sample_line.words, tuple)

    def test_parse_decisions(self, sample_parse):
        assert sample_parse.decisions[0] == SiteChoice.ELIDE

    def test_parse_meter(self, sample_parse):
        assert sample_parse.meter == "hexameter"

    def test_scored_parse_energy(self, sample_scored_parse):
        assert sample_scored_parse.e_total == -3.5
        assert sample_scored_parse.e_site == -1.0

    def test_training_example_default_observed(self, sample_training_example):
        assert "decisions" in sample_training_example.observed
        assert "syllables" in sample_training_example.observed
        assert "caesura" in sample_training_example.observed

    def test_training_example_custom_observed(self, sample_line, sample_parse):
        ex = TrainingExample(
            line=sample_line,
            gold_parse=sample_parse,
            observed=frozenset({"syllables", "foot_types"}),
        )
        assert "decisions" not in ex.observed
        assert "syllables" in ex.observed
