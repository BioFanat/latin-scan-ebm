"""Tests for serialization: JSON round-trips and Polars/Parquet round-trips."""

import json
import tempfile
from pathlib import Path

from latin_ebm.io import (
    atom_from_dict,
    atom_to_dict,
    bridge_from_dict,
    bridge_to_dict,
    example_from_dict,
    example_to_dict,
    line_from_dict,
    line_to_dict,
    lines_from_polars,
    lines_to_polars,
    load_corpus,
    parse_from_dict,
    parse_to_dict,
    save_corpus,
    save_json,
    load_json,
    scored_parse_from_dict,
    scored_parse_to_dict,
    site_from_dict,
    site_to_dict,
)


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------


class TestAtomRoundTrip:
    def test_round_trip(self, sample_atoms):
        for atom in sample_atoms:
            d = atom_to_dict(atom)
            recovered = atom_from_dict(d)
            assert recovered == atom

    def test_none_natural_length(self, sample_atoms):
        atom = sample_atoms[2]  # natural_length is None
        d = atom_to_dict(atom)
        assert d["natural_length"] is None
        recovered = atom_from_dict(d)
        assert recovered.natural_length is None


class TestBridgeRoundTrip:
    def test_round_trip(self, sample_bridges):
        for bridge in sample_bridges:
            d = bridge_to_dict(bridge)
            recovered = bridge_from_dict(d)
            assert recovered == bridge


class TestSiteRoundTrip:
    def test_round_trip(self, sample_site):
        d = site_to_dict(sample_site)
        recovered = site_from_dict(d)
        assert recovered == sample_site

    def test_dict_contains_enum_strings(self, sample_site):
        d = site_to_dict(sample_site)
        assert d["site_type"] == "SiteType.ELISION"
        assert d["default"] == "SiteChoice.ELIDE"
        assert "SiteChoice.ELIDE" in d["valid_choices"]


class TestLineRoundTrip:
    def test_round_trip(self, sample_line):
        d = line_to_dict(sample_line)
        recovered = line_from_dict(d)
        assert recovered.raw == sample_line.raw
        assert recovered.normalized == sample_line.normalized
        assert recovered.words == sample_line.words
        assert recovered.atoms == sample_line.atoms
        assert recovered.bridges == sample_line.bridges
        assert recovered.sites == sample_line.sites
        assert recovered.author == sample_line.author
        assert recovered.corpus_id == sample_line.corpus_id

    def test_json_serializable(self, sample_line):
        d = line_to_dict(sample_line)
        s = json.dumps(d)
        recovered_dict = json.loads(s)
        recovered = line_from_dict(recovered_dict)
        assert recovered.raw == sample_line.raw


class TestParseRoundTrip:
    def test_round_trip(self, sample_parse):
        d = parse_to_dict(sample_parse)
        recovered = parse_from_dict(d)
        assert recovered.decisions == sample_parse.decisions
        assert recovered.syllables == sample_parse.syllables
        assert recovered.slots == sample_parse.slots
        assert recovered.foot_boundaries == sample_parse.foot_boundaries
        assert recovered.foot_types == sample_parse.foot_types
        assert recovered.caesura == sample_parse.caesura
        assert recovered.bucolic_diaeresis == sample_parse.bucolic_diaeresis
        assert recovered.meter == sample_parse.meter


class TestScoredParseRoundTrip:
    def test_round_trip(self, sample_scored_parse):
        d = scored_parse_to_dict(sample_scored_parse)
        recovered = scored_parse_from_dict(d)
        assert recovered.e_total == sample_scored_parse.e_total
        assert recovered.e_site == sample_scored_parse.e_site
        assert recovered.parse.meter == sample_scored_parse.parse.meter


class TestTrainingExampleRoundTrip:
    def test_round_trip(self, sample_training_example):
        d = example_to_dict(sample_training_example)
        recovered = example_from_dict(d)
        assert recovered.line.raw == sample_training_example.line.raw
        assert recovered.observed == sample_training_example.observed


# ---------------------------------------------------------------------------
# JSON file I/O
# ---------------------------------------------------------------------------


class TestJsonFileIO:
    def test_save_and_load(self, sample_line):
        d = line_to_dict(sample_line)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_json(d, path)
            loaded = load_json(path)
            assert loaded == d


# ---------------------------------------------------------------------------
# Polars round-trip tests
# ---------------------------------------------------------------------------


class TestPolarsRoundTrip:
    def test_single_line(self, sample_line):
        tables = lines_to_polars([sample_line])
        assert "lines" in tables
        assert "atoms" in tables
        assert "bridges" in tables
        assert "sites" in tables
        assert len(tables["lines"]) == 1
        assert len(tables["atoms"]) == 3
        assert len(tables["bridges"]) == 2
        assert len(tables["sites"]) == 1

        recovered = lines_from_polars(tables)
        assert len(recovered) == 1
        r = recovered[0]
        assert r.raw == sample_line.raw
        assert r.normalized == sample_line.normalized
        assert r.words == sample_line.words
        assert len(r.atoms) == len(sample_line.atoms)
        assert len(r.bridges) == len(sample_line.bridges)
        assert len(r.sites) == len(sample_line.sites)

        # Verify atom content
        for orig, rec in zip(sample_line.atoms, r.atoms):
            assert orig.chars == rec.chars
            assert orig.natural_length == rec.natural_length
            assert orig.in_diphthong == rec.in_diphthong

        # Verify site content
        for orig, rec in zip(sample_line.sites, r.sites):
            assert orig.site_type == rec.site_type
            assert orig.valid_choices == rec.valid_choices

    def test_multiple_lines(self, sample_line):
        # Create a second line by modifying metadata
        line2 = line_from_dict(line_to_dict(sample_line))
        line2.corpus_id = "test_002"
        line2.line_num = 2

        tables = lines_to_polars([sample_line, line2])
        assert len(tables["lines"]) == 2
        assert len(tables["atoms"]) == 6  # 3 atoms × 2 lines

        recovered = lines_from_polars(tables)
        assert len(recovered) == 2


# ---------------------------------------------------------------------------
# Parquet round-trip tests
# ---------------------------------------------------------------------------


class TestParquetRoundTrip:
    def test_save_and_load(self, sample_line):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus"
            save_corpus(path, [sample_line])

            # Check files were created
            assert (path / "lines.parquet").exists()
            assert (path / "atoms.parquet").exists()
            assert (path / "bridges.parquet").exists()
            assert (path / "sites.parquet").exists()

            # Load and verify
            recovered = load_corpus(path)
            assert len(recovered) == 1
            assert recovered[0].raw == sample_line.raw
            assert len(recovered[0].atoms) == len(sample_line.atoms)
