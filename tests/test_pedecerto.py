"""Tests for Pedecerto XML parsing."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from latin_ebm.corpus.pedecerto import (
    decode_sy,
    parse_line_element,
    parse_xml,
)
from latin_ebm.types import (
    CaesuraType,
    FootType,
    MetricalSlot,
    PhonWeight,
    SiteChoice,
    SiteType,
)


# ---------------------------------------------------------------------------
# Inline XML fixtures
# ---------------------------------------------------------------------------

AENEID_1_1_3_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<document>
    <head>
        <author>Vergilius</author>
        <title>Aeneis</title>
    </head>
    <body>
        <division title="1">
            <line name="1" metre="H" pattern="DDSS">
                <word sy="1A1b" wb="CF">Arma</word>
                <word sy="1c2A2b" wb="CF">uirumque</word>
                <word sy="2c3A" wb="CM">cano,</word>
                <word sy="3T4A" wb="CM">Troiae</word>
                <word sy="4T" wb="DI">qui</word>
                <word sy="5A5b" wb="CF">primus</word>
                <word sy="5c" wb="DI">ab</word>
                <word sy="6A6X">oris</word>
            </line>
            <line name="2" metre="H" pattern="DSDS">
                <word sy="1A1b1c2A" wb="CM">Italiam</word>
                <word sy="2T3A" wb="CM">fato</word>
                <word sy="3b3c4A" wb="CM">profugus</word>
                <word sy="4T5A5b5c" wb="DI">Lauiniaque</word>
                <word sy="6A6X">uenit</word>
            </line>
            <line name="3" metre="H" pattern="DSSS">
                <word sy="1A1b1c" wb="DI">Litora,</word>
                <word sy="2A" mf="SY">multum</word>
                <word sy="2T" mf="SY">ille</word>
                <word sy="3A" wb="CM">et</word>
                <word sy="3T4A" wb="CM">terris</word>
                <word sy="4T5A5b" wb="CF">iactatus</word>
                <word sy="5c" wb="DI">et</word>
                <word sy="6A6X">alto</word>
            </line>
            <line name="corrupt_line" metre="H" pattern="corrupt">
                <word sy="1A">bad</word>
            </line>
        </division>
    </body>
</document>
"""


@pytest.fixture
def sample_xml_path():
    """Write the sample XML to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(AENEID_1_1_3_XML)
        return Path(f.name)


# ---------------------------------------------------------------------------
# sy decoding
# ---------------------------------------------------------------------------


class TestDecodeSy:
    def test_simple(self):
        result = decode_sy("1A1b")
        assert len(result) == 2
        assert result[0].foot == 1
        assert result[0].position == "A"
        assert result[0].slot == MetricalSlot.LONGUM
        assert result[0].weight == PhonWeight.LONG
        assert result[1].foot == 1
        assert result[1].position == "b"
        assert result[1].slot == MetricalSlot.BREVE
        assert result[1].weight == PhonWeight.SHORT

    def test_multi_foot(self):
        result = decode_sy("1c2A2b")
        assert len(result) == 3
        assert result[0].foot == 1
        assert result[1].foot == 2
        assert result[2].foot == 2

    def test_final_anceps(self):
        result = decode_sy("6A6X")
        assert len(result) == 2
        assert result[1].position == "X"
        assert result[1].slot == MetricalSlot.ANCEPS
        assert result[1].weight is None

    def test_spondee_thesis(self):
        result = decode_sy("3T4A")
        assert result[0].position == "T"
        assert result[0].slot == MetricalSlot.LONGUM
        assert result[0].weight == PhonWeight.LONG

    def test_syllable_count(self):
        # "uirumque" has sy="1c2A2b" → 3 syllables
        assert len(decode_sy("1c2A2b")) == 3
        # "Lauiniaque" has sy="4T5A5b5c" → 4 syllables
        assert len(decode_sy("4T5A5b5c")) == 4


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------


class TestParseLineElement:
    def _parse_first_line(self):
        root = ET.fromstring(AENEID_1_1_3_XML)
        line_el = root.find(".//line[@name='1']")
        return parse_line_element(line_el, "Vergilius", "Aeneis", "1")

    def test_aeneid_1_1_parses(self):
        example = self._parse_first_line()
        assert example is not None

    def test_aeneid_1_1_metadata(self):
        example = self._parse_first_line()
        assert example.line.author == "Vergilius"
        assert example.line.work == "Aeneis"
        assert example.line.book == "1"
        assert example.line.line_num == 1

    def test_aeneid_1_1_foot_types(self):
        """Line 1 pattern DDSS → feet should be D,D,S,S,D,F.

        Note: pattern only encodes feet 1-4. Foot 5 is derived from sy.
        From sy data: foot 5 has positions A,b,c → DACTYL.
        """
        example = self._parse_first_line()
        ft = example.gold_parse.foot_types
        assert ft[0] == FootType.DACTYL   # foot 1: 1A,1b,1c
        assert ft[1] == FootType.DACTYL   # foot 2: 2A,2b,2c
        assert ft[2] == FootType.SPONDEE  # foot 3: 3A,3T
        assert ft[3] == FootType.SPONDEE  # foot 4: 4A,4T
        assert ft[4] == FootType.DACTYL   # foot 5: 5A,5b,5c
        assert ft[5] == FootType.FINAL    # foot 6: 6A,6X

    def test_aeneid_1_1_syllable_count(self):
        """DDSSD F → 3+3+2+2+3+2 = 15 syllables."""
        example = self._parse_first_line()
        assert len(example.gold_parse.syllables) == 15

    def test_aeneid_1_1_slots(self):
        """Check the metrical slot sequence for DDSSD F."""
        example = self._parse_first_line()
        slots = example.gold_parse.slots
        # Foot 1 (dactyl): L B B
        assert slots[0] == MetricalSlot.LONGUM
        assert slots[1] == MetricalSlot.BREVE
        assert slots[2] == MetricalSlot.BREVE
        # Foot 6 (final): L X
        assert slots[-2] == MetricalSlot.LONGUM
        assert slots[-1] == MetricalSlot.ANCEPS

    def test_aeneid_1_1_caesura(self):
        """Line 1: 'cano,' has wb="CM" and its last syllable is 3A → penthemimeral."""
        example = self._parse_first_line()
        assert example.gold_parse.caesura == CaesuraType.PENTHEMIMERAL

    def test_aeneid_1_1_meter(self):
        example = self._parse_first_line()
        assert example.gold_parse.meter == "hexameter"

    def test_corrupt_line_skipped(self):
        root = ET.fromstring(AENEID_1_1_3_XML)
        line_el = root.find(".//line[@name='corrupt_line']")
        result = parse_line_element(line_el, "Vergilius", "Aeneis", "1")
        assert result is None

    def test_line_with_elision(self):
        """Line 3 has mf='SY' on 'multum' and 'ille' — elision markers."""
        root = ET.fromstring(AENEID_1_1_3_XML)
        line_el = root.find(".//line[@name='3']")
        example = parse_line_element(line_el, "Vergilius", "Aeneis", "1")
        assert example is not None
        assert len(example.gold_parse.syllables) == 14

    def test_alignment_populates_decisions(self):
        """Gold parse should have non-empty decisions after alignment."""
        example = self._parse_first_line()
        assert len(example.gold_parse.decisions) > 0

    def test_alignment_elision_inferred(self):
        """Line 3: mf='SY' on 'multum' and 'ille' → ELIDE decisions."""
        root = ET.fromstring(AENEID_1_1_3_XML)
        line_el = root.find(".//line[@name='3']")
        example = parse_line_element(line_el, "Vergilius", "Aeneis", "1")
        # Should have ELIDE decisions for the elision sites
        elide_decisions = [
            (k, v) for k, v in example.gold_parse.decisions.items()
            if v == SiteChoice.ELIDE
        ]
        assert len(elide_decisions) >= 1  # at least one elision

    def test_alignment_retain_for_no_elision(self):
        """Line 1 has no mf='SY' → elision sites should be RETAIN."""
        example = self._parse_first_line()
        # Line 1 has an elision site (arma + uirumque) but no mf="SY"
        for site in example.line.sites:
            if site.site_type == SiteType.ELISION:
                decision = example.gold_parse.decisions.get(site.index)
                assert decision == SiteChoice.RETAIN


# ---------------------------------------------------------------------------
# Full XML file parsing
# ---------------------------------------------------------------------------


class TestParseXml:
    def test_parse_sample(self, sample_xml_path):
        result = parse_xml(sample_xml_path)
        # 3 valid lines + 1 corrupt (skipped)
        assert result.total == 4
        assert result.skipped == 1
        assert len(result.examples) == 3

    def test_all_examples_have_metadata(self, sample_xml_path):
        result = parse_xml(sample_xml_path)
        for ex in result.examples:
            assert ex.line.author == "Vergilius"
            assert ex.line.work == "Aeneis"
            assert ex.line.book == "1"

    def test_all_examples_are_hexameter(self, sample_xml_path):
        result = parse_xml(sample_xml_path)
        for ex in result.examples:
            assert ex.gold_parse.meter == "hexameter"
            assert len(ex.gold_parse.foot_types) == 6
            assert ex.gold_parse.foot_types[-1] == FootType.FINAL


class TestParseRealFile:
    """Tests against the actual VERG-aene.xml file."""

    AENEID_PATH = Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml").exists(),
        reason="VERG-aene.xml not found",
    )
    def test_parse_aeneid(self):
        result = parse_xml(self.AENEID_PATH)
        assert len(result.examples) > 9000  # Aeneid has ~9896 lines
        assert result.skipped < result.total * 0.05  # <5% skipped

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml").exists(),
        reason="VERG-aene.xml not found",
    )
    def test_all_have_six_feet(self):
        result = parse_xml(self.AENEID_PATH)
        for ex in result.examples[:100]:  # spot check first 100
            assert len(ex.gold_parse.foot_types) == 6
            assert ex.gold_parse.foot_types[-1] == FootType.FINAL
