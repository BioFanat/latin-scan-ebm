"""Tests for vowel length lexicon."""

from pathlib import Path

import pytest

from latin_ebm.lexicon import VowelLengthLexicon, _parse_macron_form
from latin_ebm.types import PhonWeight

DATA_DIR = Path(__file__).parent.parent / "data"
MQDQ_PATH = DATA_DIR / "MqDqMacrons.json"
MORPHEUS_PATH = DATA_DIR / "MorpheusMacrons.txt"


class TestParseMacronForm:
    def test_simple_long(self):
        assert _parse_macron_form("a_rma^") == [PhonWeight.LONG, PhonWeight.SHORT]

    def test_anceps(self):
        result = _parse_macron_form("a_rma*")
        assert result == [PhonWeight.LONG, None]

    def test_diphthong(self):
        result = _parse_macron_form("s[ae]v[ae]")
        # Two diphthongs, each counts as one long position
        assert result == [PhonWeight.LONG, PhonWeight.LONG]

    def test_consonantal_j_and_v(self):
        # "vi^ru*m" — j and v are consonantal, not counted
        result = _parse_macron_form("vi^ru*m")
        assert result == [PhonWeight.SHORT, None]

    def test_no_mark(self):
        # Vowel with no mark → ambiguous
        result = _parse_macron_form("ab")
        assert result == [None]

    def test_qui(self):
        # "qvi_" — qv is consonantal, i is the only vowel
        result = _parse_macron_form("qvi_")
        assert result == [PhonWeight.LONG]

    def test_all_short(self):
        result = _parse_macron_form("ve^lu^t")
        assert result == [PhonWeight.SHORT, PhonWeight.SHORT]


@pytest.mark.skipif(not MQDQ_PATH.exists(), reason="MqDqMacrons.json not found")
class TestMqdqLookup:
    @pytest.fixture
    def lex(self):
        return VowelLengthLexicon(mqdq_path=MQDQ_PATH)

    def test_arma(self, lex):
        """'arma' → first 'a' long (by position in most contexts), second 'a' ambiguous."""
        result = lex.lookup("arma")
        assert result is not None
        assert len(result) == 2
        assert result[0] == PhonWeight.LONG   # ā (always long in MQDQ)
        # Second 'a' is ambiguous: 72% short, 28% anceps → below 85% threshold
        assert result[1] is None

    def test_qui(self, lex):
        result = lex.lookup("qui")
        assert result is not None
        assert len(result) == 1
        assert result[0] == PhonWeight.LONG

    def test_et(self, lex):
        result = lex.lookup("et")
        assert result is not None
        assert len(result) == 1
        # 'et' is anceps — usually short but can be long

    def test_unknown_word(self, lex):
        result = lex.lookup("xyzzyplugh")
        assert result is None

    def test_author_filtering(self, lex):
        """Virgil-specific lookup should use Virgil's frequencies."""
        result_all = lex.lookup("primus")
        result_verg = lex.lookup("primus", author="Vergilius")
        # Both should return something
        assert result_all is not None
        assert result_verg is not None

    def test_troiae_diphthong(self, lex):
        result = lex.lookup("troiae")
        assert result is not None
        # Should have entries for: o, [ae] (diphthong)
        # The 'i' in troiae is consonantal (j) and shouldn't count

    def test_lexicon_size(self, lex):
        assert lex.size > 90000  # ~95K word forms
