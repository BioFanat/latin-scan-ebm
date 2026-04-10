"""Tests for text normalization."""

from latin_ebm.normalize import normalize


class TestBasicNormalization:
    def test_lowercase(self):
        assert normalize("Arma") == "arma"

    def test_strip_comma(self):
        assert normalize("cano,") == "cano"

    def test_strip_period(self):
        assert normalize("Romae.") == "romae"

    def test_strip_semicolon(self):
        assert normalize("alto;") == "alto"

    def test_preserve_spaces(self):
        assert normalize("arma virumque cano") == "arma virumque cano"

    def test_collapse_multiple_spaces(self):
        assert normalize("arma   virumque  cano") == "arma virumque cano"

    def test_strip_leading_trailing(self):
        assert normalize("  arma virumque  ") == "arma virumque"

    def test_full_line(self):
        assert normalize("Arma virumque cano, Troiae qui primus ab oris") == (
            "arma virumque cano troiae qui primus ab oris"
        )


class TestUnicodeHandling:
    def test_nfc_normalization(self):
        # Combining macron: a + \u0304 → should strip the macron
        result = normalize("a\u0304rma")
        assert result == "arma"

    def test_precomposed_macron(self):
        assert normalize("ārmă") == "arma"

    def test_precomposed_breve(self):
        assert normalize("ĕt") == "et"

    def test_diaeresis_stripped(self):
        # aë → ae (combining diaeresis removed)
        assert normalize("ae\u0308ris") == "aeris"

    def test_mixed_diacritics(self):
        assert normalize("Trōiae") == "troiae"


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_punctuation(self):
        assert normalize(",.;:!?") == ""

    def test_parentheses(self):
        assert normalize("(arma)") == "arma"

    def test_quotes(self):
        assert normalize('"arma"') == "arma"

    def test_numbers_preserved(self):
        # Numbers in line references should be kept (they're \w)
        assert normalize("line 42") == "line 42"

    def test_hyphen_stripped(self):
        assert normalize("arma-virumque") == "armavirumque"
