"""Tests for vocalic atomization."""

from latin_ebm.atomize import atomize, _is_consonantal_i, _tokenize_word, _has_mcl
from latin_ebm.types import SiteType


class TestTokenization:
    def test_simple_word(self):
        units = _tokenize_word("arma")
        assert units == [("a", True), ("r", False), ("m", False), ("a", True)]

    def test_digraph_qu(self):
        units = _tokenize_word("qui")
        # qu is a single consonantal unit, i is a vowel
        assert units == [("qu", False), ("i", True)]

    def test_digraph_ch(self):
        units = _tokenize_word("chorus")
        assert units[0] == ("ch", False)

    def test_digraph_ph(self):
        units = _tokenize_word("pharetra")
        assert units[0] == ("ph", False)

    def test_consonantal_i_initial(self):
        # Word-initial i before vowel → consonantal
        assert _is_consonantal_i("iam", 0) is True

    def test_vocalic_i_initial(self):
        # Word-initial i NOT before vowel → vocalic
        assert _is_consonantal_i("in", 0) is False

    def test_consonantal_i_intervocalic(self):
        # Intervocalic i → consonantal
        assert _is_consonantal_i("troiae", 3) is True  # tro-i-a-e: i between o and a

    def test_vocalic_i_not_before_vowel(self):
        assert _is_consonantal_i("primus", 2) is False  # i in pri-mus


class TestMCL:
    def test_stop_liquid(self):
        assert _has_mcl("tr") is True
        assert _has_mcl("pl") is True
        assert _has_mcl("cr") is True
        assert _has_mcl("br") is True
        assert _has_mcl("gl") is True
        assert _has_mcl("dr") is True

    def test_not_mcl(self):
        assert _has_mcl("rm") is False
        assert _has_mcl("st") is False
        assert _has_mcl("nt") is False
        assert _has_mcl("") is False

    def test_mcl_in_cluster(self):
        assert _has_mcl("str") is True  # s + t + r: t+r is MCL


class TestAtomizeSimple:
    def test_arma(self):
        line = atomize("arma")
        assert len(line.atoms) == 2
        assert line.atoms[0].chars == "a"
        assert line.atoms[1].chars == "a"
        assert len(line.bridges) == 1
        assert line.bridges[0].chars == "rm"
        assert line.bridges[0].is_muta_cum_liquida is False
        assert line.bridges[0].has_word_boundary is False
        assert len(line.sites) == 0

    def test_qui(self):
        line = atomize("qui")
        # qu is consonantal, i is the only vowel
        assert len(line.atoms) == 1
        assert line.atoms[0].chars == "i"

    def test_patris_mcl(self):
        """Stop+liquid cluster (t+r) should be detected."""
        line = atomize("patris")
        assert len(line.atoms) == 2  # a, i
        assert len(line.bridges) == 1
        bridge = line.bridges[0]
        assert bridge.is_muta_cum_liquida is True
        # Should create an MCL ambiguity site
        mcl_sites = [s for s in line.sites if s.site_type == SiteType.MUTA_CUM_LIQUIDA]
        assert len(mcl_sites) == 1


class TestDiphthongs:
    def test_troiae_diphthong(self):
        """'troiae' has a consonantal i (between o and a), then ae diphthong."""
        line = atomize("troiae")
        # t-r-o-i(cons)-a-e → atoms: o, a, e
        # The 'i' between o and a is consonantal (intervocalic)
        # ae is a canonical diphthong
        atoms = line.atoms
        # Find the ae diphthong
        ae_atoms = [(a.chars, a.in_diphthong, a.diphthong_role) for a in atoms if a.in_diphthong]
        assert len(ae_atoms) == 2
        assert ae_atoms[0] == ("a", True, "first")
        assert ae_atoms[1] == ("e", True, "second")

    def test_saeuae_diphthong(self):
        """'saeuae' has ae diphthong twice: s-ae-u-ae."""
        line = atomize("saeuae")
        # s-a-e-u-a-e: ae at positions (a,e) and (a,e)
        diphthong_atoms = [a for a in line.atoms if a.in_diphthong]
        assert len(diphthong_atoms) == 4  # two pairs

    def test_diphthong_creates_split_site(self):
        """Each canonical diphthong should create a DIPHTHONG_SPLIT site."""
        line = atomize("troiae")
        split_sites = [s for s in line.sites if s.site_type == SiteType.DIPHTHONG_SPLIT]
        assert len(split_sites) >= 1


class TestElision:
    def test_vowel_before_vowel(self):
        """'multum ille' — m-ending before vowel → elision eligible."""
        line = atomize("multum ille")
        elision_sites = [s for s in line.sites if s.site_type == SiteType.ELISION]
        assert len(elision_sites) == 1

    def test_no_elision_consonant_ending(self):
        """'ab oris' — 'ab' ends in consonant, not eligible for elision."""
        line = atomize("ab oris")
        # ab ends in 'b' (not a vowel or vowel+m), so no elision
        elision_sites = [s for s in line.sites if s.site_type == SiteType.ELISION]
        assert len(elision_sites) == 0

    def test_vowel_ending_before_vowel(self):
        """'cano oris' — vowel-final before vowel-initial → elision."""
        line = atomize("cano oris")
        elision_sites = [s for s in line.sites if s.site_type == SiteType.ELISION]
        assert len(elision_sites) == 1

    def test_h_initial_allows_elision(self):
        """'cano habeo' — h+vowel initial still allows elision."""
        line = atomize("cano habeo")
        elision_sites = [s for s in line.sites if s.site_type == SiteType.ELISION]
        assert len(elision_sites) >= 1


class TestSynizesis:
    def test_adjacent_vowels_not_diphthong(self):
        """Adjacent vowels that are NOT a canonical diphthong → synizesis site."""
        # 'deus': d-e-u-s. 'eu' IS a canonical diphthong actually.
        # Let's use 'meo': m-e-o. 'eo' is NOT a canonical diphthong.
        line = atomize("meo")
        syn_sites = [s for s in line.sites if s.site_type == SiteType.SYNIZESIS]
        assert len(syn_sites) == 1


class TestFullLine:
    def test_aeneid_1_1(self):
        """Full line: arma uirumque cano troiae qui primus ab oris.

        Working through by hand:
        arma: a, [rm], a
        uirumque: u, [r], u, [mqu], e  (initial 'u' is vowel since not before vowel...
                   actually 'uirumque': u-i-r-u-m-qu-e. 'i' after 'u' — is 'u' consonantal?
                   In MQDQ, 'u' = 'v' word-initially before vowel. But our heuristic
                   is only for 'i', not 'u'. So 'u' stays vocalic.)
        """
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        # Just verify basic structural invariants
        assert len(line.atoms) > 0
        assert len(line.bridges) == len(line.atoms) - 1
        assert len(line.words) == 8
        assert line.normalized == "arma uirumque cano troiae qui primus ab oris"

    def test_atom_word_idx_consistency(self):
        """Every atom's word_idx should index a valid word."""
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        for atom in line.atoms:
            assert 0 <= atom.word_idx < len(line.words)

    def test_bridge_count(self):
        """Number of bridges = number of atoms - 1."""
        line = atomize("arma uirumque cano troiae qui primus ab oris")
        assert len(line.bridges) == len(line.atoms) - 1

    def test_word_boundary_bridges(self):
        """There should be word-boundary bridges between words."""
        line = atomize("arma uirumque cano")
        wb_bridges = [b for b in line.bridges if b.has_word_boundary]
        # Between arma and uirumque, between uirumque and cano
        assert len(wb_bridges) >= 2
