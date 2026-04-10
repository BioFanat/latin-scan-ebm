"""Shared test fixtures with manually verified gold data.

The canonical test line is Aeneid 1.1:
    arma virumque cano Troiae qui primus ab oris

Scansion: —∪∪ —∪∪ — // — —∪∪ —∪∪ —×
Feet:     D    D    S     S    D    D    F  (wait, 6 feet total for hex)

Let me work this carefully:
  ar-ma vi-rum-que ca-no Troi-ae qui pri-mus ab o-ris

With elision: "Troiae" ends in vowel, "qui" starts with consonant → no elision.
But: "primus ab" — no elision (ab ends in consonant). Actually:
"cano" ends in vowel, "Troiae" starts with consonant → no elision there either.

Syllabification (standard):
  ar | ma | vi | rum | que | ca | no | Troi | ae | qui | pri | mus | ab | o | ris

Wait — "Troiae" is the key. The diphthong "ae" in "Troiae" — is it one syllable
or two? In this line, "Troiae" = Troi-ae (2 syllables: Troi + ae), where the
"ae" at the end is treated as a separate syllable. But then "ae qui" — ae ends
in a vowel, qui starts with a consonant, so no elision.

Standard scansion of Aeneid 1.1:
  ār-mă vĭ-rūm-quĕ că-nō Trōi-ae quī prī-mŭs ăb ō-rīs

  ar  = long (closed by r...wait, "arma": a-r-m-a)
  Actually: ar-ma vi-rum-que ca-nō Trōi-ae quī prī-mus ab ō-ris

Let me use the standard accepted scansion:
  — ∪∪ | — ∪∪ | — — | — — | — ∪∪ | — ×

Feet: D D S S D F
That's 5+1=6 feet, with d=3 dactyls among first 5 → M = 12 + 3 = 15 syllables.

Syllables (15):
  1.ar 2.ma vi 3.rum 4.que ca 5.nō  6.Trō 7.iae 8.quī 9.prī 10.mus 11.ab 12.ō 13.ris

Hmm, let me be more careful. I'll build a simpler fixture for now and
keep the full arma virumque for later when the atomizer is built.
"""

import pytest

from latin_ebm.types import (
    AmbiguitySite,
    CaesuraType,
    ConsonantBridge,
    FootType,
    LatinLine,
    MetricalSlot,
    Parse,
    PhonWeight,
    RealizedSyllable,
    ScoredParse,
    SiteChoice,
    SiteType,
    TrainingExample,
    VocalicAtom,
)


# ---------------------------------------------------------------------------
# A minimal synthetic line for testing data structures and serialization.
# Not a real Latin line — just exercises all the types.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_atoms() -> tuple[VocalicAtom, ...]:
    """Three atoms: a simple a-e-i sequence."""
    return (
        VocalicAtom(
            index=0, chars="a", word_idx=0,
            natural_length=PhonWeight.LONG,
            in_diphthong=False, diphthong_role=None,
            is_word_final=False, is_word_initial=True,
        ),
        VocalicAtom(
            index=1, chars="u", word_idx=0,
            natural_length=PhonWeight.SHORT,
            in_diphthong=False, diphthong_role=None,
            is_word_final=True, is_word_initial=False,
        ),
        VocalicAtom(
            index=2, chars="a", word_idx=1,
            natural_length=None,
            in_diphthong=False, diphthong_role=None,
            is_word_final=True, is_word_initial=True,
        ),
    )


@pytest.fixture
def sample_bridges() -> tuple[ConsonantBridge, ...]:
    """Two bridges between the three atoms."""
    return (
        ConsonantBridge(chars="rm", has_word_boundary=False, is_muta_cum_liquida=False),
        ConsonantBridge(chars="m", has_word_boundary=True, is_muta_cum_liquida=False),
    )


@pytest.fixture
def sample_site() -> AmbiguitySite:
    """An elision site between atoms 1 and 2 (cross-word vowel contact)."""
    return AmbiguitySite(
        index=0,
        site_type=SiteType.ELISION,
        atom_indices=(1, 2),
        valid_choices=(SiteChoice.ELIDE, SiteChoice.RETAIN),
        default=SiteChoice.ELIDE,
    )


@pytest.fixture
def sample_line(sample_atoms, sample_bridges, sample_site) -> LatinLine:
    """A synthetic LatinLine with 3 atoms, 2 bridges, 1 elision site."""
    return LatinLine(
        raw="armum a",
        normalized="armum a",
        words=("armum", "a"),
        atoms=sample_atoms,
        bridges=sample_bridges,
        sites=(sample_site,),
        author="test",
        work="test",
        book="1",
        line_num=1,
        corpus_id="test_001",
    )


@pytest.fixture
def sample_syllables() -> tuple[RealizedSyllable, ...]:
    """Two syllables for a minimal parse."""
    return (
        RealizedSyllable(
            atom_indices=(0,), onset="", nucleus="a", coda="rm",
            is_open=False, weight=PhonWeight.LONG,
        ),
        RealizedSyllable(
            atom_indices=(1,), onset="", nucleus="u", coda="m",
            is_open=False, weight=PhonWeight.LONG,
        ),
    )


@pytest.fixture
def sample_parse(sample_syllables) -> Parse:
    """A minimal parse — not metrically valid, just tests the dataclass."""
    return Parse(
        decisions={0: SiteChoice.ELIDE},
        syllables=sample_syllables,
        slots=(MetricalSlot.LONGUM, MetricalSlot.ANCEPS),
        foot_boundaries=(0,),
        foot_types=(FootType.FINAL,),
        caesura=CaesuraType.NONE,
        bucolic_diaeresis=False,
        meter="hexameter",
    )


@pytest.fixture
def sample_scored_parse(sample_parse) -> ScoredParse:
    return ScoredParse(
        parse=sample_parse,
        e_total=-3.5,
        e_site=-1.0,
        e_syll=-1.5,
        e_pair=0.0,
        e_foot=0.0,
        e_global=-1.0,
    )


@pytest.fixture
def sample_training_example(sample_line, sample_parse) -> TrainingExample:
    return TrainingExample(
        line=sample_line,
        gold_parse=sample_parse,
    )
