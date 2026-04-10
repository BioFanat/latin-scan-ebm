"""Core types for Latin scansion EBM.

All enums and dataclasses live here. This module imports nothing from
the package and is the leaf of the dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Enums — small finite domains from the formal model
# ---------------------------------------------------------------------------


class PhonWeight(Enum):
    """Phonological syllable weight (ω).

    A syllable is LONG (heavy) if it has a long vowel, a diphthong nucleus,
    or is closed by one or more consonants. Otherwise it is SHORT (light).
    """

    LONG = auto()
    SHORT = auto()


class MetricalSlot(Enum):
    """Metrical position type (μ).

    Distinct from PhonWeight: ANCEPS is a metrical *permission*
    (either weight is acceptable), not a phonological fact.
    """

    LONGUM = auto()   # position requires a heavy syllable
    BREVE = auto()    # position requires a light syllable
    ANCEPS = auto()   # position accepts either weight (e.g. line-final)


class FootType(Enum):
    """Foot identity (τ) for dactylic meter."""

    DACTYL = auto()   # —∪∪
    SPONDEE = auto()  # ——
    FINAL = auto()    # —× (hexameter foot 6, always)


class SiteType(Enum):
    """Kind of prosodic ambiguity at a site in R(x)."""

    ELISION = auto()          # vowel(/vowel+m) before vowel across word boundary
    SYNIZESIS = auto()        # adjacent vowels within a word that could merge
    DIPHTHONG_SPLIT = auto()  # canonical diphthong that could separate into two nuclei
    PRODELISION = auto()      # deletion of initial vowel of following word (es/est)
    MUTA_CUM_LIQUIDA = auto() # stop+liquid cluster: closes preceding syllable or not


class SiteChoice(Enum):
    """Realization decision (ρ_r) at an ambiguity site.

    Not every choice is valid at every site type — the valid subset
    is stored in AmbiguitySite.valid_choices.
    """

    # Elision / hiatus sites
    ELIDE = auto()          # delete left vowel (standard elision / synaloepha)
    RETAIN = auto()         # keep both nuclei (hiatus)
    RETAIN_SHORT = auto()   # keep but allow shortening (correption in hiatus)
    ELIDE_RIGHT = auto()    # prodelision: delete right vowel instead

    # Synizesis sites
    MERGE = auto()          # two adjacent nuclei → one (synizesis)

    # Diphthong sites
    SPLIT = auto()          # one diphthong → two separate nuclei (diaeresis)

    # Muta cum liquida sites
    CLOSE = auto()          # cluster closes preceding syllable (→ heavy)
    ONSET = auto()          # cluster is onset of following syllable (preceding stays light)

    # Neutral
    DEFAULT = auto()        # no special action; phonological default applies


class CaesuraType(Enum):
    """Line-level caesura classification (part of κ)."""

    PENTHEMIMERAL = auto()   # after position 5 (most common in hexameter)
    TRIHEMIMERAL = auto()    # after position 3
    HEPHTHEMIMERAL = auto()  # after position 7
    KATA_TRITON = auto()     # after the third trochee
    NONE = auto()


# ---------------------------------------------------------------------------
# Input structures — fixed per line, determined by normalization/atomization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocalicAtom:
    """A single potential vocalic nucleus in the pre-syllabic representation.

    The atom sequence is finer than syllabification: canonical diphthongs
    like *ae* are two atoms linked by a default-merge relation. A realized
    nucleus is one atom, or a merge/split of adjacent atoms.
    """

    index: int                        # position in the atom sequence (0-based)
    chars: str                        # graphemic vowel character(s), e.g. "a", "e"
    word_idx: int                     # index of the containing word token
    natural_length: PhonWeight | None  # known lexical length, or None if ambiguous
    in_diphthong: bool                # participates in a canonical diphthong site
    diphthong_role: str | None        # "first" or "second" if in_diphthong, else None
    is_word_final: bool
    is_word_initial: bool


@dataclass(frozen=True)
class ConsonantBridge:
    """Consonantal material between two adjacent vocalic atoms.

    bridges[i] sits between atoms[i] and atoms[i+1], so
    len(bridges) == len(atoms) - 1.
    """

    chars: str                # consonant characters (may be empty for hiatus)
    has_word_boundary: bool   # a word boundary (#) falls within this bridge
    is_muta_cum_liquida: bool # a stop+liquid cluster is present


@dataclass(frozen=True)
class AmbiguitySite:
    """A location in the input where a prosodic decision must be made.

    Each site has a small finite domain of valid choices. The default
    is the phonologically expected outcome when no metrical pressure
    forces a different realization.
    """

    index: int                                # site index (for keying into Parse.decisions)
    site_type: SiteType
    atom_indices: tuple[int, ...]             # which vocalic atoms are involved
    valid_choices: tuple[SiteChoice, ...]      # D_r: legal options at this site
    default: SiteChoice                        # phonologically expected choice


@dataclass
class LatinLine:
    """Complete pre-syllabic representation of a single line of Latin verse.

    This is the INPUT to inference — determined entirely by the raw text
    and normalization rules. The atom/bridge/site sequences are structurally
    immutable after atomization (stored as tuples), but corpus metadata
    fields can be attached during ingestion.
    """

    raw: str
    normalized: str
    words: tuple[str, ...]
    atoms: tuple[VocalicAtom, ...]
    bridges: tuple[ConsonantBridge, ...]      # len == len(atoms) - 1
    sites: tuple[AmbiguitySite, ...]          # R(x): all ambiguity sites

    # Corpus metadata — attached during ingestion, not during atomization
    author: str = ""
    work: str = ""
    book: str = ""
    line_num: int = 0
    corpus_id: str = ""


# ---------------------------------------------------------------------------
# Output structures — one per candidate parse, varying over Y(x)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealizedSyllable:
    """A syllable produced by applying realization decisions to atoms.

    Part of g_x(ρ) — the realized syllable sequence induced by a
    particular decision bundle.
    """

    atom_indices: tuple[int, ...]  # which vocalic atoms form this nucleus
    onset: str                     # onset consonants
    nucleus: str                   # nucleus vowel(s)
    coda: str                      # coda consonants
    is_open: bool                  # True if coda is empty
    weight: PhonWeight             # phonological weight (ω_m)


@dataclass(frozen=True)
class Parse:
    """A complete prosodic parse: y = (ρ, ω, μ, β, τ, κ, m).

    Many Parse objects exist per line during inference; the model
    picks the one with lowest energy. All sites in the line should
    have an entry in decisions (not just non-default ones), so the
    parse is self-contained.

    Note: frozen=True prevents attribute reassignment but does not
    deep-freeze the decisions dict. Treat Parse as a value object.
    """

    # ρ — realization decisions, keyed by site index
    decisions: dict[int, SiteChoice]

    # g_x(ρ) with ω baked in
    syllables: tuple[RealizedSyllable, ...]

    # μ — metrical slot per syllable
    slots: tuple[MetricalSlot, ...]

    # β — syllable indices where each foot begins
    foot_boundaries: tuple[int, ...]

    # τ — foot identity per foot
    foot_types: tuple[FootType, ...]

    # κ — line-level annotations
    caesura: CaesuraType
    bucolic_diaeresis: bool

    # m — meter label
    meter: str


@dataclass(frozen=True)
class ScoredParse:
    """A parse together with its energy decomposition.

    Keeps the total and each component so that energy contributions
    can be inspected for debugging and interpretability.
    """

    parse: Parse
    e_total: float
    e_site: float      # Σ_r E_site(x, r, ρ_r)
    e_syll: float      # Σ_m E_syll(x, s_m, ω_m)
    e_pair: float      # Σ_m E_pair(x, s_m, s_{m+1}, y)
    e_foot: float      # Σ_f E_foot(y_{F_f})
    e_global: float    # E_global(x, y)


@dataclass
class TrainingExample:
    """A line paired with its gold-standard parse.

    The observed field tracks which components of the gold parse are
    directly attested in the corpus (vs. inferred during alignment).
    This supports partial supervision: if a corpus gives scanned
    quantities but not every internal ρ decision, training can
    marginalize over the unobserved structure.
    """

    line: LatinLine
    gold_parse: Parse
    observed: frozenset[str] = field(default_factory=lambda: frozenset({
        "decisions", "syllables", "slots", "foot_types", "caesura",
    }))
