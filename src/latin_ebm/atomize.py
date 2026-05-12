"""Atomization: normalized text → vocalic atoms, consonant bridges, ambiguity sites.

Converts a normalized Latin line into the conservative pre-syllabic
representation. This is the core front-end that feeds the candidate
enumeration pipeline.

The representation is finer than syllabification: canonical diphthongs
like *ae* are two atoms linked by a default-merge relation. A realized
nucleus is one atom, or a merge/split of adjacent atoms.
"""

from __future__ import annotations

from __future__ import annotations

from typing import TYPE_CHECKING

from latin_ebm.normalize import normalize
from latin_ebm.types import (
    AmbiguitySite,
    ConsonantBridge,
    LatinLine,
    PhonWeight,
    SiteChoice,
    SiteType,
    VocalicAtom,
)

if TYPE_CHECKING:
    from latin_ebm.lexicon import VowelLengthLexicon


# ---------------------------------------------------------------------------
# Character classification
# ---------------------------------------------------------------------------

VOWELS = frozenset("aeiouy")

# Canonical Latin diphthongs (ordered pairs)
DIPHTHONGS = frozenset({"ae", "oe", "au", "eu", "ei", "ui"})

# Stop consonants (muta)
STOPS = frozenset("bcdgpt")

# Liquid consonants (liquida)
LIQUIDS = frozenset("lr")

# Digraphs that count as a single consonant
DIGRAPHS = ("ch", "ph", "th", "rh", "qu")


def _is_vowel(ch: str) -> bool:
    return ch in VOWELS


def _is_consonantal_i(word: str, pos: int) -> bool:
    """Heuristic: is 'i' at position `pos` in `word` consonantal (= j)?

    Rules (deterministic, v1):
    - Word-initial 'i' before a vowel → consonantal
    - Intervocalic 'i' (vowel before, vowel after) → consonantal
    """
    if word[pos] != "i":
        return False

    has_vowel_after = pos + 1 < len(word) and _is_vowel(word[pos + 1])
    if not has_vowel_after:
        return False

    # Word-initial i before vowel
    if pos == 0:
        return True

    # Intervocalic: vowel before and vowel after
    has_vowel_before = pos > 0 and _is_vowel(word[pos - 1])
    return has_vowel_before


def _is_consonantal_u(
    word: str,
    pos: int,
    lexicon: VowelLengthLexicon | None = None,
) -> bool:
    """Heuristic+dictionary: is 'u' at position `pos` in `word` consonantal (= v)?

    Hard rules (no dictionary needed):
    - After q before a vowel → consonantal (handled by 'qu' digraph elsewhere)
    - After g before a vowel → consonantal (e.g., "lingua")
    - Word-initial u before a vowel → consonantal (e.g., "uirumque")

    Soft rule (requires lexicon):
    - Intervocalic u before vowel → consult Morpheus. If Morpheus stores
      this word's form with 'v' at this position, treat as consonantal.
      Otherwise treat as vocalic. Disambiguates "uenit/lavinia" (v) from
      "suus/deus" (u).
    """
    if word[pos] != "u":
        return False

    if pos > 0 and word[pos - 1] == "q":
        return False

    has_vowel_after = pos + 1 < len(word) and _is_vowel(word[pos + 1])
    if not has_vowel_after:
        return False

    if pos > 0 and word[pos - 1] == "g":
        return True

    if pos == 0:
        return True

    # Intervocalic u: do NOT hard-classify here. Even when Morpheus
    # consistently stores 'v', Pedecerto's gold sometimes scans the
    # syllable with a vocalic u (e.g., "nouem" 3 syllables in Aen 1.245).
    # The atomizer keeps intervocalic u as vocalic; adjacent-vowel
    # SYNIZESIS sites and DIPHTHONG_SPLIT sites give enumerate the
    # freedom to explore both readings without losing the gold.
    return False


# ---------------------------------------------------------------------------
# Enclitic stripping
# ---------------------------------------------------------------------------

# Enclitics that are stripped before tokenization.
# The enclitic itself is processed as a separate mini-word.
_ENCLITICS_3 = ("que",)   # 3-letter enclitics
_ENCLITICS_2 = ("ne", "ue", "ve")  # 2-letter enclitics

# Words ending in -que/-ne/-ve that are NOT stem+enclitic.
# These are indivisible words where the ending is part of the stem.
_NO_STRIP = frozenset({
    # -que words (conjunctions, adverbs, etc.)
    "neque", "atque", "quoque", "usque", "undique", "ubique",
    "cumque", "namque", "absque", "denique", "itaque", "utrique",
    "plerumque", "plerique", "quisque", "quandoque", "quacumque",
    "quicumque", "quodcumque", "ubicumque", "qualiscumque",
    "quantumcumque", "quotcumque", "utcumque", "sicque",
    # -ne words
    "omne", "bene", "sane",
    # -ve words
    # (most -ve words are genuinely stem+enclitic)
})


def _strip_enclitic(
    word: str,
    lexicon: VowelLengthLexicon | None = None,
) -> tuple[str, str | None]:
    """Strip a Latin enclitic suffix from a word.

    Returns (stem, enclitic) or (word, None) if no enclitic found.

    Strategy (when lexicon provided): only split if the whole word is
    NOT in any dictionary and the stripped stem IS. This covers
    arbitrary -que/-ne/-ve compounds (rare/late-Latin) without needing
    a hardcoded exception list.

    Falls back to the legacy hardcoded `_NO_STRIP` list when no lexicon
    is given (lets unit tests work without dictionary data).
    """
    if not (
        (len(word) > 4 and word[-3:] in _ENCLITICS_3)
        or (len(word) > 3 and word[-2:] in _ENCLITICS_2)
    ):
        return word, None

    if lexicon is None:
        # Legacy fallback: hardcoded exception list
        if word in _NO_STRIP:
            return word, None
        if len(word) > 4 and word[-3:] in _ENCLITICS_3:
            return word[:-3], word[-3:]
        return word[:-2], word[-2:]

    # Dictionary-driven: don't split if whole word is known
    if lexicon.is_known_form(word):
        return word, None

    enc_len = 3 if word[-3:] in _ENCLITICS_3 else 2
    stem = word[:-enc_len]
    enc = word[-enc_len:]

    # Only split if the stem is a known word — otherwise stripping is
    # likely wrong (treat as opaque word and let realize/enumerate cope).
    if lexicon.is_known_form(stem):
        return stem, enc
    return word, None


# ---------------------------------------------------------------------------
# Tokenization: word → list of (unit, is_vowel) pairs
# ---------------------------------------------------------------------------


def _tokenize_word(
    word: str,
    lexicon: VowelLengthLexicon | None = None,
) -> list[tuple[str, bool]]:
    """Break a word into phonological units: (char_or_digraph, is_vowel).

    Handles digraphs (ch, ph, th, rh, qu), consonantal i, and
    treats x/z as single units (their double-consonant effect is
    tracked separately). `lexicon` enables dictionary-driven
    intervocalic consonantal-u detection.

    Pedecerto orthography note: a leading 'v' followed by consonant(s)
    represents VOCALIC u (e.g., "Vrbs" = urbs, 1 syllable). This is
    handled here by checking the word has no other vowels.
    """
    # Pedecerto orthography: word-initial 'V' before a consonant is the
    # vocalic u (e.g., "Vrbs"=urbs, "Vnde"=unde). Word-initial 'V' before
    # a vowel is the consonantal v (e.g., "Vir"=vir).
    word_initial_v_is_vowel = (
        len(word) > 1
        and word[0] == "v"
        and not _is_vowel(word[1])
    )

    units: list[tuple[str, bool]] = []
    i = 0
    while i < len(word):
        ch = word[i]

        digraph_found = False
        if i + 1 < len(word):
            pair = word[i : i + 2]
            if pair in DIGRAPHS:
                units.append((pair, False))
                i += 2
                digraph_found = True

        if digraph_found:
            continue

        if _is_consonantal_i(word, i):
            units.append((ch, False))
            i += 1
            continue

        if _is_consonantal_u(word, i, lexicon):
            units.append((ch, False))
            i += 1
            continue

        # Word-initial 'v' before consonant — vocalic (Pedecerto convention)
        if i == 0 and word_initial_v_is_vowel:
            units.append(("u", True))  # treat as vocalic u
            i += 1
            continue

        units.append((ch, _is_vowel(ch)))
        i += 1

    return units


# ---------------------------------------------------------------------------
# Atom and bridge extraction
# ---------------------------------------------------------------------------



def _has_mcl(consonants: str) -> bool:
    """Check if a consonant string contains a muta cum liquida cluster."""
    for i in range(len(consonants) - 1):
        if consonants[i] in STOPS and consonants[i + 1] in LIQUIDS:
            return True
    return False


# ---------------------------------------------------------------------------
# Clean implementation: two-pass approach
# ---------------------------------------------------------------------------


def _build_line_units(
    words: tuple[str, ...],
    lexicon: VowelLengthLexicon | None = None,
) -> list[tuple[str, bool, int, bool]]:
    """Build a flat list of (unit, is_vowel, word_idx, is_word_start) for the line.

    Word boundaries are tracked so bridges can be annotated.
    """
    line_units: list[tuple[str, bool, int, bool]] = []
    for word_idx, word in enumerate(words):
        units = _tokenize_word(word, lexicon=lexicon)
        for i, (unit, is_v) in enumerate(units):
            is_word_start = (i == 0)
            line_units.append((unit, is_v, word_idx, is_word_start))
    return line_units


def atomize(raw: str, lexicon: VowelLengthLexicon | None = None) -> LatinLine:
    """Convert raw Latin text into a pre-syllabic LatinLine representation.

    The pipeline:
    1. Normalize text
    2. Tokenize into phonological units per word
    3. Build flat unit list across the line
    4. Extract vocalic atoms (vowel units) and consonant bridges (between atoms)
    5. Detect ambiguity sites (elision, diphthong split, synizesis, MCL)
    6. If lexicon provided, populate natural_length on atoms
    """
    normalized = normalize(raw)
    raw_words = normalized.split()

    # Expand enclitics: "uirumque" → "uirum", "que"
    expanded: list[str] = []
    for w in raw_words:
        stem, enclitic = _strip_enclitic(w, lexicon=lexicon)
        expanded.append(stem)
        if enclitic is not None:
            expanded.append(enclitic)
    words = tuple(expanded)

    if not words:
        return LatinLine(
            raw=raw, normalized=normalized, words=words,
            atoms=(), bridges=(), sites=(),
        )

    # Build flat unit list — pass lexicon for dictionary-driven consonantal-u
    line_units = _build_line_units(words, lexicon=lexicon)

    # Pass 1: identify vowel positions and extract atoms
    atoms: list[VocalicAtom] = []
    vowel_unit_indices: list[int] = []  # indices into line_units

    for i, (unit, is_v, word_idx, _) in enumerate(line_units):
        if is_v:
            vowel_unit_indices.append(i)

    # For each word, find first/last vowel
    word_first_vowel: dict[int, int] = {}  # word_idx → atom index
    word_last_vowel: dict[int, int] = {}

    for atom_idx, unit_idx in enumerate(vowel_unit_indices):
        _, _, word_idx, _ = line_units[unit_idx]
        if word_idx not in word_first_vowel:
            word_first_vowel[word_idx] = atom_idx
        word_last_vowel[word_idx] = atom_idx

    # Build atoms with greedy left-to-right diphthong detection.
    # A vowel claimed as "second" of a diphthong can't be "first" of another.
    diphthong_info: list[tuple[bool, str | None]] = [(False, None)] * len(vowel_unit_indices)

    for atom_idx in range(len(vowel_unit_indices) - 1):
        # Skip if this atom is already claimed as "second"
        if diphthong_info[atom_idx][1] == "second":
            continue

        unit_idx = vowel_unit_indices[atom_idx]
        next_unit_idx = vowel_unit_indices[atom_idx + 1]
        unit, _, word_idx, _ = line_units[unit_idx]
        next_unit, _, next_word_idx, _ = line_units[next_unit_idx]

        # Same word and adjacent (no consonants between)
        if next_word_idx == word_idx and next_unit_idx == unit_idx + 1:
            pair = unit + next_unit
            if pair in DIPHTHONGS:
                diphthong_info[atom_idx] = (True, "first")
                diphthong_info[atom_idx + 1] = (True, "second")

    # If lexicon provided, compute natural_length per word using character-aligned lookup.
    natural_lengths: list[PhonWeight | None] = [None] * len(vowel_unit_indices)
    if lexicon is not None:
        by_word: dict[int, list[int]] = {}
        for atom_idx, unit_idx in enumerate(vowel_unit_indices):
            _, _, w_idx, _ = line_units[unit_idx]
            by_word.setdefault(w_idx, []).append(atom_idx)
        for w_idx, idx_list in by_word.items():
            atom_chars = [line_units[vowel_unit_indices[i]][0] for i in idx_list]
            lengths = lexicon.lookup_aligned(words[w_idx], atom_chars)
            for slot, length in zip(idx_list, lengths):
                natural_lengths[slot] = length

    for atom_idx, unit_idx in enumerate(vowel_unit_indices):
        unit, _, word_idx, _ = line_units[unit_idx]
        in_diphthong, diphthong_role = diphthong_info[atom_idx]

        atoms.append(VocalicAtom(
            index=atom_idx,
            chars=unit,
            word_idx=word_idx,
            natural_length=natural_lengths[atom_idx],
            in_diphthong=in_diphthong,
            diphthong_role=diphthong_role,
            is_word_final=(atom_idx == word_last_vowel.get(word_idx, -1)),
            is_word_initial=(atom_idx == word_first_vowel.get(word_idx, -1)),
        ))

    # Pass 2: build bridges between consecutive atoms
    bridges: list[ConsonantBridge] = []
    for i in range(len(atoms) - 1):
        # Consonant material between vowel_unit_indices[i] and vowel_unit_indices[i+1]
        start = vowel_unit_indices[i] + 1
        end = vowel_unit_indices[i + 1]

        cons_units: list[str] = []
        has_word_boundary = False
        word_boundary_char_pos = 0
        char_cursor = 0

        for j in range(start, end):
            unit, _, _, is_word_start = line_units[j]
            if is_word_start and not has_word_boundary:
                # First word-boundary marker in this bridge — record split point
                has_word_boundary = True
                word_boundary_char_pos = char_cursor
            cons_units.append(unit)
            char_cursor += len(unit)

        # Also check if the atom after the bridge starts a new word
        _, _, next_word_idx, next_is_word_start = line_units[vowel_unit_indices[i + 1]]
        if next_is_word_start and atoms[i].word_idx != next_word_idx:
            if not has_word_boundary:
                # All bridge consonants belong to the LEFT word (vowel-initial next word)
                has_word_boundary = True
                word_boundary_char_pos = char_cursor

        cons_str = "".join(cons_units)
        bridges.append(ConsonantBridge(
            chars=cons_str,
            has_word_boundary=has_word_boundary,
            is_muta_cum_liquida=_has_mcl(cons_str),
            word_boundary_pos=word_boundary_char_pos if has_word_boundary else 0,
        ))

    # Pass 3: detect ambiguity sites
    sites = _detect_sites(atoms, bridges, words)

    return LatinLine(
        raw=raw, normalized=normalized, words=words,
        atoms=tuple(atoms), bridges=tuple(bridges), sites=tuple(sites),
    )


# ---------------------------------------------------------------------------
# Ambiguity site detection
# ---------------------------------------------------------------------------


def _detect_sites(
    atoms: list[VocalicAtom],
    bridges: list[ConsonantBridge],
    words: tuple[str, ...],
) -> list[AmbiguitySite]:
    """Detect all prosodic ambiguity sites in the atom/bridge sequence."""
    sites: list[AmbiguitySite] = []
    site_idx = 0

    for i, bridge in enumerate(bridges):
        left_atom = atoms[i]
        right_atom = atoms[i + 1]

        # --- Elision / prodelision ---
        if bridge.has_word_boundary and left_atom.word_idx != right_atom.word_idx:
            if _is_elision_eligible(left_atom, bridge, right_atom, words):
                # Check if this is prodelision (following word is es/est)
                right_word = words[right_atom.word_idx]
                if right_word in ("es", "est") and right_atom.is_word_initial:
                    sites.append(AmbiguitySite(
                        index=site_idx,
                        site_type=SiteType.PRODELISION,
                        atom_indices=(i, i + 1),
                        valid_choices=(SiteChoice.ELIDE, SiteChoice.ELIDE_RIGHT, SiteChoice.RETAIN),
                        default=SiteChoice.ELIDE,
                    ))
                else:
                    sites.append(AmbiguitySite(
                        index=site_idx,
                        site_type=SiteType.ELISION,
                        atom_indices=(i, i + 1),
                        valid_choices=(SiteChoice.ELIDE, SiteChoice.RETAIN, SiteChoice.RETAIN_SHORT),
                        default=SiteChoice.ELIDE,
                    ))
                site_idx += 1

        # --- Diphthong split ---
        if (left_atom.in_diphthong and left_atom.diphthong_role == "first"
                and right_atom.in_diphthong and right_atom.diphthong_role == "second"
                and left_atom.word_idx == right_atom.word_idx
                and bridge.chars == ""):
            sites.append(AmbiguitySite(
                index=site_idx,
                site_type=SiteType.DIPHTHONG_SPLIT,
                atom_indices=(i, i + 1),
                valid_choices=(SiteChoice.DEFAULT, SiteChoice.SPLIT),
                default=SiteChoice.DEFAULT,  # default: keep as diphthong
            ))
            site_idx += 1

        # --- Synizesis ---
        # Two adjacent vowels in the same word, NOT a canonical diphthong,
        # with no consonants between them
        if (left_atom.word_idx == right_atom.word_idx
                and bridge.chars == ""
                and not (left_atom.in_diphthong and right_atom.in_diphthong
                         and left_atom.diphthong_role == "first"
                         and right_atom.diphthong_role == "second")):
            sites.append(AmbiguitySite(
                index=site_idx,
                site_type=SiteType.SYNIZESIS,
                atom_indices=(i, i + 1),
                valid_choices=(SiteChoice.DEFAULT, SiteChoice.MERGE),
                default=SiteChoice.DEFAULT,  # default: keep separate
            ))
            site_idx += 1

        # --- Muta cum liquida ---
        if bridge.is_muta_cum_liquida and not bridge.has_word_boundary:
            sites.append(AmbiguitySite(
                index=site_idx,
                site_type=SiteType.MUTA_CUM_LIQUIDA,
                atom_indices=(i, i + 1),
                valid_choices=(SiteChoice.ONSET, SiteChoice.CLOSE),
                default=SiteChoice.ONSET,  # anceps parity: muta+liquida is short by default
            ))
            site_idx += 1

    return sites


def _is_elision_eligible(
    left_atom: VocalicAtom,
    bridge: ConsonantBridge,
    right_atom: VocalicAtom,
    words: tuple[str, ...],
) -> bool:
    """Check if a cross-word contact is eligible for elision.

    Elision is possible when:
    - The left word ends in a vowel, or ends in vowel + 'm'
    - The right word begins with a vowel (or h + vowel)
    - The left atom is the final vowel of its word
    """
    if not left_atom.is_word_final:
        return False
    if not right_atom.is_word_initial:
        return False

    left_word = words[left_atom.word_idx]
    right_word = words[right_atom.word_idx]

    # Left word must end in vowel or vowel+m
    left_ok = (
        left_word[-1] in VOWELS
        or (len(left_word) >= 2 and left_word[-1] == "m" and left_word[-2] in VOWELS)
    )

    # Right word must begin with vowel or h+vowel
    right_ok = (
        right_word[0] in VOWELS
        or (len(right_word) >= 2 and right_word[0] == "h" and right_word[1] in VOWELS)
    )

    return left_ok and right_ok
