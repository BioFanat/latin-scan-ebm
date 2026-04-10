"""Pedecerto/MQDQ XML ingestion.

Parses Pedecerto XML files into LatinLine and gold Parse objects,
using the sy/wb/mf attributes to reconstruct the metrical analysis.

XML format (confirmed from VERG-aene.xml):

    <line name="1" metre="H" pattern="DDSS">
        <word sy="1A1b" wb="CF">Arma</word>
        <word sy="1c2A2b" wb="CF">uirumque</word>
        ...
    </line>

sy encoding: 2-char pairs (foot_number, position_letter).
    A = arsis (long), T = spondee thesis (long),
    b/c = dactyl thesis (short), X = final anceps.
wb: CM = caesura masculine, CF = caesura feminine, DI = diaeresis.
mf: SY = synalepha (elision), PE = prodelision.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from latin_ebm.atomize import atomize
from latin_ebm.types import (
    CaesuraType,
    FootType,
    LatinLine,
    MetricalSlot,
    PhonWeight,
    RealizedSyllable,
    Parse,
    SiteChoice,
    SiteType,
    TrainingExample,
)

if TYPE_CHECKING:
    from latin_ebm.lexicon import VowelLengthLexicon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sy attribute decoding
# ---------------------------------------------------------------------------

# Position letter → (MetricalSlot, PhonWeight)
_POSITION_MAP: dict[str, tuple[MetricalSlot, PhonWeight | None]] = {
    "A": (MetricalSlot.LONGUM, PhonWeight.LONG),     # arsis: always long
    "T": (MetricalSlot.LONGUM, PhonWeight.LONG),     # spondee thesis: long
    "b": (MetricalSlot.BREVE, PhonWeight.SHORT),      # dactyl thesis 1: short
    "c": (MetricalSlot.BREVE, PhonWeight.SHORT),      # dactyl thesis 2: short
    "X": (MetricalSlot.ANCEPS, None),                 # final anceps
}


@dataclass
class SyllableInfo:
    """Decoded syllable from sy attribute."""
    foot: int              # foot number (1-6)
    position: str          # position letter (A, T, b, c, X)
    slot: MetricalSlot     # metrical slot type
    weight: PhonWeight | None  # phonological weight (None for anceps)


@dataclass
class WordInfo:
    """Parsed word element from XML."""
    text: str              # word text (may include punctuation)
    syllables: list[SyllableInfo]  # decoded syllable assignments
    wb: str | None         # word boundary type (CM, CF, DI, or None)
    mf: str | None         # metrical feature (SY, PE, or None)


def decode_sy(sy: str) -> list[SyllableInfo]:
    """Decode sy attribute into syllable assignments.

    '1A1b1c2A' → [(1,'A',LONGUM,LONG), (1,'b',BREVE,SHORT),
                   (1,'c',BREVE,SHORT), (2,'A',LONGUM,LONG)]
    """
    result: list[SyllableInfo] = []
    for i in range(0, len(sy), 2):
        foot = int(sy[i])
        position = sy[i + 1]
        slot, weight = _POSITION_MAP[position]
        result.append(SyllableInfo(
            foot=foot, position=position, slot=slot, weight=weight,
        ))
    return result


def _parse_word_element(word_el: ET.Element) -> WordInfo:
    """Parse a <word> XML element."""
    text = word_el.text or ""
    sy = word_el.get("sy", "")
    wb = word_el.get("wb")
    mf = word_el.get("mf")

    syllables = decode_sy(sy) if sy else []

    return WordInfo(text=text, syllables=syllables, wb=wb, mf=mf)


# ---------------------------------------------------------------------------
# Gold parse construction from decoded words
# ---------------------------------------------------------------------------


def _derive_foot_types(all_syllables: list[SyllableInfo]) -> tuple[FootType, ...]:
    """Derive foot types from the syllable assignments.

    Groups syllables by foot number, then determines foot type
    from the position letters present.
    """
    feet: dict[int, list[str]] = {}
    for syl in all_syllables:
        feet.setdefault(syl.foot, []).append(syl.position)

    foot_types: list[FootType] = []
    for foot_num in sorted(feet.keys()):
        positions = feet[foot_num]
        if "X" in positions:
            foot_types.append(FootType.FINAL)
        elif "b" in positions or "c" in positions:
            foot_types.append(FootType.DACTYL)
        else:
            # Only A and/or T → spondee
            foot_types.append(FootType.SPONDEE)

    return tuple(foot_types)


def _compute_foot_boundaries(foot_types: tuple[FootType, ...]) -> tuple[int, ...]:
    """Compute syllable indices where each foot begins."""
    boundaries: list[int] = [0]
    for ft in foot_types[:-1]:  # don't need boundary after last foot
        if ft == FootType.DACTYL:
            boundaries.append(boundaries[-1] + 3)
        else:  # SPONDEE or FINAL
            boundaries.append(boundaries[-1] + 2)
    return tuple(boundaries)


def _classify_caesura_from_wb(words: list[WordInfo]) -> CaesuraType:
    """Determine caesura type from word boundary attributes.

    CM (caesura masculine) after arsis, CF (caesura feminine) after
    first breve of dactyl. We use the first CM or CF found in the
    middle of the line as the main caesura.
    """
    for word in words:
        if word.wb == "CM":
            # Need to check which foot this is in to classify
            if word.syllables:
                last_syl = word.syllables[-1]
                if last_syl.foot == 3 and last_syl.position == "A":
                    return CaesuraType.PENTHEMIMERAL
                elif last_syl.foot == 2 and last_syl.position == "A":
                    return CaesuraType.TRIHEMIMERAL
                elif last_syl.foot == 4 and last_syl.position == "A":
                    return CaesuraType.HEPHTHEMIMERAL
        elif word.wb == "CF":
            if word.syllables:
                last_syl = word.syllables[-1]
                if last_syl.foot == 3 and last_syl.position == "b":
                    return CaesuraType.KATA_TRITON

    # Fallback: look for any CM in feet 2-4
    for word in words:
        if word.wb == "CM" and word.syllables:
            return CaesuraType.PENTHEMIMERAL  # approximate

    return CaesuraType.NONE


def _build_gold_parse(words: list[WordInfo]) -> Parse | None:
    """Build a gold Parse from decoded word info.

    Returns None if the line can't be parsed (e.g., corrupt data).
    """
    # Collect all non-elided syllables across all words
    all_syllables: list[SyllableInfo] = []
    decisions: dict[int, SiteChoice] = {}
    elision_word_indices: set[int] = set()

    for word_idx, word in enumerate(words):
        if word.mf == "SY":
            elision_word_indices.add(word_idx)
        all_syllables.extend(word.syllables)

    if not all_syllables:
        return None

    # Derive foot types from syllable data
    foot_types = _derive_foot_types(all_syllables)

    if len(foot_types) != 6:
        logger.debug("Expected 6 feet, got %d", len(foot_types))
        return None

    foot_boundaries = _compute_foot_boundaries(foot_types)
    slots = tuple(s.slot for s in all_syllables)

    # Build minimal realized syllables (we don't have full onset/coda info
    # from the XML alone — just the weight and position)
    syllables = tuple(
        RealizedSyllable(
            atom_indices=(),  # not aligned to atoms yet
            onset="",
            nucleus="",
            coda="",
            is_open=True,
            weight=s.weight if s.weight is not None else PhonWeight.LONG,
        )
        for s in all_syllables
    )

    caesura = _classify_caesura_from_wb(words)

    # Check bucolic diaeresis: DI on word ending at foot 4/5 boundary
    bucolic = False
    for word in words:
        if word.wb == "DI" and word.syllables:
            last_syl = word.syllables[-1]
            # Bucolic diaeresis: word boundary at start of foot 5
            if last_syl.foot == 4:
                bucolic = True

    return Parse(
        decisions=decisions,
        syllables=syllables,
        slots=slots,
        foot_boundaries=foot_boundaries,
        foot_types=foot_types,
        caesura=caesura,
        bucolic_diaeresis=bucolic,
        meter="hexameter",
    )


# ---------------------------------------------------------------------------
# Gold alignment: map MQDQ decisions to our atom-level representation
# ---------------------------------------------------------------------------


def align_gold_parse(
    line: LatinLine,
    words: list[WordInfo],
    gold_parse: Parse,
) -> Parse:
    """Align a gold parse to our atom-level representation.

    Infers site decisions from MQDQ markers:
    - mf="SY" → ELIDE at the corresponding elision site
    - mf="PE" → ELIDE_RIGHT at the corresponding prodelision site
    - Diphthong merge/split inferred from syllable count comparison
    - MCL and other ambiguous decisions left unresolved

    Returns a new Parse with populated decisions dict.
    """
    decisions: dict[int, SiteChoice] = {}
    inferred_sites: set[int] = set()

    # Step 1: Mark elision sites from mf="SY" markers.
    # When mf="SY" on a word, that word's final vowel is elided before
    # the next word's initial vowel. Find the matching elision site.
    for word_idx, word in enumerate(words):
        if word.mf == "SY":
            # Find elision site where left atom is in this word
            # and right atom is in the next word
            for site in line.sites:
                if site.site_type != SiteType.ELISION:
                    continue
                left_idx, right_idx = site.atom_indices
                left_atom = line.atoms[left_idx]
                right_atom = line.atoms[right_idx]
                if (left_atom.word_idx == word_idx
                        and right_atom.word_idx == word_idx + 1
                        and left_atom.is_word_final):
                    decisions[site.index] = SiteChoice.ELIDE
                    inferred_sites.add(site.index)
                    break

        elif word.mf == "PE":
            # Prodelision: right word's initial vowel is deleted
            for site in line.sites:
                if site.site_type != SiteType.PRODELISION:
                    continue
                left_idx, right_idx = site.atom_indices
                left_atom = line.atoms[left_idx]
                right_atom = line.atoms[right_idx]
                if (left_atom.word_idx == word_idx
                        and right_atom.word_idx == word_idx + 1):
                    decisions[site.index] = SiteChoice.ELIDE_RIGHT
                    inferred_sites.add(site.index)
                    break

    # Step 2: Infer diphthong merge/split from syllable counts.
    # For each word, compare MQDQ syllable count to our atom count.
    # If the word has a diphthong site and MQDQ has fewer syllables
    # than atoms → diphthong was merged (DEFAULT).
    # If equal → diphthong was split (SPLIT).
    for word_idx, word in enumerate(words):
        if word.mf in ("SY", "PE"):
            continue  # elided words don't contribute normally

        mqdq_syl_count = len(word.syllables)

        # Count atoms in this word
        word_atoms = [a for a in line.atoms if a.word_idx == word_idx]
        our_atom_count = len(word_atoms)

        # Find diphthong sites in this word
        for site in line.sites:
            if site.index in inferred_sites:
                continue
            if site.site_type != SiteType.DIPHTHONG_SPLIT:
                continue
            left_idx, right_idx = site.atom_indices
            if line.atoms[left_idx].word_idx != word_idx:
                continue

            # If MQDQ has fewer syllables than atoms → diphthong merged
            # A merged diphthong reduces atom count by 1
            if mqdq_syl_count < our_atom_count:
                decisions[site.index] = SiteChoice.DEFAULT  # merge
                inferred_sites.add(site.index)
                our_atom_count -= 1  # account for the merge
            else:
                decisions[site.index] = SiteChoice.SPLIT  # keep separate
                inferred_sites.add(site.index)

    # Step 3: For unresolved sites, use defaults.
    # Elision sites not marked with mf="SY" → RETAIN (no elision)
    for site in line.sites:
        if site.index in inferred_sites:
            continue
        if site.site_type == SiteType.ELISION:
            decisions[site.index] = SiteChoice.RETAIN
            inferred_sites.add(site.index)
        elif site.site_type == SiteType.PRODELISION:
            decisions[site.index] = SiteChoice.RETAIN
            inferred_sites.add(site.index)
        elif site.site_type == SiteType.SYNIZESIS:
            # Default: no merge (keep separate)
            decisions[site.index] = SiteChoice.DEFAULT
            inferred_sites.add(site.index)
        # MCL sites: leave out of decisions (ambiguous)

    # Build updated parse
    return Parse(
        decisions=decisions,
        syllables=gold_parse.syllables,
        slots=gold_parse.slots,
        foot_boundaries=gold_parse.foot_boundaries,
        foot_types=gold_parse.foot_types,
        caesura=gold_parse.caesura,
        bucolic_diaeresis=gold_parse.bucolic_diaeresis,
        meter=gold_parse.meter,
    )


# ---------------------------------------------------------------------------
# XML file parsing
# ---------------------------------------------------------------------------


@dataclass
class ParseResult:
    """Result of parsing a single XML file."""
    examples: list[TrainingExample]
    skipped: int
    total: int


def parse_line_element(
    line_el: ET.Element,
    author: str,
    title: str,
    book: str,
    lexicon: VowelLengthLexicon | None = None,
) -> TrainingExample | None:
    """Parse a single <line> XML element into a TrainingExample.

    Returns None if the line should be skipped (corrupt, non-hexameter, etc.).
    """
    name = line_el.get("name", "")
    metre = line_el.get("metre", "")
    pattern = line_el.get("pattern", "")

    # Filter: only numeric line names, hexameter, non-corrupt
    if not name.isdigit():
        return None
    if metre != "H":
        return None
    if not pattern or pattern == "corrupt":
        return None

    # Parse word elements
    word_elements = line_el.findall("word")
    if not word_elements:
        return None

    words = [_parse_word_element(w) for w in word_elements]

    # Build raw text from word texts
    raw_text = " ".join(w.text for w in words)

    # Atomize the raw text (our representation)
    line = atomize(raw_text, lexicon=lexicon)

    # Attach metadata
    line.author = author
    line.work = title
    line.book = book
    line.line_num = int(name)
    line.corpus_id = f"{author}_{title}_{book}_{name}"

    # Build gold parse from MQDQ data
    gold_parse = _build_gold_parse(words)
    if gold_parse is None:
        return None

    # Align gold parse to our atom-level representation
    aligned_parse = align_gold_parse(line, words, gold_parse)

    # Track which components are directly observed from the corpus
    observed = frozenset({"slots", "foot_types", "caesura", "decisions"})

    return TrainingExample(
        line=line,
        gold_parse=aligned_parse,
        observed=observed,
    )


def parse_xml(
    path: Path,
    lexicon: VowelLengthLexicon | None = None,
) -> ParseResult:
    """Parse a Pedecerto XML file into TrainingExamples.

    Returns a ParseResult with the examples and statistics.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    author = root.findtext(".//author") or "unknown"
    title = root.findtext(".//title") or "unknown"

    examples: list[TrainingExample] = []
    skipped = 0
    total = 0

    for div in root.iter("division"):
        book = div.get("title", "")
        for line_el in div.iter("line"):
            total += 1
            example = parse_line_element(line_el, author, title, book, lexicon=lexicon)
            if example is not None:
                examples.append(example)
            else:
                skipped += 1

    return ParseResult(examples=examples, skipped=skipped, total=total)
