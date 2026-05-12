"""Per-line failure-mode classifier for ceiling diagnostics.

Tags each (line, gold) pair as one of:
  - CORRECT: gold foot pattern is reachable (in candidate set)
  - GOLD_UNREACHABLE: candidates exist but gold's foot pattern is absent
  - NO_CANDIDATES: zero candidates pass enumeration filters

Sub-reasons further classify NO_CANDIDATES and GOLD_UNREACHABLE.

Output TSV columns: line_id, status, reason, n_candidates, n_atoms, n_sites
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from latin_ebm.types import LatinLine, Parse, PhonWeight
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.realize import syllable_count


class LineStatus(str, Enum):
    CORRECT = "correct"             # gold foot pattern reachable
    GOLD_UNREACHABLE = "gold_unreachable"
    NO_CANDIDATES = "no_candidates"


@dataclass(frozen=True)
class DiagnosisRow:
    line_id: str
    status: LineStatus
    reason: Optional[str]
    n_candidates: int
    n_atoms: int
    n_sites: int


def _foot_types_match(candidate: Parse, gold: Parse) -> bool:
    return candidate.foot_types == gold.foot_types


def _no_candidates_reason(line: LatinLine) -> str:
    by_word: dict[int, int] = {}
    for atom in line.atoms:
        by_word[atom.word_idx] = by_word.get(atom.word_idx, 0) + 1
    if by_word and max(by_word.values()) >= 4:
        return "vowel_chain"
    default_decisions = {s.index: s.default for s in line.sites}
    try:
        m = syllable_count(line, default_decisions)
    except Exception:
        return "realize_error"
    if m < 12 or m > 17:
        return "syllable_count_oob"
    return "weight_filter"


def _gold_unreachable_reason(line: LatinLine, gold: Parse, candidates: list[Parse]) -> str:
    if not candidates:
        return "no_candidates"

    def shared_prefix(c: Parse) -> int:
        n = 0
        for cs, gs in zip(c.slots, gold.slots):
            if cs != gs:
                break
            n += 1
        return n

    closest = max(candidates, key=shared_prefix)
    idx = shared_prefix(closest)
    if idx >= len(gold.slots) or idx >= len(closest.slots):
        return "decisions_mismatch"
    if idx >= len(closest.syllables) or idx >= len(gold.syllables):
        return "syllable_count_mismatch"
    cand_syl = closest.syllables[idx]
    gold_syl = gold.syllables[idx]
    if cand_syl.is_open and gold_syl.weight == PhonWeight.LONG and cand_syl.weight == PhonWeight.SHORT:
        return "open_syllable_length"
    cand_has_diph = any(line.atoms[i].in_diphthong for i in cand_syl.atom_indices)
    gold_has_diph = any(line.atoms[i].in_diphthong for i in gold_syl.atom_indices)
    if cand_has_diph != gold_has_diph:
        return "diphthong_mismatch"
    if len(closest.syllables) != len(gold.syllables):
        return "elision_mismatch"
    return "other"


def classify_line(line: LatinLine, gold: Parse) -> DiagnosisRow:
    try:
        candidates = enumerate_parses(line, meter=None)
    except Exception as e:
        return DiagnosisRow(
            line_id=line.corpus_id,
            status=LineStatus.NO_CANDIDATES,
            reason=f"enumerate_error:{type(e).__name__}",
            n_candidates=0,
            n_atoms=len(line.atoms),
            n_sites=len(line.sites),
        )
    if not candidates:
        return DiagnosisRow(
            line_id=line.corpus_id,
            status=LineStatus.NO_CANDIDATES,
            reason=_no_candidates_reason(line),
            n_candidates=0,
            n_atoms=len(line.atoms),
            n_sites=len(line.sites),
        )
    if any(_foot_types_match(c, gold) for c in candidates):
        return DiagnosisRow(
            line_id=line.corpus_id,
            status=LineStatus.CORRECT,
            reason=None,
            n_candidates=len(candidates),
            n_atoms=len(line.atoms),
            n_sites=len(line.sites),
        )
    return DiagnosisRow(
        line_id=line.corpus_id,
        status=LineStatus.GOLD_UNREACHABLE,
        reason=_gold_unreachable_reason(line, gold, candidates),
        n_candidates=len(candidates),
        n_atoms=len(line.atoms),
        n_sites=len(line.sites),
    )


def main():
    from latin_ebm.corpus.pedecerto import parse_xml
    from latin_ebm.lexicon import VowelLengthLexicon

    p = argparse.ArgumentParser()
    p.add_argument("xml")
    p.add_argument("--books", default="1,2")
    p.add_argument("--out", required=True)
    p.add_argument("--no-lexicon", action="store_true")
    p.add_argument("--data-dir", default="data",
                   help="directory containing MqDqMacrons.json and MorpheusMacrons.txt")
    args = p.parse_args()
    books = set(args.books.split(","))

    lexicon = None
    if not args.no_lexicon:
        data = Path(args.data_dir)
        lexicon = VowelLengthLexicon(
            mqdq_path=data / "MqDqMacrons.json",
            morpheus_path=data / "MorpheusMacrons.txt",
        )

    result = parse_xml(Path(args.xml), lexicon=lexicon)

    n_written = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["line_id", "status", "reason", "n_candidates", "n_atoms", "n_sites"])
        for ex in result.examples:
            if ex.line.book not in books:
                continue
            ex.line.corpus_id = f"VERG-aene.{ex.line.book}.{ex.line.line_num}"
            row = classify_line(ex.line, ex.gold_parse)
            w.writerow([row.line_id, row.status.value, row.reason or "",
                        row.n_candidates, row.n_atoms, row.n_sites])
            n_written += 1
    print(f"Wrote {n_written} rows to {args.out}")


if __name__ == "__main__":
    main()
