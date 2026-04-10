"""Serialization for Latin scansion types.

Three levels:
1. JSON dicts — for test fixtures, debugging, human inspection
2. Polars DataFrames — star schema for corpus-level data
3. Parquet — persistent storage via Polars
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import polars as pl

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
# Enum serialization helpers
# ---------------------------------------------------------------------------

# Registry for deserialization: "PhonWeight.LONG" → PhonWeight.LONG
_ENUM_CLASSES: list[type[Enum]] = [
    PhonWeight, MetricalSlot, FootType, SiteType, SiteChoice, CaesuraType,
]
_ENUM_LOOKUP: dict[str, Enum] = {}
for _cls in _ENUM_CLASSES:
    for _member in _cls:
        _ENUM_LOOKUP[f"{_cls.__name__}.{_member.name}"] = _member


def _enum_to_str(e: Enum) -> str:
    return f"{type(e).__name__}.{e.name}"


def _str_to_enum(s: str) -> Enum:
    return _ENUM_LOOKUP[s]


# ---------------------------------------------------------------------------
# JSON: individual objects ↔ dicts
# ---------------------------------------------------------------------------


def atom_to_dict(atom: VocalicAtom) -> dict[str, Any]:
    return {
        "index": atom.index,
        "chars": atom.chars,
        "word_idx": atom.word_idx,
        "natural_length": _enum_to_str(atom.natural_length) if atom.natural_length else None,
        "in_diphthong": atom.in_diphthong,
        "diphthong_role": atom.diphthong_role,
        "is_word_final": atom.is_word_final,
        "is_word_initial": atom.is_word_initial,
    }


def atom_from_dict(d: dict[str, Any]) -> VocalicAtom:
    return VocalicAtom(
        index=d["index"],
        chars=d["chars"],
        word_idx=d["word_idx"],
        natural_length=_str_to_enum(d["natural_length"]) if d["natural_length"] else None,  # type: ignore[arg-type]
        in_diphthong=d["in_diphthong"],
        diphthong_role=d["diphthong_role"],
        is_word_final=d["is_word_final"],
        is_word_initial=d["is_word_initial"],
    )


def bridge_to_dict(bridge: ConsonantBridge) -> dict[str, Any]:
    return {
        "chars": bridge.chars,
        "has_word_boundary": bridge.has_word_boundary,
        "is_muta_cum_liquida": bridge.is_muta_cum_liquida,
    }


def bridge_from_dict(d: dict[str, Any]) -> ConsonantBridge:
    return ConsonantBridge(
        chars=d["chars"],
        has_word_boundary=d["has_word_boundary"],
        is_muta_cum_liquida=d["is_muta_cum_liquida"],
    )


def site_to_dict(site: AmbiguitySite) -> dict[str, Any]:
    return {
        "index": site.index,
        "site_type": _enum_to_str(site.site_type),
        "atom_indices": list(site.atom_indices),
        "valid_choices": [_enum_to_str(c) for c in site.valid_choices],
        "default": _enum_to_str(site.default),
    }


def site_from_dict(d: dict[str, Any]) -> AmbiguitySite:
    return AmbiguitySite(
        index=d["index"],
        site_type=_str_to_enum(d["site_type"]),  # type: ignore[arg-type]
        atom_indices=tuple(d["atom_indices"]),
        valid_choices=tuple(_str_to_enum(c) for c in d["valid_choices"]),  # type: ignore[arg-type]
        default=_str_to_enum(d["default"]),  # type: ignore[arg-type]
    )


def line_to_dict(line: LatinLine) -> dict[str, Any]:
    return {
        "raw": line.raw,
        "normalized": line.normalized,
        "words": list(line.words),
        "atoms": [atom_to_dict(a) for a in line.atoms],
        "bridges": [bridge_to_dict(b) for b in line.bridges],
        "sites": [site_to_dict(s) for s in line.sites],
        "author": line.author,
        "work": line.work,
        "book": line.book,
        "line_num": line.line_num,
        "corpus_id": line.corpus_id,
    }


def line_from_dict(d: dict[str, Any]) -> LatinLine:
    return LatinLine(
        raw=d["raw"],
        normalized=d["normalized"],
        words=tuple(d["words"]),
        atoms=tuple(atom_from_dict(a) for a in d["atoms"]),
        bridges=tuple(bridge_from_dict(b) for b in d["bridges"]),
        sites=tuple(site_from_dict(s) for s in d["sites"]),
        author=d.get("author", ""),
        work=d.get("work", ""),
        book=d.get("book", ""),
        line_num=d.get("line_num", 0),
        corpus_id=d.get("corpus_id", ""),
    )


def syllable_to_dict(syll: RealizedSyllable) -> dict[str, Any]:
    return {
        "atom_indices": list(syll.atom_indices),
        "onset": syll.onset,
        "nucleus": syll.nucleus,
        "coda": syll.coda,
        "is_open": syll.is_open,
        "weight": _enum_to_str(syll.weight),
    }


def syllable_from_dict(d: dict[str, Any]) -> RealizedSyllable:
    return RealizedSyllable(
        atom_indices=tuple(d["atom_indices"]),
        onset=d["onset"],
        nucleus=d["nucleus"],
        coda=d["coda"],
        is_open=d["is_open"],
        weight=_str_to_enum(d["weight"]),  # type: ignore[arg-type]
    )


def parse_to_dict(parse: Parse) -> dict[str, Any]:
    return {
        "decisions": {str(k): _enum_to_str(v) for k, v in parse.decisions.items()},
        "syllables": [syllable_to_dict(s) for s in parse.syllables],
        "slots": [_enum_to_str(s) for s in parse.slots],
        "foot_boundaries": list(parse.foot_boundaries),
        "foot_types": [_enum_to_str(ft) for ft in parse.foot_types],
        "caesura": _enum_to_str(parse.caesura),
        "bucolic_diaeresis": parse.bucolic_diaeresis,
        "meter": parse.meter,
    }


def parse_from_dict(d: dict[str, Any]) -> Parse:
    return Parse(
        decisions={int(k): _str_to_enum(v) for k, v in d["decisions"].items()},  # type: ignore[misc]
        syllables=tuple(syllable_from_dict(s) for s in d["syllables"]),
        slots=tuple(_str_to_enum(s) for s in d["slots"]),  # type: ignore[arg-type]
        foot_boundaries=tuple(d["foot_boundaries"]),
        foot_types=tuple(_str_to_enum(ft) for ft in d["foot_types"]),  # type: ignore[arg-type]
        caesura=_str_to_enum(d["caesura"]),  # type: ignore[arg-type]
        bucolic_diaeresis=d["bucolic_diaeresis"],
        meter=d["meter"],
    )


def scored_parse_to_dict(sp: ScoredParse) -> dict[str, Any]:
    return {
        "parse": parse_to_dict(sp.parse),
        "e_total": sp.e_total,
        "e_site": sp.e_site,
        "e_syll": sp.e_syll,
        "e_pair": sp.e_pair,
        "e_foot": sp.e_foot,
        "e_global": sp.e_global,
    }


def scored_parse_from_dict(d: dict[str, Any]) -> ScoredParse:
    return ScoredParse(
        parse=parse_from_dict(d["parse"]),
        e_total=d["e_total"],
        e_site=d["e_site"],
        e_syll=d["e_syll"],
        e_pair=d["e_pair"],
        e_foot=d["e_foot"],
        e_global=d["e_global"],
    )


def example_to_dict(ex: TrainingExample) -> dict[str, Any]:
    return {
        "line": line_to_dict(ex.line),
        "gold_parse": parse_to_dict(ex.gold_parse),
        "observed": sorted(ex.observed),
    }


def example_from_dict(d: dict[str, Any]) -> TrainingExample:
    return TrainingExample(
        line=line_from_dict(d["line"]),
        gold_parse=parse_from_dict(d["gold_parse"]),
        observed=frozenset(d["observed"]),
    )


# ---------------------------------------------------------------------------
# JSON file I/O
# ---------------------------------------------------------------------------


def save_json(obj: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Polars: star-schema DataFrames for corpus data
# ---------------------------------------------------------------------------


def lines_to_polars(lines: Sequence[LatinLine]) -> dict[str, pl.DataFrame]:
    """Convert a sequence of LatinLine objects to a star schema of DataFrames.

    Returns {"lines": ..., "atoms": ..., "bridges": ..., "sites": ...}.
    """
    line_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []

    for line in lines:
        cid = line.corpus_id or f"{line.author}_{line.work}_{line.book}_{line.line_num}"
        line_rows.append({
            "corpus_id": cid,
            "raw": line.raw,
            "normalized": line.normalized,
            "words": json.dumps(list(line.words)),
            "author": line.author,
            "work": line.work,
            "book": line.book,
            "line_num": line.line_num,
            "n_atoms": len(line.atoms),
            "n_sites": len(line.sites),
        })

        for atom in line.atoms:
            atom_rows.append({
                "corpus_id": cid,
                "atom_index": atom.index,
                "chars": atom.chars,
                "word_idx": atom.word_idx,
                "natural_length": _enum_to_str(atom.natural_length) if atom.natural_length else None,
                "in_diphthong": atom.in_diphthong,
                "diphthong_role": atom.diphthong_role,
                "is_word_final": atom.is_word_final,
                "is_word_initial": atom.is_word_initial,
            })

        for i, bridge in enumerate(line.bridges):
            bridge_rows.append({
                "corpus_id": cid,
                "bridge_index": i,
                "chars": bridge.chars,
                "has_word_boundary": bridge.has_word_boundary,
                "is_muta_cum_liquida": bridge.is_muta_cum_liquida,
            })

        for site in line.sites:
            site_rows.append({
                "corpus_id": cid,
                "site_index": site.index,
                "site_type": _enum_to_str(site.site_type),
                "atom_indices": json.dumps(list(site.atom_indices)),
                "valid_choices": json.dumps([_enum_to_str(c) for c in site.valid_choices]),
                "default": _enum_to_str(site.default),
            })

    return {
        "lines": pl.DataFrame(line_rows) if line_rows else pl.DataFrame(),
        "atoms": pl.DataFrame(atom_rows) if atom_rows else pl.DataFrame(),
        "bridges": pl.DataFrame(bridge_rows) if bridge_rows else pl.DataFrame(),
        "sites": pl.DataFrame(site_rows) if site_rows else pl.DataFrame(),
    }


def lines_from_polars(tables: dict[str, pl.DataFrame]) -> list[LatinLine]:
    """Reconstruct LatinLine objects from a star-schema dict of DataFrames."""
    lines_df = tables["lines"]
    atoms_df = tables["atoms"]
    bridges_df = tables["bridges"]
    sites_df = tables["sites"]

    result: list[LatinLine] = []

    for row in lines_df.iter_rows(named=True):
        cid = row["corpus_id"]

        # Filter child tables by corpus_id
        line_atoms_df = atoms_df.filter(pl.col("corpus_id") == cid).sort("atom_index")
        line_bridges_df = bridges_df.filter(pl.col("corpus_id") == cid).sort("bridge_index")
        line_sites_df = sites_df.filter(pl.col("corpus_id") == cid).sort("site_index")

        atoms = tuple(
            VocalicAtom(
                index=r["atom_index"],
                chars=r["chars"],
                word_idx=r["word_idx"],
                natural_length=_str_to_enum(r["natural_length"]) if r["natural_length"] else None,  # type: ignore[arg-type]
                in_diphthong=r["in_diphthong"],
                diphthong_role=r["diphthong_role"],
                is_word_final=r["is_word_final"],
                is_word_initial=r["is_word_initial"],
            )
            for r in line_atoms_df.iter_rows(named=True)
        )

        bridges = tuple(
            ConsonantBridge(
                chars=r["chars"],
                has_word_boundary=r["has_word_boundary"],
                is_muta_cum_liquida=r["is_muta_cum_liquida"],
            )
            for r in line_bridges_df.iter_rows(named=True)
        )

        sites = tuple(
            AmbiguitySite(
                index=r["site_index"],
                site_type=_str_to_enum(r["site_type"]),  # type: ignore[arg-type]
                atom_indices=tuple(json.loads(r["atom_indices"])),
                valid_choices=tuple(_str_to_enum(c) for c in json.loads(r["valid_choices"])),  # type: ignore[arg-type]
                default=_str_to_enum(r["default"]),  # type: ignore[arg-type]
            )
            for r in line_sites_df.iter_rows(named=True)
        )

        result.append(LatinLine(
            raw=row["raw"],
            normalized=row["normalized"],
            words=tuple(json.loads(row["words"])),
            atoms=atoms,
            bridges=bridges,
            sites=sites,
            author=row["author"],
            work=row["work"],
            book=row["book"],
            line_num=row["line_num"],
            corpus_id=cid,
        ))

    return result


# ---------------------------------------------------------------------------
# Parquet: persistent corpus storage
# ---------------------------------------------------------------------------


def save_corpus(path: Path, lines: Sequence[LatinLine]) -> None:
    """Save a corpus as a directory of Parquet files."""
    path.mkdir(parents=True, exist_ok=True)
    tables = lines_to_polars(lines)
    for name, df in tables.items():
        df.write_parquet(path / f"{name}.parquet")


def load_corpus(path: Path) -> list[LatinLine]:
    """Load a corpus from a directory of Parquet files."""
    tables = {
        name: pl.read_parquet(path / f"{name}.parquet")
        for name in ("lines", "atoms", "bridges", "sites")
    }
    return lines_from_polars(tables)
