#!/usr/bin/env python3
"""CRF baseline: sequence labeling on gold syllabification.

This is the standard pipeline approach from Nolden/Ycreak:
1. Use Pedecerto's gold syllabification (syllable boundaries known)
2. Label each syllable as L(ong) or S(hort)
3. Train a CRF on per-syllable features
4. Derive foot pattern from predicted L/S sequence

This isolates the value of the EBM's joint inference vs a pipeline.
If the CRF gets ~90%+, the EBM's 63.6% is mostly a data/enumeration problem.
If the CRF also gets ~65%, the features are insufficient.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

from latin_ebm.corpus.pedecerto import decode_sy, WordInfo, _parse_word_element
from latin_ebm.normalize import normalize
from latin_ebm.types import FootType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data extraction: syllable-level features from Pedecerto XML
# ---------------------------------------------------------------------------

VOWELS = set("aeiouy")


@dataclass
class SyllableData:
    """One syllable with features and gold label."""
    word_text: str
    vowel: str          # nucleus vowel(s)
    is_open: bool       # crude: does syllable end before a consonant cluster?
    position: int       # position in line (0-indexed)
    total_sylls: int    # total syllables in line
    word_position: str  # "initial", "medial", "final"
    word_idx: int
    n_words: int
    label: str          # "L" or "S"


def _extract_syllables_from_line(line_el: ET.Element) -> list[SyllableData] | None:
    """Extract syllable data from a <line> element."""
    name = line_el.get("name", "")
    metre = line_el.get("metre", "")
    pattern = line_el.get("pattern", "")

    if not name.isdigit() or metre != "H" or not pattern or pattern == "corrupt":
        return None

    word_elements = line_el.findall("word")
    if not word_elements:
        return None

    words = [_parse_word_element(w) for w in word_elements]

    # Collect all syllables with labels
    all_syllables: list[SyllableData] = []
    total_syls = sum(len(w.syllables) for w in words)

    if total_syls < 12 or total_syls > 17:
        return None

    pos = 0
    for word_idx, word in enumerate(words):
        word_text = normalize(word.text)
        n_word_syls = len(word.syllables)

        for syl_idx, syl_info in enumerate(word.syllables):
            # Determine label
            if syl_info.position in ("A", "T"):
                label = "L"
            elif syl_info.position in ("b", "c"):
                label = "S"
            elif syl_info.position == "X":
                label = "L"  # anceps, treat as long for labeling
            else:
                label = "L"

            # Crude vowel extraction from word text
            vowels_in_word = [ch for ch in word_text if ch in VOWELS]
            vowel = vowels_in_word[min(syl_idx, len(vowels_in_word) - 1)] if vowels_in_word else "a"

            # Word position
            if n_word_syls == 1:
                word_pos = "mono"
            elif syl_idx == 0:
                word_pos = "initial"
            elif syl_idx == n_word_syls - 1:
                word_pos = "final"
            else:
                word_pos = "medial"

            all_syllables.append(SyllableData(
                word_text=word_text,
                vowel=vowel,
                is_open=(syl_idx < n_word_syls - 1),  # crude approximation
                position=pos,
                total_sylls=total_syls,
                word_position=word_pos,
                word_idx=word_idx,
                n_words=len(words),
                label=label,
            ))
            pos += 1

    return all_syllables


# ---------------------------------------------------------------------------
# CRF feature extraction
# ---------------------------------------------------------------------------


def syllable_features(syl: SyllableData, prev_syl: SyllableData | None, next_syl: SyllableData | None) -> dict[str, str | float]:
    """Extract features for one syllable."""
    features: dict[str, str | float] = {
        "vowel": syl.vowel,
        "word_position": syl.word_position,
        "position_in_line": str(syl.position),
        "total_sylls": str(syl.total_sylls),
        "rel_position": f"{syl.position / syl.total_sylls:.1f}",
        "word": syl.word_text[:6],  # truncated word form
        "word_idx": str(syl.word_idx),
        "n_words": str(syl.n_words),
    }

    # Metrical position features (which foot position this could be)
    # Position mod patterns for hexameter
    features["pos_mod_2"] = str(syl.position % 2)
    features["pos_mod_3"] = str(syl.position % 3)

    if prev_syl:
        features["prev_vowel"] = prev_syl.vowel
        features["prev_word_pos"] = prev_syl.word_position
        features["same_word_as_prev"] = str(syl.word_idx == prev_syl.word_idx)

    if next_syl:
        features["next_vowel"] = next_syl.vowel
        features["next_word_pos"] = next_syl.word_position
        features["same_word_as_next"] = str(syl.word_idx == next_syl.word_idx)

    # Line-final syllable is anceps
    if syl.position == syl.total_sylls - 1:
        features["is_final"] = "1"

    return features


def line_to_features_and_labels(syllables: list[SyllableData]) -> tuple[list[dict], list[str]]:
    """Convert a line's syllables into CRF features and labels."""
    features = []
    labels = []
    for i, syl in enumerate(syllables):
        prev_syl = syllables[i - 1] if i > 0 else None
        next_syl = syllables[i + 1] if i < len(syllables) - 1 else None
        features.append(syllable_features(syl, prev_syl, next_syl))
        labels.append(syl.label)
    return features, labels


# ---------------------------------------------------------------------------
# Foot pattern derivation
# ---------------------------------------------------------------------------


def labels_to_foot_types(labels: list[str]) -> tuple[FootType, ...] | None:
    """Derive foot types from a sequence of L/S labels."""
    n = len(labels)
    if n < 12 or n > 17:
        return None

    # Try to parse as hexameter: 6 feet
    feet: list[FootType] = []
    pos = 0
    for foot_idx in range(6):
        if foot_idx == 5:
            # Final foot: L + X (2 syllables)
            if pos + 2 <= n:
                feet.append(FootType.FINAL)
                pos += 2
            else:
                return None
        else:
            # Try dactyl (L S S = 3 syllables) or spondee (L L = 2 syllables)
            if pos + 3 <= n and labels[pos] == "L" and labels[pos + 1] == "S" and labels[pos + 2] == "S":
                feet.append(FootType.DACTYL)
                pos += 3
            elif pos + 2 <= n and labels[pos] == "L" and labels[pos + 1] == "L":
                feet.append(FootType.SPONDEE)
                pos += 2
            elif pos + 3 <= n and labels[pos + 1] == "S":
                # Force dactyl if second position is short
                feet.append(FootType.DACTYL)
                pos += 3
            elif pos + 2 <= n:
                # Default to spondee
                feet.append(FootType.SPONDEE)
                pos += 2
            else:
                return None

    if pos != n:
        return None

    return tuple(feet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    xml_path = Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml"
    test_books = {"11", "12"}

    logger.info("Parsing XML...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    train_lines: list[list[SyllableData]] = []
    test_lines: list[list[SyllableData]] = []
    test_gold_feet: list[tuple[FootType, ...]] = []
    train_gold_feet: list[tuple[FootType, ...]] = []

    for div in root.iter("division"):
        book = div.get("title", "")
        for line_el in div.iter("line"):
            syllables = _extract_syllables_from_line(line_el)
            if syllables is None:
                continue

            gold_labels = [s.label for s in syllables]
            gold_feet = labels_to_foot_types(gold_labels)
            if gold_feet is None or len(gold_feet) != 6:
                continue

            if book in test_books:
                test_lines.append(syllables)
                test_gold_feet.append(gold_feet)
            else:
                train_lines.append(syllables)
                train_gold_feet.append(gold_feet)

    logger.info("Train: %d lines, Test: %d lines", len(train_lines), len(test_lines))

    # Extract features
    X_train = [line_to_features_and_labels(line)[0] for line in train_lines]
    y_train = [line_to_features_and_labels(line)[1] for line in train_lines]
    X_test = [line_to_features_and_labels(line)[0] for line in test_lines]
    y_test = [line_to_features_and_labels(line)[1] for line in test_lines]

    # Train CRF
    logger.info("Training CRF...")
    t0 = time.time()
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.01,
        c2=0.01,
        max_iterations=200,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    t1 = time.time()
    logger.info("CRF training took %.1fs", t1 - t0)

    # Predict
    y_pred = crf.predict(X_test)

    # Syllable-level accuracy
    correct = sum(
        pred == gold
        for pred_line, gold_line in zip(y_pred, y_test)
        for pred, gold in zip(pred_line, gold_line)
    )
    total = sum(len(line) for line in y_test)
    syl_acc = correct / total
    logger.info("Syllable accuracy: %.1f%%", syl_acc * 100)

    # Foot pattern accuracy
    foot_correct = 0
    foot_total = 0
    for pred_labels, gold_feet in zip(y_pred, test_gold_feet):
        pred_feet = labels_to_foot_types(list(pred_labels))
        foot_total += 1
        if pred_feet == gold_feet:
            foot_correct += 1

    foot_acc = foot_correct / foot_total if foot_total > 0 else 0
    logger.info("Foot pattern accuracy: %.1f%%", foot_acc * 100)

    print(f"\n{'=' * 60}")
    print(f"  CRF Baseline (gold syllabification)")
    print(f"{'=' * 60}")
    print(f"  Train lines:          {len(train_lines)}")
    print(f"  Test lines:           {len(test_lines)}")
    print(f"  Syllable accuracy:    {syl_acc:.1%}")
    print(f"  Foot pattern acc:     {foot_acc:.1%}")
    print(f"\n  For comparison:")
    print(f"  EBM v1 (joint):       63.6% foot pattern (on subset with candidates)")
    print(f"  Default baseline:     39.7%")
    print(f"  Random baseline:      30.3%")

    # Per-label breakdown
    print(f"\n  CRF per-label report:")
    flat_pred = [label for line in y_pred for label in line]
    flat_gold = [label for line in y_test for label in line]
    for label in ["L", "S"]:
        tp = sum(1 for p, g in zip(flat_pred, flat_gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(flat_pred, flat_gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(flat_pred, flat_gold) if p != label and g == label)
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        print(f"    {label}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
