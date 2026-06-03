"""Tests for the v2 feature additions (Phase B/C/D) and dense per-foot extractor."""
from __future__ import annotations

from pathlib import Path

import pytest

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.features import FeatureIndex, extract_features
from latin_ebm.features_v2 import PER_FOOT_DIM, per_foot_dense_features
from latin_ebm.lexicon import VowelLengthLexicon


@pytest.fixture(scope="module")
def aen_examples():
    lex = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    r = parse_xml(Path("pedecerto-raw/VERG-aene.xml"), lexicon=lex)
    return r.examples[:50], lex


def test_per_syllable_weight_features(aen_examples):
    exs, lex = aen_examples
    line = exs[0].line
    cands = enumerate_parses(line)
    idx = FeatureIndex()
    extract_features(line, cands[0], idx, lexicon=lex)
    names = idx.names
    assert any(n.startswith("syl_pos:0:") for n in names)


def test_foot_bigram_features(aen_examples):
    exs, lex = aen_examples
    line = exs[0].line
    cands = enumerate_parses(line)
    idx = FeatureIndex()
    extract_features(line, cands[0], idx, lexicon=lex)
    names = idx.names
    bigrams = [n for n in names if n.startswith("foot_bigram:")]
    assert len(bigrams) >= 5


def test_per_foot_features(aen_examples):
    exs, lex = aen_examples
    line = exs[0].line
    cands = enumerate_parses(line)
    idx = FeatureIndex()
    extract_features(line, cands[0], idx, lexicon=lex)
    names = idx.names
    for foot_idx in range(5):
        assert any(n.startswith(f"foot:{foot_idx}:") for n in names)


def test_lexicon_lemma_lookup(aen_examples):
    _, lex = aen_examples
    assert lex.lemma("arma") is not None
    assert lex.lemma("zzzz_not_a_word") is None


def test_per_foot_dense_features_shape(aen_examples):
    exs, _ = aen_examples
    line = exs[0].line
    cands = enumerate_parses(line)
    feats = per_foot_dense_features(line, cands[0])
    assert feats.shape == (6, PER_FOOT_DIM)
    for i in range(6):
        assert feats[i, 0:3].sum() == 1.0


def test_mlp_forward():
    import torch
    from latin_ebm.mlp import PerFootMLP
    m = PerFootMLP(input_dim=22, hidden_dim=8)
    x = torch.randn(6, 22)
    y = m(x)
    assert y.shape == ()
