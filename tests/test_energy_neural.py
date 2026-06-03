import torch

from latin_ebm.encoder import MetricalTokenizer, MetricalEncoder
from latin_ebm.energy_neural import (
    pool_positions, syllable_token_positions,
    StructuredEnergyHead, NeuralScorer, AuxHeads,
)
from latin_ebm.enumerate import enumerate_parses


def test_pool_mean(sample_line):
    tok = MetricalTokenizer.build([sample_line])
    tl = tok.encode(sample_line)
    h = torch.arange(tl.n_tokens * 4, dtype=torch.float32).view(tl.n_tokens, 4)
    pooled = pool_positions(h, [0, 2])  # mean of token rows 0 and 2
    expected = (h[0] + h[2]) / 2
    assert torch.allclose(pooled, expected)


def test_syllable_positions_include_flanking_bridges(sample_line, sample_syllables):
    tok = MetricalTokenizer.build([sample_line])
    tl = tok.encode(sample_line)
    # syllable over atom 0 -> token 0, plus flanking bridge token 1
    positions = syllable_token_positions(sample_syllables[0], tl)
    assert 0 in positions          # the nucleus atom token
    assert 1 in positions          # the bridge to its right (onset/coda material)


def test_candidate_energy_is_scalar_and_decomposed(sample_line, sample_parse):
    tok = MetricalTokenizer.build([sample_line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    head = StructuredEnergyHead(d_model=32)
    tl = tok.encode(sample_line)
    h, h_line = enc(tl)
    dec = head.candidate_energy(h, h_line, tl, sample_line, sample_parse)
    assert dec.e_total.shape == ()
    # decomposition sums to total
    assert torch.allclose(
        dec.e_total, dec.e_site + dec.e_syll + dec.e_foot + dec.e_global
    )
    assert dec.e_total.requires_grad


def test_scorer_energies_and_exact_partition(real_aeneid_line):
    line, _gold = real_aeneid_line
    cands = enumerate_parses(line)
    assert len(cands) >= 2
    tok = MetricalTokenizer.build([line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    head = StructuredEnergyHead(d_model=32)
    scorer = NeuralScorer(enc, head, tok)
    energies = scorer.energies(line, cands)            # [n_cands], from ONE encode
    assert energies.shape == (len(cands),)
    log_z = torch.logsumexp(-energies, dim=0)
    brute = torch.log(torch.exp(-energies).sum())
    assert torch.allclose(log_z, brute, atol=1e-5)     # exact finite partition


def test_vectorized_matches_singular(real_aeneid_line):
    """The batched candidate_energies must equal stacking the per-candidate
    candidate_energy — same numbers, just computed in one pass. This is the
    guarantee that the vectorization optimization preserves semantics."""
    line, _gold = real_aeneid_line
    cands = enumerate_parses(line)
    assert len(cands) >= 2
    tok = MetricalTokenizer.build([line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    head = StructuredEnergyHead(d_model=32)
    tl = tok.encode(line)
    h, h_line = enc(tl)
    batched = head.candidate_energies(h, h_line, tl, line, cands)
    singular = torch.stack([head.candidate_energy(h, h_line, tl, line, c).e_total for c in cands])
    assert batched.e_total.shape == singular.shape
    assert torch.allclose(batched.e_total, singular, atol=1e-5)
    # decomposition still sums to total in the batched path
    assert torch.allclose(
        batched.e_total,
        batched.e_site + batched.e_syll + batched.e_foot + batched.e_global,
        atol=1e-5,
    )


def test_plan_matches_vectorized(real_aeneid_line):
    """energies_from_plan(precompute_plan(...)) must equal candidate_energies —
    the cached-structure path is numerically identical to the inline one. This
    guards the precompute/caching optimization (and the exact finite sum)."""
    line, _gold = real_aeneid_line
    cands = enumerate_parses(line)
    assert len(cands) >= 2
    tok = MetricalTokenizer.build([line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    head = StructuredEnergyHead(d_model=32)
    tl = tok.encode(line)
    h, h_line = enc(tl)
    inline = head.candidate_energies(h, h_line, tl, line, cands).e_total
    plan = head.precompute_plan(tl, line, cands)
    cached = head.energies_from_plan(h, h_line, plan).e_total
    assert torch.allclose(inline, cached, atol=1e-6)
    # plan is h-independent: same plan, a different h gives self-consistent results
    h2, h2_line = enc(tl)
    again = head.energies_from_plan(h2, h2_line, plan).e_total
    assert again.shape == cached.shape


def test_batched_energies_matches_per_line(real_aeneid_line):
    """batched_energies over a minibatch must equal looping energies_from_plan
    per line — the cross-line batching is a pure dispatch optimization, not a
    semantic change. Uses the same line twice (a valid 2-line minibatch) so the
    flatten/index_add/un-flatten round-trips on real candidate counts."""
    line, _gold = real_aeneid_line
    cands = enumerate_parses(line)
    assert len(cands) >= 2
    tok = MetricalTokenizer.build([line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    head = StructuredEnergyHead(d_model=32)
    tl = tok.encode(line)
    plan = head.precompute_plan(tl, line, cands)

    # build a 2-line minibatch (distinct h per slot, same structure)
    h0, hl0 = enc(tl)
    h1, hl1 = enc(tl)
    per_line = [
        head.energies_from_plan(h0, hl0, plan).e_total,
        head.energies_from_plan(h1, hl1, plan).e_total,
    ]
    batched = head.batched_energies([h0, h1], [hl0, hl1], [plan, plan])
    assert len(batched) == 2
    for a, b in zip(batched, per_line):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-5)


def test_aux_heads_shapes(sample_line):
    tok = MetricalTokenizer.build([sample_line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    aux = AuxHeads(d_model=32, text_vocab_size=tok.text_vocab_size)
    tl = tok.encode(sample_line)
    h, _ = enc(tl)
    mlm_logits = aux.mlm(h)                 # [T, vocab]
    assert mlm_logits.shape == (tl.n_tokens, tok.text_vocab_size)
    pooled = h[0]
    weight_logits = aux.weight(pooled)      # [2]
    assert weight_logits.shape == (2,)
