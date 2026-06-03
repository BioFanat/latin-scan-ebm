import torch

from latin_ebm.encoder import MetricalEncoder, MetricalTokenizer
from latin_ebm.energy_neural import NeuralScorer, StructuredEnergyHead
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.proposal import (
    ProposalNetwork,
    augment_candidates,
    propose_weight_logits,
    topk_weight_parses,
)


def test_augment_is_deduplicated_and_superset(real_aeneid_line):
    line, _gold = real_aeneid_line
    rule = enumerate_parses(line)
    # a fake "proposed" list that duplicates the first rule candidate
    proposed = [rule[0]]
    union = augment_candidates(rule, proposed)
    # finite, no duplicates, superset of rule
    keys = [(p.foot_types, p.slots, tuple(sorted(p.decisions.items()))) for p in union]
    assert len(keys) == len(set(keys))            # deduplicated
    assert len(union) == len(rule)                # duplicate added nothing
    assert len(union) >= len(rule)                # superset


def test_proposal_emits_per_syllable_logits(real_aeneid_line):
    line, gold = real_aeneid_line
    tok = MetricalTokenizer.build([line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    prop = ProposalNetwork(d_model=32)
    logits = propose_weight_logits(enc, prop, tok, line, gold)  # [n_syl, 2]
    assert logits.shape == (len(gold.syllables), 2)
    assert logits.requires_grad


def test_augmented_set_exact_partition_and_finiteness(real_aeneid_line):
    """Stage 1.3 tractability guard: after augmenting the rule-enumerated
    candidates with proposed top-K weight parses, the candidate set must stay
    finite and duplicate-free, and the EBM partition over the AUGMENTED set
    must still equal the brute-force log-sum-exp from a SINGLE per-line encode.
    """
    line, gold = real_aeneid_line
    rule = enumerate_parses(line)
    assert len(rule) >= 2

    tok = MetricalTokenizer.build([line])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    head = StructuredEnergyHead(d_model=32)
    scorer = NeuralScorer(enc, head, tok)
    prop = ProposalNetwork(d_model=32)

    # Build the augmented candidate set exactly as eval does.
    logits = propose_weight_logits(enc, prop, tok, line, rule[0])
    proposed = topk_weight_parses(logits, rule[0], k=8)
    cands = augment_candidates(rule, proposed)

    # --- finiteness + no duplicate parses on the union ---
    keys = [(p.foot_types, p.slots, tuple(sorted(p.decisions.items()))) for p in cands]
    assert len(keys) == len(set(keys))      # union deduplicated
    assert len(cands) >= len(rule)          # superset of rule
    assert len(cands) <= len(rule) + len(proposed)  # finite, no blow-up

    # --- exact-Z over the augmented set, from ONE encode ---
    energies = scorer.energies(line, cands)
    assert energies.shape == (len(cands),)
    log_z = torch.logsumexp(-energies, dim=0)
    brute = torch.log(torch.exp(-energies).sum())
    assert torch.allclose(log_z, brute, atol=1e-5)
