"""Learned proposal network (Stage A: per-syllable weights) that augments
the rule-enumerated candidate set to lift the gold-reachability ceiling.
The union stays finite and deduplicated -> exact Z preserved."""
from __future__ import annotations

from itertools import product

import torch
import torch.nn as nn

from latin_ebm.energy_neural import (
    WEIGHT_ID,
    pool_positions,
    syllable_token_positions,
)
from latin_ebm.types import Parse, PhonWeight


def _parse_key(p: Parse):
    return (p.foot_types, p.slots, tuple(sorted(p.decisions.items())))


def augment_candidates(rule: list[Parse], proposed: list[Parse]) -> list[Parse]:
    """Deduplicated union of rule + proposed parses (rule first)."""
    seen = set()
    out: list[Parse] = []
    for p in list(rule) + list(proposed):
        k = _parse_key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


class ProposalNetwork(nn.Module):
    """q(weight per syllable | x) over pooled encoder states. Shares the
    encoder via the scorer at call time. Stage A: factored Bernoulli over L/S."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight_logit = nn.Linear(d_model, 2)


def propose_weight_logits(encoder, prop: ProposalNetwork, tokenizer, line, parse) -> torch.Tensor:
    """Per-syllable L/S logits from pooled encoder states. [n_syl, 2]."""
    tl = tokenizer.encode(line)
    h, _ = encoder(tl)
    rows = [prop.weight_logit(pool_positions(h, syllable_token_positions(s, tl)))
            for s in parse.syllables]
    return torch.stack(rows) if rows else h.new_zeros((0, 2))


def proposal_recall_loss(logits: torch.Tensor, gold_parse) -> torch.Tensor:
    """Per-syllable cross-entropy toward gold weights (recall proxy)."""
    if logits.shape[0] == 0:
        return logits.new_zeros(())
    targets = torch.tensor([WEIGHT_ID[s.weight] for s in gold_parse.syllables])
    return torch.nn.functional.cross_entropy(logits, targets)


def topk_weight_parses(logits: torch.Tensor, base: Parse, k: int) -> list[Parse]:
    """Materialize up to k parses by overriding syllable weights with the
    highest-probability L/S combinations from `logits` (independent per
    syllable; take top-k joint by greedy product of per-syllable bests)."""
    if logits.shape[0] == 0:
        return []
    probs = torch.softmax(logits, dim=-1).detach()
    # per-syllable (weight, logprob), best first
    per_syl = []
    for row in probs:
        ranked = sorted([(PhonWeight.LONG, float(row[0])), (PhonWeight.SHORT, float(row[1]))],
                        key=lambda t: -t[1])
        per_syl.append(ranked)
    # greedy top-k over the joint: start from all-best, flip lowest-margin syllables
    combos = list(product(*[[w for w, _ in ranked] for ranked in per_syl]))[: max(k, 1)]
    out = []
    for combo in combos:
        new_syls = tuple(
            type(s)(atom_indices=s.atom_indices, onset=s.onset, nucleus=s.nucleus,
                    coda=s.coda, is_open=s.is_open, weight=w)
            for s, w in zip(base.syllables, combo)
        )
        out.append(type(base)(decisions=base.decisions, syllables=new_syls,
                              slots=base.slots, foot_boundaries=base.foot_boundaries,
                              foot_types=base.foot_types, caesura=base.caesura,
                              bucolic_diaeresis=base.bucolic_diaeresis, meter=base.meter))
    return out
