"""Measure the gold-reachability ceiling and split unreachable lines into
buckets: 'no_candidates' (enumeration empty) vs 'weight_unreachable'
(candidates exist but none match gold foot_types+slots).

With --proposal, the candidate set for each line is augmented via the
proposal network's top-K weight bundles (augment_candidates(...)) before the
reachability check, and the result is written to a separate JSON. This
measures the new (lifted) ceiling for Stage 1.3. A trained checkpoint can be
supplied via --ckpt; otherwise a random-init encoder/proposal is used (which
still exercises the augmentation path and finiteness, just without learned
weights)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.encoder import MetricalEncoder, MetricalTokenizer
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.proposal import (
    ProposalNetwork,
    augment_candidates,
    propose_weight_logits,
    topk_weight_parses,
)


def _reachable(cands, gold) -> bool:
    return any(c.foot_types == gold.foot_types and c.slots == gold.slots for c in cands)


def _load_encoder_and_proposal(ckpt_path, d_model, n_layers, tok):
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size,
                          d_model=d_model, n_layers=n_layers)
    prop = ProposalNetwork(d_model=d_model)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu")
        enc.load_state_dict(ck["encoder"])
        if "proposal" in ck:
            prop.load_state_dict(ck["proposal"])
    enc.eval()
    prop.eval()
    return enc, prop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--proposal", action="store_true",
                   help="augment candidates via the proposal net before checking reachability")
    p.add_argument("--proposal-topk", type=int, default=8)
    p.add_argument("--ckpt", default=None, help="trained train_v5 checkpoint (.pt)")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    lexicon = VowelLengthLexicon(
        Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt")
    )
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    test = [e for e in result.examples if e.line.book in test_books]

    enc = prop = tok = None
    if args.proposal:
        tok = MetricalTokenizer.build([e.line for e in test])
        d_model, n_layers = args.d_model, args.n_layers
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location="cpu")
            tok = MetricalTokenizer(ck["text_vocab"], ck["word_vocab"])
            cfg = ck.get("config", {})
            d_model = cfg.get("d_model", d_model)
            n_layers = cfg.get("n_layers", n_layers)
        enc, prop = _load_encoder_and_proposal(args.ckpt, d_model, n_layers, tok)

    reachable = no_candidates = weight_unreachable = 0
    with torch.no_grad():
        for ex in test:
            cands = enumerate_parses(ex.line)
            if not cands:
                no_candidates += 1
                continue
            if args.proposal:
                logits = propose_weight_logits(enc, prop, tok, ex.line, cands[0])
                proposed = topk_weight_parses(logits, cands[0], args.proposal_topk)
                cands = augment_candidates(cands, proposed)
            gold = ex.gold_parse
            if _reachable(cands, gold):
                reachable += 1
            else:
                weight_unreachable += 1

    n = len(test)
    out = {
        "n_test": n,
        "proposal": args.proposal,
        "reachable": reachable,
        "reachable_pct": reachable / n,
        "no_candidates": no_candidates,
        "no_candidates_pct": no_candidates / n,
        "weight_unreachable": weight_unreachable,
        "weight_unreachable_pct": weight_unreachable / n,
    }
    out_path = args.out or (
        "results/ceiling_split_proposal.json" if args.proposal
        else "results/ceiling_split_b12.json"
    )
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
