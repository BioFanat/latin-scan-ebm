"""Multi-task neural EBM trainer (Stage 1.x). Supervised-only at first;
aux losses gated behind flags added in later tasks."""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path

import torch
import torch.optim as optim

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.encoder import MetricalEncoder, MetricalTokenizer
from latin_ebm.energy_neural import NeuralScorer, StructuredEnergyHead
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import evaluate
from latin_ebm.lexicon import VowelLengthLexicon
from latin_ebm.proposal import (
    ProposalNetwork,
    augment_candidates,
    propose_weight_logits,
    proposal_recall_loss,
    topk_weight_parses,
)

logger = logging.getLogger(__name__)


def build_line_data(examples):
    """[(line, candidates, gold_indices)] for enumerable, gold-reachable lines."""
    data = []
    for ex in examples:
        cands = enumerate_parses(ex.line)
        if not cands:
            continue
        g = ex.gold_parse
        gi = [i for i, c in enumerate(cands)
              if c.foot_types == g.foot_types and c.slots == g.slots]
        if not gi:
            continue
        data.append((ex.line, cands, gi, g))
    return data


def evaluate_split(scorer, data, prop=None, proposal_topk=0):
    """Predict per line. When `proposal_topk > 0`, augment each line's rule
    candidates with the top-K proposed weight bundles before scoring. The
    augmented set stays finite + deduplicated, so exact Z is preserved."""
    preds = []
    for (line, cands, _gi, gold) in data:
        scored_cands = cands
        if prop is not None and proposal_topk > 0:
            logits = propose_weight_logits(scorer.encoder, prop, scorer.tokenizer, line, cands[0])
            proposed = topk_weight_parses(logits, cands[0], proposal_topk)
            scored_cands = augment_candidates(cands, proposed)
        preds.append((line, scored_cands[scorer.predict(line, scored_cands)], gold))
    return evaluate(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="pedecerto-raw/VERG-aene.xml")
    p.add_argument("--test-books", default="1,2")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-mlm", type=float, default=0.1)
    p.add_argument("--lambda-mwm", type=float, default=0.1)
    p.add_argument("--mlm-mask-frac", type=float, default=0.15)
    p.add_argument("--lambda-proposal", type=float, default=0.0)
    p.add_argument("--proposal-topk", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16,
                   help="lines per minibatch (encoder batched + one backward per minibatch)")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop if train loss doesn't improve by --min-delta for this many epochs (0=off)")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="minimum train-loss improvement to reset early-stop patience")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=leave default)")
    p.add_argument("--compile", action="store_true", help="torch.compile the head MLPs (dynamic shapes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    lexicon = VowelLengthLexicon(Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt"))
    result = parse_xml(Path(args.xml), lexicon=lexicon)
    test_books = set(args.test_books.split(","))
    train_ex = [e for e in result.examples if e.line.book not in test_books]
    test_ex = [e for e in result.examples if e.line.book in test_books]

    train_data = build_line_data(train_ex)
    test_data = build_line_data(test_ex)
    logger.info("train lines=%d test lines=%d", len(train_data), len(test_data))

    tok = MetricalTokenizer.build([line for (line, *_rest) in train_data])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size,
                          d_model=args.d_model, n_layers=args.n_layers)
    head = StructuredEnergyHead(d_model=args.d_model)
    scorer = NeuralScorer(enc, head, tok)

    from latin_ebm.energy_neural import AuxHeads, WEIGHT_ID
    aux = AuxHeads(d_model=args.d_model, text_vocab_size=tok.text_vocab_size)
    use_proposal = args.proposal_topk > 0 or args.lambda_proposal > 0
    prop = ProposalNetwork(d_model=args.d_model) if use_proposal else None
    params = list(scorer.parameters()) + list(aux.parameters())
    if prop is not None:
        params += list(prop.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.compile:
        head.compile_mlps()

    import random
    from latin_ebm.energy_neural import syllable_token_positions, pool_positions
    ce = torch.nn.functional.cross_entropy

    # Precompute per-line tokenization + candidate STRUCTURE once. Both are
    # h-independent and constant across epochs, so caching them removes the
    # dominant per-step Python cost (tok.encode + precompute_plan).
    train_items = [
        (line, cands, gi, gold, tok.encode(line))
        for (line, cands, gi, gold) in train_data
    ]
    from latin_ebm.energy_neural import gold_pool
    train_items = [
        (line, cands, gi, gold, tl, head.precompute_plan(tl, line, cands),
         gold_pool(tl, gold) if gold.syllables else (None, None))
        for (line, cands, gi, gold, tl) in train_items
    ]
    logger.info("cached tokenization + plans + gold pools for %d lines", len(train_items))

    best_loss = float("inf")
    stale = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        scorer.train()
        total = 0.0
        random.shuffle(train_items)
        # Minibatched: encode B lines in one padded transformer call, score each
        # line's candidates over its own (vectorized) h, accumulate the summed
        # loss over the minibatch, then ONE backward + step. This is minibatch
        # SGD (vs the old per-line step); exact-Z per line is unchanged.
        for start in range(0, len(train_items), args.batch_size):
            batch = train_items[start:start + args.batch_size]
            tls = [it[4] for it in batch]  # cached TokenizedLine

            # MLM masking overrides (per line); None where nothing masked.
            overrides, mask_positions = [], []
            for tl in tls:
                ov = list(tl.token_text_id)
                mp = [p for p, is_a in enumerate(tl.token_is_atom)
                      if is_a and random.random() < args.mlm_mask_frac]
                for p in mp:
                    ov[p] = tok.mask_id
                overrides.append(ov if mp else None)
                mask_positions.append(mp)

            # Two batched encodes: clean (scansion/MWM/proposal) + masked (MLM).
            hs, h_lines = enc.forward_batch(tls)
            need_mlm = args.lambda_mlm > 0 and any(mask_positions)
            hs_masked = enc.forward_batch(tls, overrides)[0] if need_mlm else None
            plans = [it[5] for it in batch]

            # --- scansion NLL: ALL candidates of ALL lines scored in one set of
            # MLP calls (cross-line batched head). Per-line NLL is unchanged. ---
            energies_list = head.batched_energies(hs, h_lines, plans)
            batch_loss = hs[0].new_zeros(())
            for i, (line, cands, gi, gold, tl, plan, gpool) in enumerate(batch):
                energies = energies_list[i]
                log_z = torch.logsumexp(-energies, dim=0)
                log_num = torch.logsumexp(-energies[gi], dim=0)
                scan_loss = -(log_num - log_z)
                batch_loss = batch_loss + scan_loss
                total += float(scan_loss.item())

            # --- MLM: one cross-line Linear over all masked rows, per-line CE ---
            if need_mlm:
                flat_rows, flat_tgt, mlm_owner = [], [], []
                for i, mp in enumerate(mask_positions):
                    if mp:
                        flat_rows.append(hs_masked[i][mp])
                        flat_tgt += [tls[i].token_text_id[p] for p in mp]
                        mlm_owner += [i] * len(mp)
                if flat_rows:
                    logits = aux.mlm(torch.cat(flat_rows))           # one Linear call
                    tgt = torch.tensor(flat_tgt)
                    owner = torch.tensor(mlm_owner)
                    # per-line mean CE (preserves the old per-line loss semantics)
                    for i in range(len(batch)):
                        m = owner == i
                        if m.any():
                            batch_loss = batch_loss + args.lambda_mlm * ce(logits[m], tgt[m])

            # --- MWM + proposal: pool with cached gold matrices, one Linear each ---
            if args.lambda_mwm > 0 or (prop is not None and args.lambda_proposal > 0):
                pooled_rows, pl_owner = [], []
                for i, it in enumerate(batch):
                    A, _w = it[6]
                    if A is not None:
                        pooled_rows.append(A @ hs[i])               # [G_i, d]
                        pl_owner += [i] * A.shape[0]
                if pooled_rows:
                    allp = torch.cat(pooled_rows)
                    owner = torch.tensor(pl_owner)
                    if args.lambda_mwm > 0:
                        wl = aux.weight(allp)                        # one Linear call
                        for i, it in enumerate(batch):
                            A, w = it[6]
                            if A is not None:
                                batch_loss = batch_loss + args.lambda_mwm * ce(wl[owner == i], w)
                    if prop is not None and args.lambda_proposal > 0:
                        ql = prop.weight_logit(allp)                 # one Linear call
                        for i, it in enumerate(batch):
                            A, w = it[6]
                            if A is not None:
                                batch_loss = batch_loss + args.lambda_proposal * ce(ql[owner == i], w)

            batch_loss = batch_loss / len(batch)
            opt.zero_grad(); batch_loss.backward(); opt.step()
        sched.step()
        avg = total / max(len(train_items), 1)
        logger.info("epoch %d/%d loss=%.4f elapsed=%.1fs",
                    epoch + 1, args.epochs, avg, time.time() - t0)
        if args.patience:
            if best_loss - avg > args.min_delta:
                best_loss = avg
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    logger.info("early stop: no >%.4g loss improvement for %d epochs",
                                args.min_delta, args.patience)
                    break

    scorer.eval()
    test_eval = evaluate_split(scorer, test_data, prop=prop, proposal_topk=args.proposal_topk)
    out = {
        "config": vars(args),
        "n_train": len(train_data), "n_test": len(test_data),
        "test": {"foot_accuracy": test_eval.foot_pattern_accuracy,
                 "syllable_accuracy": test_eval.syllable_accuracy,
                 "line_exact_match": test_eval.line_exact_match,
                 "caesura_accuracy": test_eval.caesura_accuracy,
                 "per_book": test_eval.per_book},
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    ckpt = {"encoder": enc.state_dict(), "head": head.state_dict(),
            "aux": aux.state_dict(),
            "text_vocab": tok.text_vocab, "word_vocab": tok.word_vocab,
            "config": vars(args)}
    if prop is not None:
        ckpt["proposal"] = prop.state_dict()
    torch.save(ckpt, args.out.replace(".json", ".pt"))
    logger.info("TEST foot_acc=%.4f", test_eval.foot_pattern_accuracy)


if __name__ == "__main__":
    main()
