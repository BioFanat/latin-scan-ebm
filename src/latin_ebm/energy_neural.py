"""Structured neural energy head over MetricalEncoder outputs.

E(x,y) = E_site + E_syll + E_foot + E_global, each from a small MLP that
reads candidate decisions off the cached per-token states h via atom_indices
pooling. All candidates of a line pool over the SAME h -> Z stays an exact
finite sum (the EBM's core invariant).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from latin_ebm.encoder import TokenizedLine
from latin_ebm.types import (
    CaesuraType, FootType, Parse, PhonWeight, RealizedSyllable,
    SiteChoice, SiteType,
)

# Generic enum -> contiguous index maps (never drift from the type defs).
SITE_CHOICE_ID = {c: i for i, c in enumerate(SiteChoice)}
SITE_TYPE_ID = {t: i for i, t in enumerate(SiteType)}
FOOT_TYPE_ID = {f: i for i, f in enumerate(FootType)}
WEIGHT_ID = {PhonWeight.LONG: 0, PhonWeight.SHORT: 1}
CAESURA_ID = {c: i for i, c in enumerate(CaesuraType)}


def pool_positions(h: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Masked mean of token rows at `positions`. Empty -> zeros."""
    if not positions:
        return torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    idx = torch.tensor(sorted(set(positions)), device=h.device)
    return h.index_select(0, idx).mean(dim=0)


def syllable_token_positions(syl: RealizedSyllable, tl: TokenizedLine) -> list[int]:
    """Token positions for a syllable: its nucleus atom tokens plus the
    flanking bridge tokens (onset/coda material)."""
    positions: list[int] = []
    for ai in syl.atom_indices:
        if 0 <= ai < len(tl.atom_pos):
            positions.append(tl.atom_pos[ai])
            # bridge to the left (index ai-1) and right (index ai)
            if ai - 1 < len(tl.bridge_pos) and ai - 1 >= 0:
                positions.append(tl.bridge_pos[ai - 1])
            if ai < len(tl.bridge_pos):
                positions.append(tl.bridge_pos[ai])
    return positions


def site_token_positions(atom_indices: tuple[int, ...], tl: TokenizedLine) -> list[int]:
    positions: list[int] = []
    for ai in atom_indices:
        if 0 <= ai < len(tl.atom_pos):
            positions.append(tl.atom_pos[ai])
            if ai < len(tl.bridge_pos):
                positions.append(tl.bridge_pos[ai])
            if ai - 1 >= 0 and ai - 1 < len(tl.bridge_pos):
                positions.append(tl.bridge_pos[ai - 1])
    return positions


def _pool_many(h: torch.Tensor, pos_lists: list[list[int]]) -> torch.Tensor:
    """Masked-mean pool MANY position-sets at once via a single matmul.

    Builds a constant averaging matrix A [N, T] where row i has 1/|u| at the
    unique in-range positions u of pos_lists[i] (0 elsewhere), then returns
    A @ h -> [N, d]. This is numerically identical to calling pool_positions
    on each list (mean over sorted-unique positions), but collapses N tiny
    index_select+mean ops into one matmul — the core of the vectorization.
    Empty position-sets yield a zero row.
    """
    T, d = h.shape
    n = len(pos_lists)
    if n == 0:
        return h.new_zeros((0, d))
    a = h.new_zeros((n, T))  # constant (requires_grad=False); grad flows via h
    for i, pos in enumerate(pos_lists):
        u = sorted({p for p in pos if 0 <= p < T})
        if u:
            a[i, u] = 1.0 / len(u)
    return a @ h


def gold_pool(tl: TokenizedLine, gold) -> tuple[torch.Tensor, torch.Tensor]:
    """(A, w_ids) for a parse's syllables: a mean-pooling matrix [G, T] over
    each syllable's token positions, plus its L/S weight targets [G]. Both are
    h-independent (cache once per line) so the MWM/proposal aux forwards can be
    batched across a minibatch with one Linear call instead of a Python loop."""
    n_tokens = tl.n_tokens
    pos = [syllable_token_positions(s, tl) for s in gold.syllables]
    a = torch.zeros((len(pos), n_tokens))
    for i, pl in enumerate(pos):
        u = sorted({p for p in pl if 0 <= p < n_tokens})
        if u:
            a[i, u] = 1.0 / len(u)
    w = torch.tensor([WEIGHT_ID[s.weight] for s in gold.syllables], dtype=torch.long)
    return a, w


@dataclass
class EnergyDecomposition:
    e_site: torch.Tensor
    e_syll: torch.Tensor
    e_foot: torch.Tensor
    e_global: torch.Tensor
    e_total: torch.Tensor


@dataclass
class LinePlan:
    """Precomputed, h-independent structure for one line's candidate set.

    All tensors are constant across training epochs (they depend only on the
    parses and token positions). Built once by StructuredEnergyHead.precompute_plan
    and reused every step; energies_from_plan turns it + the encoder output into
    energies with pure fixed-shape tensor math.
    """
    n: int
    A_site: torch.Tensor | None
    st_ids: torch.Tensor | None
    choice_ids: torch.Tensor | None
    A_syll: torch.Tensor
    w_ids: torch.Tensor
    syl_owner: torch.Tensor
    A_foot: torch.Tensor
    foot_ft: torch.Tensor
    foot_owner: torch.Tensor
    caes: torch.Tensor
    scalars: torch.Tensor


def _mlp(in_dim: int, hidden: int = 32) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class StructuredEnergyHead(nn.Module):
    def __init__(self, d_model: int, dec_dim: int = 8) -> None:
        super().__init__()
        self.decision_emb = nn.Embedding(len(SITE_CHOICE_ID), dec_dim)
        self.sitetype_emb = nn.Embedding(len(SITE_TYPE_ID), dec_dim)
        self.weight_emb = nn.Embedding(len(WEIGHT_ID), dec_dim)
        self.foot_emb = nn.Embedding(len(FOOT_TYPE_ID), dec_dim)
        self.mlp_site = _mlp(d_model + 2 * dec_dim)
        self.mlp_syll = _mlp(d_model + dec_dim)
        self.mlp_foot = _mlp(d_model + dec_dim)
        n_caes = len(CAESURA_ID)
        self.mlp_global = _mlp(d_model + n_caes + 3)  # caesura onehot + bucolic + n_syl + n_spondee
        self.n_caes = n_caes

    def compile_mlps(self) -> None:
        """Opt-in: torch.compile the four MLP sub-modules. Their input dims are
        FIXED (only the row/batch count varies per line), so dynamic=True lets
        Inductor emit one fused kernel reused across line shapes instead of
        recompiling per shape. No-op-safe: failures leave the eager modules."""
        try:
            self.mlp_site = torch.compile(self.mlp_site, dynamic=True)
            self.mlp_syll = torch.compile(self.mlp_syll, dynamic=True)
            self.mlp_foot = torch.compile(self.mlp_foot, dynamic=True)
            self.mlp_global = torch.compile(self.mlp_global, dynamic=True)
        except Exception:
            pass

    def candidate_energy(self, h, h_line, tl: TokenizedLine, line, parse: Parse) -> EnergyDecomposition:
        device = h.device
        # --- E_site ---
        e_site = h.new_zeros(())
        sites_by_idx = {s.index: s for s in line.sites}
        for site in line.sites:
            choice = parse.decisions.get(site.index, site.default)
            pooled = pool_positions(h, site_token_positions(site.atom_indices, tl))
            dec = self.decision_emb(torch.tensor(SITE_CHOICE_ID[choice], device=device))
            st = self.sitetype_emb(torch.tensor(SITE_TYPE_ID[site.site_type], device=device))
            e_site = e_site + self.mlp_site(torch.cat([pooled, dec, st])).squeeze()

        # --- E_syll ---
        e_syll = h.new_zeros(())
        for syl in parse.syllables:
            pooled = pool_positions(h, syllable_token_positions(syl, tl))
            w = self.weight_emb(torch.tensor(WEIGHT_ID[syl.weight], device=device))
            e_syll = e_syll + self.mlp_syll(torch.cat([pooled, w])).squeeze()

        # --- E_foot ---
        e_foot = h.new_zeros(())
        bounds = list(parse.foot_boundaries) + [len(parse.syllables)]
        for fi, ftype in enumerate(parse.foot_types):
            start = parse.foot_boundaries[fi]
            end = bounds[fi + 1]
            positions: list[int] = []
            for syl in parse.syllables[start:end]:
                positions += syllable_token_positions(syl, tl)
            pooled = pool_positions(h, positions)
            ft = self.foot_emb(torch.tensor(FOOT_TYPE_ID[ftype], device=device))
            e_foot = e_foot + self.mlp_foot(torch.cat([pooled, ft])).squeeze()

        # --- E_global ---
        caes = torch.zeros(self.n_caes, device=device)
        caes[CAESURA_ID[parse.caesura]] = 1.0
        n_syl = float(len(parse.syllables))
        n_spondee = float(sum(1 for f in parse.foot_types if f == FootType.SPONDEE))
        scalars = torch.tensor([1.0 if parse.bucolic_diaeresis else 0.0, n_syl, n_spondee], device=device)
        e_global = self.mlp_global(torch.cat([h_line, caes, scalars])).squeeze()

        e_total = e_site + e_syll + e_foot + e_global
        return EnergyDecomposition(e_site, e_syll, e_foot, e_global, e_total)

    def precompute_plan(self, tl: TokenizedLine, line, parses: list[Parse]) -> "LinePlan":
        """Precompute every per-line/per-candidate STRUCTURE tensor that does
        NOT depend on the encoder output h. This is invariant across training
        epochs (parses + token positions never change; only encoder weights do),
        so building it once and reusing it removes the dominant per-step Python
        cost. The returned plan feeds energies_from_plan, whose body is pure
        fixed-shape tensor math (a clean torch.compile target).

        Averaging ('pool') matrices A_* are stored dense: row i has 1/|u| at the
        sorted-unique in-range token positions u of group i, so A @ h reproduces
        the masked-mean pooling exactly (grad flows through h; A is constant).
        """
        n = len(parses)
        T = tl.n_tokens

        def _pool_matrix(pos_lists: list[list[int]]) -> torch.Tensor:
            m = len(pos_lists)
            a = torch.zeros((m, T))
            for i, pos in enumerate(pos_lists):
                u = sorted({p for p in pos if 0 <= p < T})
                if u:
                    a[i, u] = 1.0 / len(u)
            return a

        # --- sites (candidate-invariant pool; per-candidate decision id) ---
        if line.sites:
            A_site = _pool_matrix([site_token_positions(s.atom_indices, tl) for s in line.sites])
            st_ids = torch.tensor([SITE_TYPE_ID[s.site_type] for s in line.sites])
            choice_ids = torch.tensor(
                [[SITE_CHOICE_ID[p.decisions.get(s.index, s.default)] for s in line.sites]
                 for p in parses]
            )  # [n, S]
        else:
            A_site = st_ids = choice_ids = None

        # --- syllables (flattened over candidates, segment-summed by owner) ---
        syl_pos, syl_w, syl_owner = [], [], []
        for ci, p in enumerate(parses):
            for syl in p.syllables:
                syl_pos.append(syllable_token_positions(syl, tl))
                syl_w.append(WEIGHT_ID[syl.weight])
                syl_owner.append(ci)
        A_syll = _pool_matrix(syl_pos) if syl_pos else torch.zeros((0, T))
        w_ids = torch.tensor(syl_w, dtype=torch.long)
        syl_owner_t = torch.tensor(syl_owner, dtype=torch.long)

        # --- feet (flattened over candidates, segment-summed by owner) ---
        foot_pos, foot_ft, foot_owner = [], [], []
        for ci, p in enumerate(parses):
            bounds = list(p.foot_boundaries) + [len(p.syllables)]
            for fi, ftype in enumerate(p.foot_types):
                positions: list[int] = []
                for syl in p.syllables[p.foot_boundaries[fi]:bounds[fi + 1]]:
                    positions += syllable_token_positions(syl, tl)
                foot_pos.append(positions)
                foot_ft.append(FOOT_TYPE_ID[ftype])
                foot_owner.append(ci)
        A_foot = _pool_matrix(foot_pos) if foot_pos else torch.zeros((0, T))
        foot_ft_t = torch.tensor(foot_ft, dtype=torch.long)
        foot_owner_t = torch.tensor(foot_owner, dtype=torch.long)

        # --- global scalars + caesura one-hot (per candidate) ---
        caes = torch.zeros((n, self.n_caes))
        if n:
            caes[torch.arange(n), torch.tensor([CAESURA_ID[p.caesura] for p in parses])] = 1.0
        scalars = torch.zeros((n, 3))
        for ci, p in enumerate(parses):
            scalars[ci, 0] = 1.0 if p.bucolic_diaeresis else 0.0
            scalars[ci, 1] = float(len(p.syllables))
            scalars[ci, 2] = float(sum(1 for f in p.foot_types if f == FootType.SPONDEE))

        return LinePlan(
            n=n, A_site=A_site, st_ids=st_ids, choice_ids=choice_ids,
            A_syll=A_syll, w_ids=w_ids, syl_owner=syl_owner_t,
            A_foot=A_foot, foot_ft=foot_ft_t, foot_owner=foot_owner_t,
            caes=caes, scalars=scalars,
        )

    def energies_from_plan(self, h, h_line, plan: "LinePlan") -> EnergyDecomposition:
        """Pure fixed-shape tensor math from a precomputed LinePlan. No Python
        structure-building, no enum lookups — just A @ h pools, embedding
        lookups, and the four MLPs. Numerically identical to candidate_energies
        (guarded by test_plan_matches_vectorized); exact finite sum preserved."""
        n = plan.n
        if n == 0:
            z = h.new_zeros(0)
            return EnergyDecomposition(z, z, z, z, z)
        d = h.shape[1]

        if plan.A_site is not None:
            s = plan.A_site.shape[0]
            pooled_sites = plan.A_site @ h                       # [S, d]
            st_emb = self.sitetype_emb(plan.st_ids)              # [S, dec]
            dec_emb = self.decision_emb(plan.choice_ids)         # [n, S, dec]
            inp = torch.cat([
                pooled_sites.unsqueeze(0).expand(n, s, d),
                dec_emb,
                st_emb.unsqueeze(0).expand(n, s, -1),
            ], dim=-1)
            e_site = self.mlp_site(inp).squeeze(-1).sum(dim=1)   # [n]
        else:
            e_site = h.new_zeros(n)

        e_syll = h.new_zeros(n)
        if plan.A_syll.shape[0]:
            pooled = plan.A_syll @ h
            w = self.weight_emb(plan.w_ids)
            vals = self.mlp_syll(torch.cat([pooled, w], dim=-1)).squeeze(-1)
            e_syll = e_syll.index_add(0, plan.syl_owner, vals)

        e_foot = h.new_zeros(n)
        if plan.A_foot.shape[0]:
            pooled = plan.A_foot @ h
            ft = self.foot_emb(plan.foot_ft)
            vals = self.mlp_foot(torch.cat([pooled, ft], dim=-1)).squeeze(-1)
            e_foot = e_foot.index_add(0, plan.foot_owner, vals)

        inp = torch.cat([h_line.unsqueeze(0).expand(n, -1), plan.caes, plan.scalars], dim=-1)
        e_global = self.mlp_global(inp).squeeze(-1)

        e_total = e_site + e_syll + e_foot + e_global
        return EnergyDecomposition(e_site, e_syll, e_foot, e_global, e_total)

    def candidate_energies(self, h, h_line, tl: TokenizedLine, line, parses: list[Parse]) -> EnergyDecomposition:
        """Vectorized energy for ALL candidates of a line. Thin wrapper:
        precompute the (h-independent) structure plan, then evaluate it against
        h. Kept as the public entry point so existing callers/tests are
        unchanged; the training loop caches the plan to skip the precompute."""
        return self.energies_from_plan(h, h_line, self.precompute_plan(tl, line, parses))

    def batched_energies(self, hs, h_lines, plans: list["LinePlan"]) -> list[torch.Tensor]:
        """Cross-line batched scansion energies for a whole minibatch.

        Returns a list of per-line e_total tensors ([n_i]) numerically identical
        to [energies_from_plan(hs[i], h_lines[i], plans[i]).e_total for i] —
        guarded by test_batched_energies_matches_per_line. Pooling stays per
        line (a dense block-diagonal would waste ~B x the FLOPs since each row
        is mostly zeros), but every line's groups are CONCATENATED so each MLP /
        embedding / index_add runs ONCE over the minibatch instead of B times.
        index_add places each group at its own GLOBAL candidate index, so
        gradients flow only through that line's h (no cross-line coupling) and
        each line's exact finite sum is preserved.
        """
        B = len(plans)
        ns = [p.n for p in plans]
        N = sum(ns)
        cand_off, acc = [0] * B, 0
        for i in range(B):
            cand_off[i] = acc
            acc += ns[i]
        if N == 0:
            return [hs[i].new_zeros(0) for i in range(B)]
        device = hs[0].device
        e_total = hs[0].new_zeros(N)

        # --- E_site: flatten all (candidate, site) rows across lines ---
        site_inp, site_owner = [], []
        for i, p in enumerate(plans):
            if p.A_site is None or p.n == 0:
                continue
            s = p.A_site.shape[0]
            pooled = p.A_site @ hs[i]                       # [S, d]
            d = pooled.shape[1]
            st = self.sitetype_emb(p.st_ids)               # [S, dec]
            dec = self.decision_emb(p.choice_ids)          # [n, S, dec]
            inp = torch.cat([
                pooled.unsqueeze(0).expand(p.n, s, d),
                dec,
                st.unsqueeze(0).expand(p.n, s, -1),
            ], dim=-1).reshape(p.n * s, -1)
            site_inp.append(inp)
            site_owner.append((cand_off[i] + torch.arange(p.n, device=device)).repeat_interleave(s))
        if site_inp:
            vals = self.mlp_site(torch.cat(site_inp)).squeeze(-1)
            e_total = e_total.index_add(0, torch.cat(site_owner), vals)

        # --- E_syll: flatten all (candidate, syllable) rows across lines ---
        syll_pooled, syll_w, syll_owner = [], [], []
        for i, p in enumerate(plans):
            if p.A_syll.shape[0] == 0:
                continue
            syll_pooled.append(p.A_syll @ hs[i])
            syll_w.append(p.w_ids)
            syll_owner.append(cand_off[i] + p.syl_owner)
        if syll_pooled:
            w = self.weight_emb(torch.cat(syll_w))
            vals = self.mlp_syll(torch.cat([torch.cat(syll_pooled), w], dim=-1)).squeeze(-1)
            e_total = e_total.index_add(0, torch.cat(syll_owner), vals)

        # --- E_foot: flatten all (candidate, foot) rows across lines ---
        foot_pooled, foot_ft, foot_owner = [], [], []
        for i, p in enumerate(plans):
            if p.A_foot.shape[0] == 0:
                continue
            foot_pooled.append(p.A_foot @ hs[i])
            foot_ft.append(p.foot_ft)
            foot_owner.append(cand_off[i] + p.foot_owner)
        if foot_pooled:
            ft = self.foot_emb(torch.cat(foot_ft))
            vals = self.mlp_foot(torch.cat([torch.cat(foot_pooled), ft], dim=-1)).squeeze(-1)
            e_total = e_total.index_add(0, torch.cat(foot_owner), vals)

        # --- E_global: per-candidate, concatenated across lines in order ---
        g_inp, g_owner = [], []
        for i, p in enumerate(plans):
            if p.n == 0:
                continue
            g_inp.append(torch.cat(
                [h_lines[i].unsqueeze(0).expand(p.n, -1), p.caes, p.scalars], dim=-1))
            g_owner.append(cand_off[i] + torch.arange(p.n, device=device))
        if g_inp:
            vals = self.mlp_global(torch.cat(g_inp)).squeeze(-1)
            e_total = e_total.index_add(0, torch.cat(g_owner), vals)

        return [e_total[cand_off[i]:cand_off[i] + ns[i]] for i in range(B)]


class NeuralScorer(nn.Module):
    """Encode a line once, score all candidates by pooling over the same h."""

    def __init__(self, encoder, head: StructuredEnergyHead, tokenizer) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.tokenizer = tokenizer

    def energies(self, line, candidates: list[Parse], text_override=None) -> torch.Tensor:
        tl = self.tokenizer.encode(line)
        h, h_line = self.encoder(tl, text_override=text_override)  # ONE encode
        # Vectorized: all candidates scored in a handful of big ops over the
        # same cached h. Exact finite sum preserved (-> Z stays exact).
        return self.head.candidate_energies(h, h_line, tl, line, candidates).e_total

    def predict(self, line, candidates: list[Parse]) -> int:
        with torch.no_grad():
            e = self.energies(line, candidates)
        return int(e.argmin().item())


class AuxHeads(nn.Module):
    """Self-supervised auxiliary heads sharing the encoder."""

    def __init__(self, d_model: int, text_vocab_size: int) -> None:
        super().__init__()
        self.mlm = nn.Linear(d_model, text_vocab_size)
        self.weight = nn.Linear(d_model, 2)  # LONG=0, SHORT=1
