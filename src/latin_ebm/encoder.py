"""Metrical encoder: tokenize a LatinLine into the candidate-invariant
atom/bridge skeleton, then a small Transformer over those tokens.

Token layout convention: tokens interleave atoms and bridges in surface
order. Atom k -> token position 2k; bridge k -> token position 2k+1.
n_tokens = 2 * len(atoms) - 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from latin_ebm.types import LatinLine

PAD, UNK, MASK = "<pad>", "<unk>", "<mask>"
ATOM_KIND, BRIDGE_KIND = 0, 1


@dataclass
class TokenizedLine:
    n_tokens: int
    token_text_id: list[int]
    token_kind: list[int]
    token_word_id: list[int]
    token_word_boundary: list[int]
    token_is_atom: list[bool]
    atom_pos: list[int]     # atom_index -> token position
    bridge_pos: list[int]   # bridge_index -> token position


class MetricalTokenizer:
    """Builds vocabularies and encodes LatinLine -> TokenizedLine."""

    def __init__(self, text_vocab: dict[str, int], word_vocab: dict[str, int]) -> None:
        self.text_vocab = text_vocab
        self.word_vocab = word_vocab

    @classmethod
    def build(cls, lines: list[LatinLine]) -> "MetricalTokenizer":
        text_vocab = {PAD: 0, UNK: 1, MASK: 2}
        word_vocab = {PAD: 0, UNK: 1}
        for line in lines:
            for atom in line.atoms:
                text_vocab.setdefault(atom.chars, len(text_vocab))
            for bridge in line.bridges:
                text_vocab.setdefault("|" + bridge.chars, len(text_vocab))  # prefix avoids atom/bridge collision
            for w in line.words:
                word_vocab.setdefault(w, len(word_vocab))
        return cls(text_vocab, word_vocab)

    @property
    def text_vocab_size(self) -> int:
        return len(self.text_vocab)

    @property
    def word_vocab_size(self) -> int:
        return len(self.word_vocab)

    @property
    def mask_id(self) -> int:
        return self.text_vocab[MASK]

    def _text_id(self, s: str) -> int:
        return self.text_vocab.get(s, self.text_vocab[UNK])

    def _word_id(self, w: str) -> int:
        return self.word_vocab.get(w, self.word_vocab[UNK])

    def encode(self, line: LatinLine) -> TokenizedLine:
        n_atoms = len(line.atoms)
        n_tokens = max(2 * n_atoms - 1, 1)
        text_id = [0] * n_tokens
        kind = [0] * n_tokens
        word_id = [0] * n_tokens
        wb = [0] * n_tokens
        is_atom = [False] * n_tokens
        atom_pos = [2 * k for k in range(n_atoms)]
        bridge_pos = [2 * k + 1 for k in range(len(line.bridges))]

        for k, atom in enumerate(line.atoms):
            p = atom_pos[k]
            text_id[p] = self._text_id(atom.chars)
            kind[p] = ATOM_KIND
            word_id[p] = self._word_id(line.words[atom.word_idx]) if line.words else self.word_vocab[UNK]
            is_atom[p] = True
        for k, bridge in enumerate(line.bridges):
            p = bridge_pos[k]
            text_id[p] = self._text_id("|" + bridge.chars)
            kind[p] = BRIDGE_KIND
            word_id[p] = self.word_vocab[PAD]  # bridges have no single word
            wb[p] = 1 if bridge.has_word_boundary else 0

        return TokenizedLine(
            n_tokens=n_tokens, token_text_id=text_id, token_kind=kind,
            token_word_id=word_id, token_word_boundary=wb, token_is_atom=is_atom,
            atom_pos=atom_pos, bridge_pos=bridge_pos,
        )


class MetricalEncoder(nn.Module):
    """Transformer over the atom/bridge token sequence. Encodes ONE line."""

    def __init__(
        self, text_vocab_size: int, word_vocab_size: int,
        d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
        dropout: float = 0.1, max_len: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.text_emb = nn.Embedding(text_vocab_size, d_model)
        self.kind_emb = nn.Embedding(2, d_model)
        self.word_emb = nn.Embedding(word_vocab_size, d_model)
        self.wb_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_query = nn.Parameter(torch.randn(d_model))

    def forward(self, tl: "TokenizedLine", text_override: list[int] | None = None):
        """Returns (h: [T, d], h_line: [d]). text_override allows MLM masking."""
        device = self.text_emb.weight.device
        text_ids = torch.tensor(text_override or tl.token_text_id, device=device)
        kind_ids = torch.tensor(tl.token_kind, device=device)
        word_ids = torch.tensor(tl.token_word_id, device=device)
        wb_ids = torch.tensor(tl.token_word_boundary, device=device)
        pos = torch.arange(tl.n_tokens, device=device)
        x = (
            self.text_emb(text_ids) + self.kind_emb(kind_ids)
            + self.word_emb(word_ids) + self.wb_emb(wb_ids) + self.pos_emb(pos)
        )
        h = self.transformer(x.unsqueeze(0)).squeeze(0)  # [T, d]
        # attention pool: softmax(h @ query) weighted sum
        scores = h @ self.pool_query  # [T]
        weights = torch.softmax(scores, dim=0)
        h_line = (weights.unsqueeze(-1) * h).sum(dim=0)  # [d]
        return h, h_line

    def forward_batch(
        self,
        tls: list["TokenizedLine"],
        text_overrides: list[list[int] | None] | None = None,
    ):
        """Encode B lines in ONE padded transformer call (vs B single calls).

        Returns (hs, h_lines):
          hs       : list of per-line [T_i, d] tensors (padding sliced off)
          h_lines  : [B, d] attention-pooled line vectors
        Padded positions are masked out of self-attention via key_padding_mask
        and excluded from the attention pool, so each line's real-token states
        are numerically identical (in eval mode) to forward(tl) — guarded by
        test_batched_encoder_matches_single. This batches only the encoder; the
        structured head still scores each line over its own h.
        """
        device = self.text_emb.weight.device
        b = len(tls)
        lengths = [tl.n_tokens for tl in tls]
        max_t = max(lengths)

        def _pad(seq: list[int]) -> list[int]:
            return seq + [0] * (max_t - len(seq))

        text_rows, kind_rows, word_rows, wb_rows = [], [], [], []
        for i, tl in enumerate(tls):
            override = text_overrides[i] if text_overrides is not None else None
            text_rows.append(_pad(list(override) if override is not None else list(tl.token_text_id)))
            kind_rows.append(_pad(list(tl.token_kind)))
            word_rows.append(_pad(list(tl.token_word_id)))
            wb_rows.append(_pad(list(tl.token_word_boundary)))

        text_ids = torch.tensor(text_rows, device=device)   # [B, maxT]
        kind_ids = torch.tensor(kind_rows, device=device)
        word_ids = torch.tensor(word_rows, device=device)
        wb_ids = torch.tensor(wb_rows, device=device)
        pos = torch.arange(max_t, device=device).unsqueeze(0).expand(b, max_t)
        x = (
            self.text_emb(text_ids) + self.kind_emb(kind_ids)
            + self.word_emb(word_ids) + self.wb_emb(wb_ids) + self.pos_emb(pos)
        )  # [B, maxT, d]

        # True = position is padding -> ignored by attention.
        key_padding_mask = torch.zeros((b, max_t), dtype=torch.bool, device=device)
        for i, n in enumerate(lengths):
            if n < max_t:
                key_padding_mask[i, n:] = True

        h_pad = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [B, maxT, d]

        # Masked attention pool: -inf on padded positions before softmax.
        scores = h_pad @ self.pool_query  # [B, maxT]
        scores = scores.masked_fill(key_padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, maxT, 1]
        h_lines = (weights * h_pad).sum(dim=1)  # [B, d]

        hs = [h_pad[i, : lengths[i]] for i in range(b)]
        return hs, h_lines
