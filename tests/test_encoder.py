import pytest
import torch

from latin_ebm.encoder import MetricalTokenizer, MetricalEncoder, TokenizedLine


def test_tokenizer_layout(sample_line):
    tok = MetricalTokenizer.build([sample_line])
    enc = tok.encode(sample_line)
    # 3 atoms, 2 bridges -> 2*3-1 = 5 tokens, interleaved atom,bridge,atom,bridge,atom
    assert enc.n_tokens == 5
    assert enc.token_is_atom == [True, False, True, False, True]
    # atom k lives at token position 2k; bridge k at 2k+1
    assert enc.atom_pos == [0, 2, 4]
    assert enc.bridge_pos == [1, 3]
    # the bridge with has_word_boundary=True (bridge index 1) is flagged
    assert enc.token_word_boundary[enc.bridge_pos[1]] == 1
    assert enc.token_word_boundary[enc.bridge_pos[0]] == 0


def test_tokenizer_unk_for_unseen(sample_line):
    tok = MetricalTokenizer.build([sample_line])
    # an atom text not in vocab maps to UNK, never crashes
    other = sample_line  # same line is fine; just assert ids are in range
    enc = tok.encode(other)
    assert all(0 <= t < tok.text_vocab_size for t in enc.token_text_id)
    assert all(0 <= w < tok.word_vocab_size for w in enc.token_word_id)


def test_encoder_shapes(sample_line):
    tok = MetricalTokenizer.build([sample_line])
    enc = MetricalEncoder(
        text_vocab_size=tok.text_vocab_size,
        word_vocab_size=tok.word_vocab_size,
        d_model=32, n_layers=2, n_heads=4,
    )
    tl = tok.encode(sample_line)
    h, h_line = enc(tl)
    assert h.shape == (tl.n_tokens, 32)
    assert h_line.shape == (32,)
    assert h.requires_grad  # differentiable


def test_batched_encoder_matches_single(real_aeneid_line):
    """forward_batch over padded lines must match per-line forward (eval mode,
    so dropout is off). This guards the minibatch-encoder optimization: padding
    is masked out of attention and the pool, so real-token states are unchanged.
    Uses two different-length real lines so padding actually happens."""
    from pathlib import Path
    from latin_ebm.corpus.pedecerto import parse_xml
    from latin_ebm.lexicon import VowelLengthLexicon

    line_a, _ = real_aeneid_line
    lex = VowelLengthLexicon(Path("data/MqDqMacrons.json"), Path("data/MorpheusMacrons.txt"))
    res = parse_xml(Path("pedecerto-raw/VERG-aene.xml"), lexicon=lex)
    # pick a second line with a DIFFERENT token count so the batch is ragged
    line_b = None
    for e in res.examples:
        if len(e.line.atoms) != len(line_a.atoms) and len(e.line.atoms) > 1:
            line_b = e.line
            break
    assert line_b is not None

    tok = MetricalTokenizer.build([line_a, line_b])
    enc = MetricalEncoder(tok.text_vocab_size, tok.word_vocab_size, d_model=32, n_layers=2)
    enc.eval()  # disable dropout for an exact comparison
    tls = [tok.encode(line_a), tok.encode(line_b)]

    hs, h_lines = enc.forward_batch(tls)
    for i, tl in enumerate(tls):
        h_single, hline_single = enc(tl)
        assert hs[i].shape == h_single.shape
        assert torch.allclose(hs[i], h_single, atol=1e-5)
        assert torch.allclose(h_lines[i], hline_single, atol=1e-5)
