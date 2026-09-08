#!/usr/bin/env python3
"""
Unit tests for enhanced word sampling and discrete edits in DSBHybrid:
  1. Aligned feature-space distance gating.
  2. Anchor stutter suppression on INSERT slots.
  3. Edit-operation conditioning in GenHead.
  4. Stopword-exempt and local-window repetition penalty.
  5. Dual-head logit blending for factual retrieval.
  6. Progressive in-filling for multi-token spans.
  7. Checkpoint loading backward compatibility.
"""

import math
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dllm.dsb import DiffSchrodingerBridge, MLPScoreNet
from dllm.dsb_hybrid import (
    DSBHybrid,
    GenHead,
    TaggerHead,
    KEEP,
    DELETE,
    REPLACE,
    INSERT,
    EXPAND,
    NUM_TAGS,
)


class MockModule(nn.Module):
    def __init__(self, fn, cond_dim=0):
        super().__init__()
        self.fn = fn
        self.cond_dim = cond_dim

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class DummyTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.mask_token_id = 3
        self.the_id = 10
        self.france_id = 50
        self.paris_id = 51
        self.id_to_str = {
            0: "<s>", 1: "</s>", 2: "<pad>", 3: "<mask>",
            10: "the", 11: "of", 12: "is", 50: "France", 51: "Paris"
        }
        self.str_to_id = {v: k for k, v in self.id_to_str.items()}

    def encode(self, text, add_special_tokens=False):
        t = text.strip()
        if t in self.str_to_id:
            return [self.str_to_id[t]]
        return [hash(t) % 500 + 100]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join([self.id_to_str.get(t, f"tok_{t}") for t in token_ids])


class DummyEmbedder(nn.Module):
    def __init__(self, dim=64, vocab_size=1000):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_length = 32
        self.embedding = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def embed_ids(self, input_ids, attention_mask):
        return self.embedding(input_ids)

    def decode_logits(self, hidden_states):
        return self.lm_head(hidden_states)


class TestDSBHybridSampling(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.vocab_size = 1000
        self.score_net = MLPScoreNet(dim=self.dim, hidden_dim=128, num_layers=2)
        self.bridge = DiffSchrodingerBridge(dim=self.dim, score_net=self.score_net, num_steps=10)
        self.tokenizer = DummyTokenizer(vocab_size=self.vocab_size)
        self.embedder = DummyEmbedder(dim=self.dim, vocab_size=self.vocab_size)

    def test_genhead_features_and_op_conditioning(self):
        """Test GenHead returns aligned features and supports op conditioning."""
        # Unconditioned GenHead (op_embed_dim = 0)
        gh_standard = GenHead(dim=self.dim, vocab_size=self.vocab_size, op_embed_dim=0)
        x = torch.randn(4, self.dim)
        t = torch.zeros(4)
        out = gh_standard(x, t, return_features=True)
        self.assertIsInstance(out, tuple)
        logits, feat = out
        self.assertEqual(logits.shape, (4, self.vocab_size))
        self.assertEqual(feat.shape, (4, self.dim))

        # Operation-conditioned GenHead (op_embed_dim = 16)
        gh_op = GenHead(dim=self.dim, vocab_size=self.vocab_size, op_embed_dim=16)
        op_ids = torch.tensor([REPLACE, INSERT, REPLACE, INSERT], dtype=torch.long)
        logits_op, feat_op = gh_op(x, t, op_ids=op_ids, return_features=True)
        self.assertEqual(logits_op.shape, (4, self.vocab_size))
        self.assertEqual(feat_op.shape, (4, self.dim))

    def test_sample_topk_aligned_distance_gating(self):
        """Test _sample_topk prunes tokens using aligned feat_states and decoder_weight."""
        logits = torch.randn(2, self.vocab_size)
        feat = torch.randn(2, self.dim)
        w_dec = torch.randn(self.vocab_size, self.dim)

        norm_f = F.normalize(feat, dim=-1)
        norm_w = F.normalize(w_dec, dim=-1)
        cos = norm_f @ norm_w.T  # (2, V)
        thresh = min(cos[0].max().item(), cos[1].max().item()) - 0.05

        sampled = DSBHybrid._sample_topk(
            logits.clone(),
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            feat_states=feat,
            decoder_weight=w_dec,
            distance_threshold=thresh,
        )
        self.assertEqual(len(sampled), 2)
        for i, tok in enumerate(sampled):
            self.assertGreaterEqual(cos[i, tok].item() + 1e-4, thresh)

    def test_sample_topk_stopword_and_local_window_exemption(self):
        """Test repetition penalty respects exempt_tokens and local windows."""
        # 1 slot, 1000 vocab
        logits = torch.zeros(1, 100)
        the_tok = self.tokenizer.the_id
        paris_tok = self.tokenizer.paris_id

        # Make both tokens have positive logit
        logits[0, the_tok] = 10.0
        logits[0, paris_tok] = 10.0

        # Penalize [the_tok, paris_tok] with 'the_tok' in exempt_tokens
        penalized = [[the_tok, paris_tok]]
        exempt = {the_tok}

        sampled = DSBHybrid._sample_topk(
            logits.clone(),
            temperature=0.0,  # argmax
            top_k=5,
            top_p=1.0,
            generated_ids=penalized,
            repetition_penalty=2.0,
            exempt_tokens=exempt,
        )
        # the_tok should NOT be penalized (stays 10.0), paris_tok is penalized (10.0 / 2 = 5.0)
        # Thus argmax should be the_tok!
        self.assertEqual(sampled[0], the_tok)

    def test_anchor_stutter_suppression_on_insert(self):
        """Test that an INSERT op prevents generating the anchor token itself."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            condition_heads=False,
            time_embed_dim=0,
            embed_weight=self.embedder.embedding.weight,
        )

        canvas_tokens = [self.tokenizer.bos_token_id, self.tokenizer.france_id, self.tokenizer.eos_token_id]
        cur = torch.randn(1, len(canvas_tokens), self.dim)

        # Mock tagger to predict INSERT at pos 1
        def mock_tagger_fn(emb, t, cond=None):
            t_logits = torch.zeros(1, len(canvas_tokens), NUM_TAGS)
            t_logits[0, 0, KEEP] = 10.0
            t_logits[0, 1, INSERT] = 10.0
            t_logits[0, 2, KEEP] = 10.0
            return t_logits

        hybrid.tagger = MockModule(mock_tagger_fn)

        # Mock generator to strongly favor France (anchor token)
        def mock_generator_fn(x, t, cond=None, op_ids=None, return_features=False):
            gl = torch.zeros(x.shape[0], self.vocab_size)
            gl[:, self.tokenizer.france_id] = 20.0  # anchor token strongly favored
            gl[:, self.tokenizer.paris_id] = 15.0   # valid second choice
            feat = torch.randn(x.shape[0], self.dim)
            if return_features:
                return gl, feat
            return gl

        hybrid.generator = MockModule(mock_generator_fn)

        # Run 1 iteration of generate_text
        decoded = hybrid.generate_text(
            cur, self.tokenizer, self.embedder,
            max_iterations=1,
            seed_ids=[canvas_tokens],
            temperature=0.0,  # argmax
        )
        self.assertTrue(len(decoded) == 1)
        # Should NOT contain two Frances: "France France"
        words = decoded[0].split()
        self.assertNotIn("France France", decoded[0])
        # Second choice "Paris" should have been chosen before "France"
        self.assertIn("Paris", words)

    def test_dual_head_logit_blending(self):
        """Test lm_blend_weight blends MLM head logits on REPLACE slots."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            condition_heads=False,
            time_embed_dim=0,
            embed_weight=self.embedder.embedding.weight,
        )

        canvas_tokens = [self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id]
        cur = torch.randn(1, len(canvas_tokens), self.dim)

        def mock_tagger_fn(emb, t, cond=None):
            t_logits = torch.zeros(1, len(canvas_tokens), NUM_TAGS)
            t_logits[0, 1, REPLACE] = 10.0
            return t_logits

        hybrid.tagger = MockModule(mock_tagger_fn)

        # GenHead predicts token 100, LM Head predicts token 51 ("Paris")
        def mock_generator_fn(x, t, cond=None, op_ids=None, return_features=False):
            gl = torch.zeros(x.shape[0], self.vocab_size)
            gl[:, 100] = 10.0
            feat = torch.randn(x.shape[0], self.dim)
            if return_features:
                return gl, feat
            return gl

        hybrid.generator = MockModule(mock_generator_fn)

        def mock_decode_logits(h):
            logits = torch.zeros(h.shape[0], self.vocab_size)
            logits[:, self.tokenizer.paris_id] = 30.0  # Strong MLM knowledge
            return logits

        self.embedder.decode_logits = mock_decode_logits

        # With lm_blend_weight = 0.5, Paris (30.0 * 0.5 = 15.0) beats token 100 (10.0 * 0.5 = 5.0)
        decoded = hybrid.generate_text(
            cur, self.tokenizer, self.embedder,
            max_iterations=1,
            seed_ids=[canvas_tokens],
            temperature=0.0,
            lm_blend_weight=0.5,
        )
        self.assertIn("Paris", decoded[0])

    def test_progressive_fill_multitoken(self):
        """Test progressive_fill resolves multi-token spans without error."""
        canvas_tokens = [self.tokenizer.bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.mask_token_id, self.tokenizer.eos_token_id]
        cur = torch.randn(1, len(canvas_tokens), self.dim)

        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            embed_weight=self.embedder.embedding.weight,
        )

        def mock_tagger_fn(emb, t, cond=None):
            t_logits = torch.zeros(1, len(canvas_tokens), NUM_TAGS)
            t_logits[0, 1, REPLACE] = 10.0
            t_logits[0, 2, REPLACE] = 10.0
            return t_logits

        hybrid.tagger = MockModule(mock_tagger_fn)

        decoded = hybrid.generate_text(
            cur, self.tokenizer, self.embedder,
            max_iterations=1,
            seed_ids=[canvas_tokens],
            temperature=0.0,
            progressive_fill=True,
        )
        self.assertEqual(len(decoded), 1)

    def test_checkpoint_backward_compatibility(self):
        """Test that legacy state dicts without op_emb load cleanly with strict=False."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            op_embed_dim=16,
        )
        # Create legacy state dict without generator.op_emb
        legacy_sd = {k: v for k, v in hybrid.state_dict().items() if "generator.op_emb" not in k}
        # strict=False should load without raising KeyError
        incompatible = hybrid.load_state_dict(legacy_sd, strict=False)
        self.assertIn("generator.op_emb.weight", incompatible.missing_keys)


if __name__ == "__main__":
    unittest.main()
