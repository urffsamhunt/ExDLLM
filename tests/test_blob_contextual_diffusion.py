#!/usr/bin/env python3
"""
Unit tests for Blob-Restricted Contextual Diffusion Token Prediction (BRCD):
  1. Blob candidate selector shapes, cosine ranking, and active-set target guarantee.
  2. GenHead contextual sequence projection and candidate-restricted local projection.
  3. DSBHybrid loss and diagnostic tracking for blob_hit and blob_acc.
  4. End-to-end generate_text with blob diffusion and full backward compatibility.
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
    select_blob_candidates,
    KEEP,
    DELETE,
    REPLACE,
    INSERT,
    EXPAND,
    NUM_TAGS,
)


class DummyTokenizer:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.mask_token_id = 3
        self.id_to_str = {
            0: "<s>", 1: "</s>", 2: "<pad>", 3: "<mask_id>",
            10: "The", 11: "capital", 12: "of", 13: "France", 14: "is", 15: "Paris",
            20: "Pakistan", 21: "Islamabad", 22: "Lahore",
            30: "India", 31: "Delhi", 32: "Punjab"
        }
        self.str_to_id = {v: k for k, v in self.id_to_str.items()}

    def encode(self, text, add_special_tokens=False):
        tokens = text.strip().split()
        res = []
        for t in tokens:
            if t in self.str_to_id:
                res.append(self.str_to_id[t])
            else:
                res.append((hash(t) % 400) + 50)
        return res

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join([self.id_to_str.get(t, f"tok_{t}") for t in token_ids])


class DummyEmbedder(nn.Module):
    def __init__(self, dim=64, vocab_size=500):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_length = 32
        self.embedding = nn.Embedding(vocab_size, dim)
        self.encoder = self
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids=None, attention_mask=None):
        out = type('Obj', (object,), {})()
        out.last_hidden_state = self.embedding(input_ids)
        return out

    def get_input_embeddings(self):
        return self.embedding

    def embed_ids(self, input_ids, attention_mask=None):
        return self.embedding(input_ids)

    def decode_logits(self, hidden_states):
        return self.lm_head(hidden_states)


class TestBlobContextualDiffusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.vocab_size = 500
        self.blob_size = 32
        self.score_net = MLPScoreNet(dim=self.dim, hidden_dim=128, num_layers=2)
        self.bridge = DiffSchrodingerBridge(dim=self.dim, score_net=self.score_net, num_steps=10)
        self.tokenizer = DummyTokenizer(vocab_size=self.vocab_size)
        self.embedder = DummyEmbedder(dim=self.dim, vocab_size=self.vocab_size)

    def test_blob_candidate_selector_shapes_and_ranking(self):
        """Test candidate selection shape and top cosine similarity ranking."""
        x = torch.randn(8, self.dim)
        w = torch.randn(self.vocab_size, self.dim)

        cands = select_blob_candidates(x, w, blob_size=self.blob_size)
        self.assertEqual(cands.shape, (8, self.blob_size))
        self.assertTrue((cands >= 0).all())
        self.assertTrue((cands < self.vocab_size).all())

        # Verify that for the nearest token, it is in candidate blob
        norm_x = F.normalize(x, dim=-1)
        norm_w = F.normalize(w, dim=-1)
        sims = norm_x @ norm_w.T
        top1 = sims.argmax(dim=-1)
        for i in range(8):
            self.assertIn(top1[i].item(), cands[i].tolist())

    def test_blob_candidate_selector_active_set_target_guarantee(self):
        """Test that ground-truth targets are guaranteed to be in the active candidate set."""
        x = torch.randn(10, self.dim)
        w = torch.randn(self.vocab_size, self.dim)

        # Deliberately pick targets with the lowest cosine similarity
        norm_x = F.normalize(x, dim=-1)
        norm_w = F.normalize(w, dim=-1)
        sims = norm_x @ norm_w.T
        worst_targets = sims.argmin(dim=-1)

        # Without targets, the worst targets should NOT be in top blob_size
        cands_no_tgt = select_blob_candidates(x, w, blob_size=self.blob_size)
        for i in range(10):
            self.assertNotIn(worst_targets[i].item(), cands_no_tgt[i].tolist())

        # With targets, every row MUST contain its target
        cands_with_tgt = select_blob_candidates(x, w, blob_size=self.blob_size, targets=worst_targets)
        self.assertEqual(cands_with_tgt.shape, (10, self.blob_size))
        for i in range(10):
            self.assertIn(worst_targets[i].item(), cands_with_tgt[i].tolist())

    def test_genhead_contextual_and_candidate_projections(self):
        """Test GenHead forward over candidate blobs and contextual injection."""
        head = GenHead(
            dim=self.dim,
            vocab_size=self.vocab_size,
            embed_weight=self.embedder.embedding.weight,
            contextual_gen=True,
            subspace_factorization=True,
            macro_dim=48,
            lexical_dim=16,
            angular_margin=0.05,
        )

        n_slots = 6
        x = torch.randn(n_slots, self.dim)
        t = torch.rand(n_slots)
        c = torch.randn(n_slots, self.dim)
        cand_ids = torch.randint(0, self.vocab_size, (n_slots, self.blob_size))
        targets = cand_ids[:, 0]

        # 1. Full vocab projection when candidate_ids is None
        full_logits = head(x, t, cond=c)
        self.assertEqual(full_logits.shape, (n_slots, self.vocab_size))

        # 2. Candidate restricted projection
        blob_logits = head(x, t, cond=c, candidate_ids=cand_ids)
        self.assertEqual(blob_logits.shape, (n_slots, self.blob_size))

        # 3. Contextual projection zero-init backward compatibility:
        # ctx_proj is initialized with zero weight & bias, so passing context gives exact same result
        head.eval()
        logits_with_ctx = head(x, t, cond=c, candidate_ids=cand_ids, context=c)
        logits_without_ctx = head(x, t, cond=c, candidate_ids=cand_ids, context=None)
        torch.testing.assert_close(logits_with_ctx, logits_without_ctx)

        # When ctx_proj has non-zero weights, context influences features
        with torch.no_grad():
            head.ctx_proj.weight.fill_(0.5)
        logits_modified = head(x, t, cond=c, candidate_ids=cand_ids, context=c)
        self.assertFalse(torch.allclose(logits_modified, logits_without_ctx))

        # 4. Angular margin on local target
        margin_logits = head(x, t, cond=c, targets=targets, candidate_ids=cand_ids)
        self.assertEqual(margin_logits.shape, (n_slots, self.blob_size))

    def test_dsb_hybrid_loss_with_blob_diffusion(self):
        """Test DSBHybrid training forward pass with blob diffusion enabled."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            embed_weight=self.embedder.embedding.weight,
            condition_heads=True,
            time_embed_dim=32,
            blob_diffusion=True,
            blob_size=self.blob_size,
            contextual_gen=True,
        )

        B, S = 2, 8
        noisy_ids = torch.randint(4, 100, (B, S))
        clean_ids = torch.randint(4, 100, (B, S))
        attn = torch.ones(B, S)

        dp1 = self.embedder.embed_ids(noisy_ids, attn)
        dp2 = self.embedder.embed_ids(clean_ids, attn)

        tag_labels = torch.zeros(B, S, dtype=torch.long)
        tag_labels[:, 2] = REPLACE
        tag_labels[:, 5] = INSERT

        gen_labels = torch.full((B, S), -100, dtype=torch.long)
        gen_labels[:, 2] = clean_ids[:, 2]
        gen_labels[:, 5] = clean_ids[:, 5]

        loss, res_dict = hybrid.loss(
            dp1, dp2, clean_ids, noisy_ids, attn,
            tag_labels=tag_labels,
            gen_labels=gen_labels,
            expose_ratio=0.0,
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("blob_hit", res_dict)
        self.assertIn("blob_acc", res_dict)
        self.assertGreaterEqual(res_dict["blob_hit"], 0.0)
        self.assertLessEqual(res_dict["blob_hit"], 100.0)

    def test_dsb_hybrid_diagnostics_with_blob_diffusion(self):
        """Test compute_diagnostics returns blob_hit and blob_acc."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            embed_weight=self.embedder.embedding.weight,
            blob_diffusion=True,
            blob_size=self.blob_size,
            contextual_gen=True,
        )

        B, S = 2, 8
        clean_ids = torch.randint(4, 100, (B, S))
        noisy_ids = clean_ids.clone()
        noisy_ids[:, 3] = self.tokenizer.mask_token_id

        dp1 = self.embedder.embed_ids(noisy_ids)
        dp2 = self.embedder.embed_ids(clean_ids)

        tag_labels = torch.zeros(B, S, dtype=torch.long)
        tag_labels[:, 3] = REPLACE
        gen_labels = torch.full((B, S), -100, dtype=torch.long)
        gen_labels[:, 3] = clean_ids[:, 3]

        with torch.no_grad():
            diag = hybrid.compute_diagnostics(
                dp1, dp2, tag_labels, gen_labels,
                recon_steps=2,
                embed_weight=self.embedder.embedding.weight,
            )

        self.assertIn("blob_hit", diag)
        self.assertIn("blob_acc", diag)
        self.assertGreaterEqual(diag["blob_hit"], 0.0)
        self.assertLessEqual(diag["blob_hit"], 100.0)

    def test_dsb_hybrid_generate_text_with_blob_diffusion(self):
        """Test variable-length iterative text generation with blob diffusion active."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            embed_weight=self.embedder.embedding.weight,
            blob_diffusion=True,
            blob_size=self.blob_size,
            contextual_gen=True,
        )

        prompt_ids = [self.tokenizer.bos_token_id, 10, 11, 12, 13, 14, 3, self.tokenizer.eos_token_id]
        ids_t = torch.tensor([prompt_ids])
        x = self.embedder.embed_ids(ids_t)

        out_texts = hybrid.generate_text(
            x,
            self.tokenizer,
            self.embedder,
            dp1=x,
            seed_ids=[prompt_ids],
            max_iterations=2,
            blob_diffusion=True,
            blob_size=self.blob_size,
        )

        self.assertIsInstance(out_texts, list)
        self.assertEqual(len(out_texts), 1)
        self.assertIsInstance(out_texts[0], str)
        self.assertGreater(len(out_texts[0]), 0)


if __name__ == "__main__":
    unittest.main()
