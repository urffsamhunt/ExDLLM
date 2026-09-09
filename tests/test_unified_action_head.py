#!/usr/bin/env python3
"""
Unit tests for Paradigm C: Unified Action Space (Single-Head Homogeneous Architecture)
and Speculative Subword Unrolling in DSBHybrid.

Test suite covers:
  1. UnifiedEditHead dimensions, projection weights, and logits concatenation (V + 3).
  2. build_unified_targets target mapping from discrete tags + tokens to V*.
  3. Unified cross-entropy loss computation and end-to-end gradient flow.
  4. Speculative subword unrolling (_unroll_subwords) reproducing multi-token entities
     (e.g. [" Islam", "abad"] vs single-token [" Karachi"]).
  5. Variable-length multi-subword splice in _apply_edits.
  6. End-to-end generate_text in unified head mode.
"""

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
    UnifiedEditHead,
    ACTION_OFFSET_KEEP,
    ACTION_OFFSET_DELETE,
    ACTION_OFFSET_EXPAND,
    NUM_UNIFIED_ACTIONS,
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

        # Tokens for Pakistan capital scenario
        self.the_id = 10
        self.capital_id = 11
        self.of_id = 12
        self.pakistan_id = 13
        self.is_id = 14
        self.lol_id = 15
        self.islam_id = 100
        self.abad_id = 101
        self.karachi_id = 102
        self.dot_id = 20

        self.id_to_str = {
            0: "<s>", 1: "</s>", 2: "<pad>", 3: "<mask>",
            10: " The", 11: " capital", 12: " of", 13: " Pakistan", 14: " is",
            15: " lol", 100: " Islam", 101: "abad", 102: " Karachi", 20: "."
        }
        self.str_to_id = {v: k for k, v in self.id_to_str.items()}

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self.id_to_str.get(t, f"<unk_{t}>") for t in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, int):
            return self.id_to_str.get(token_ids, f"<unk_{token_ids}>")
        return [self.id_to_str.get(t, f"<unk_{t}>") for t in token_ids]

    def encode(self, text, add_special_tokens=False):
        tokens = []
        for word in text.split():
            w_space = " " + word
            if w_space in self.str_to_id:
                tokens.append(self.str_to_id[w_space])
            elif word in self.str_to_id:
                tokens.append(self.str_to_id[word])
        return tokens


class DummyEmbedder(nn.Module):
    def __init__(self, dim=64, vocab_size=500, tokenizer=None):
        super().__init__()
        self.dim = dim
        self.max_length = 32
        self.trainable = False
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(vocab_size, dim)
        self.lm_decoder = nn.Linear(dim, vocab_size, bias=False)
        self.lm_decoder.weight = self.embed.weight

    def embed_ids(self, input_ids, attention_mask=None):
        return self.embed(input_ids)

    def decode_logits(self, hidden_states):
        # Decode hidden states to logits
        logits = self.lm_decoder(hidden_states)
        # For testing speculative unrolling:
        # If the last token is " Islam", make "abad" the highest logit
        # If the last token is " Karachi" or "abad", make "." the highest logit
        return logits


class TestUnifiedActionHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.vocab_size = 500
        self.tokenizer = DummyTokenizer(vocab_size=self.vocab_size)
        self.embedder = DummyEmbedder(dim=self.dim, vocab_size=self.vocab_size, tokenizer=self.tokenizer)

        score_net = MLPScoreNet(dim=self.dim, hidden_dim=128, num_layers=2, time_embed_dim=32, cond_dim=self.dim)
        self.bridge = DiffSchrodingerBridge(dim=self.dim, score_net=score_net, num_steps=20, condition_on_dp1=True)

    def test_unified_edit_head_shape_and_projections(self):
        """Verify UnifiedEditHead produces (..., V + 3) logits and aligns with embedding weights."""
        u_head = UnifiedEditHead(
            dim=self.dim,
            vocab_size=self.vocab_size,
            time_embed_dim=32,
            cond_dim=self.dim,
            embed_weight=self.embedder.embed.weight,
            tie_weights=True,
        )

        B, S = 2, 8
        x = torch.randn(B, S, self.dim)
        t = torch.tensor([0.5, 0.5])
        cond = torch.randn(B, S, self.dim)

        logits, feat = u_head(x, t, cond=cond, return_features=True)
        self.assertEqual(logits.shape, (B, S, self.vocab_size + NUM_UNIFIED_ACTIONS))
        self.assertEqual(feat.shape, (B, S, self.dim))

        # Check that vocabulary projection is tied
        self.assertTrue(torch.equal(u_head.vocab_proj.weight, self.embedder.embed.weight))

    def test_build_unified_targets(self):
        """Verify build_unified_targets maps discrete tags + tokens to V* targets."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            head_mode="unified",
            time_embed_dim=32,
        )

        B, S = 1, 6
        tag_labels = torch.tensor([[KEEP, DELETE, EXPAND, REPLACE, REPLACE, -100]])
        gen_labels = torch.tensor([[-100, -100, -100, 100, 101, -100]])
        clean_ids = torch.tensor([[10, 11, 12, 100, 101, 0]])
        noisy_ids = torch.tensor([[10, 11, 12, 15, 15, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])

        u_targets = hybrid.build_unified_targets(
            tag_labels, gen_labels, clean_ids, noisy_ids, attention_mask
        )

        V = self.vocab_size
        expected = torch.tensor([
            V + ACTION_OFFSET_KEEP,       # KEEP -> V + 0
            V + ACTION_OFFSET_DELETE,     # DELETE -> V + 1
            V + ACTION_OFFSET_EXPAND,     # EXPAND -> V + 2
            100,                          # REPLACE -> clean token 100
            101,                          # REPLACE -> clean token 101
            -100,                         # pad/ignored -> -100
        ])
        self.assertTrue(torch.equal(u_targets[0], expected))

    def test_unified_loss_and_gradient_flow(self):
        """Verify unified cross-entropy loss computation and parameter gradient flow."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            head_mode="unified",
            time_embed_dim=32,
            lambda_sm=1.0,
            lambda_unified=1.0,
        )

        B, S = 2, 6
        dp1 = torch.randn(B, S, self.dim)
        dp2 = torch.randn(B, S, self.dim)
        clean_ids = torch.randint(0, self.vocab_size, (B, S))
        noisy_ids = torch.randint(0, self.vocab_size, (B, S))
        attention_mask = torch.ones(B, S, dtype=torch.long)

        tag_labels = torch.tensor([
            [KEEP, REPLACE, DELETE, EXPAND, KEEP, KEEP],
            [REPLACE, KEEP, DELETE, KEEP, EXPAND, KEEP],
        ])
        gen_labels = torch.full((B, S), -100, dtype=torch.long)
        gen_labels[tag_labels == REPLACE] = clean_ids[tag_labels == REPLACE]

        u_targets = hybrid.build_unified_targets(tag_labels, gen_labels, clean_ids, noisy_ids, attention_mask)

        loss, loss_dict = hybrid.loss(
            dp1=dp1,
            dp2=dp2,
            clean_ids=clean_ids,
            noisy_ids=noisy_ids,
            attention_mask=attention_mask,
            unified_labels=u_targets,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("unified", loss_dict)
        self.assertIn("score_matching", loss_dict)
        self.assertGreater(loss_dict["unified"], 0.0)

        loss.backward()
        for name, p in hybrid.unified_head.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Missing gradient for {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}")

    def test_subword_unrolling_multi_token(self):
        """Verify speculative subword unrolling reconstructs multi-token subword spans."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            head_mode="unified",
            time_embed_dim=32,
        )

        class MockDecoderEmbedder(DummyEmbedder):
            def embed_ids(self, input_ids, attention_mask=None):
                self._last_inp = input_ids
                return self.embed(input_ids)

            def decode_logits(self, hidden_states):
                B, L, D = hidden_states.shape
                logits = torch.zeros(B, L, self.lm_decoder.out_features, device=hidden_states.device)
                for b in range(B):
                    for pos in range(L):
                        prev_tok = self._last_inp[b, pos - 1].item() if pos > 0 else 0
                        if prev_tok == 100:  # Islam -> predict abad (101)
                            logits[b, pos, 101] = 20.0
                        else:  # after abad (101) or anything else -> predict "." (20)
                            logits[b, pos, 20] = 20.0
                return logits

        mock_embedder = MockDecoderEmbedder(dim=self.dim, vocab_size=self.vocab_size, tokenizer=self.tokenizer)

        # 1. Open subword prefix: " Islam" -> unrolls to [" Islam", "abad"]
        prefix = [0, self.tokenizer.the_id, self.tokenizer.capital_id, self.tokenizer.of_id, self.tokenizer.pakistan_id, self.tokenizer.is_id]
        unrolled = hybrid._unroll_subwords(
            prefix_ids=prefix,
            start_token_id=self.tokenizer.islam_id,
            tokenizer=self.tokenizer,
            embedder=mock_embedder,
            max_unroll=4,
        )
        self.assertEqual(unrolled, [self.tokenizer.islam_id, self.tokenizer.abad_id])

        # 2. Closed word: " Karachi" followed by a space or punctuation token -> does NOT unroll
        class MockClosedEmbedder(DummyEmbedder):
            def decode_logits(self, hidden_states):
                B, L, D = hidden_states.shape
                logits = torch.zeros(B, L, self.lm_decoder.out_features, device=hidden_states.device)
                for b in range(B):
                    logits[b, -1, 20] = 20.0  # dot "."
                return logits

        mock_closed = MockClosedEmbedder(dim=self.dim, vocab_size=self.vocab_size, tokenizer=self.tokenizer)
        unrolled_karachi = hybrid._unroll_subwords(
            prefix_ids=prefix,
            start_token_id=self.tokenizer.karachi_id,
            tokenizer=self.tokenizer,
            embedder=mock_closed,
            max_unroll=4,
        )
        self.assertEqual(unrolled_karachi, [self.tokenizer.karachi_id])

    def test_apply_edits_variable_length_splicing(self):
        """Verify _apply_edits expands single-token canvas slot into multi-token spliced span."""
        # Canvas: [<s>, The, capital, of, Pakistan, is, lol, </s>]
        canvas = [
            self.tokenizer.bos_token_id,
            self.tokenizer.the_id,
            self.tokenizer.capital_id,
            self.tokenizer.of_id,
            self.tokenizer.pakistan_id,
            self.tokenizer.is_id,
            self.tokenizer.lol_id,
            self.tokenizer.eos_token_id,
        ]
        tags = [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP, REPLACE, KEEP]
        # REPLACE slot has multi-token list [Islam, abad]
        gen_toks = [
            None, None, None, None, None, None,
            [self.tokenizer.islam_id, self.tokenizer.abad_id],
            None
        ]

        out_ids = DSBHybrid._apply_edits(
            canvas_ids=canvas,
            tags=tags,
            gen_toks=gen_toks,
            bos=self.tokenizer.bos_token_id,
            eos=self.tokenizer.eos_token_id,
            pad=self.tokenizer.pad_token_id,
            M=self.tokenizer.mask_token_id,
        )

        expected = [
            self.tokenizer.bos_token_id,
            self.tokenizer.the_id,
            self.tokenizer.capital_id,
            self.tokenizer.of_id,
            self.tokenizer.pakistan_id,
            self.tokenizer.is_id,
            self.tokenizer.islam_id,
            self.tokenizer.abad_id,
            self.tokenizer.eos_token_id,
        ]
        self.assertEqual(out_ids, expected)
        self.assertEqual(len(out_ids), 9)
        decoded = self.tokenizer.decode(out_ids)
        self.assertIn("Islamabad", decoded)

    def test_unified_generate_text_end_to_end(self):
        """Verify end-to-end generate_text runs without error under head_mode='unified'."""
        hybrid = DSBHybrid(
            bridge=self.bridge,
            vocab_size=self.vocab_size,
            head_mode="unified",
            time_embed_dim=32,
        )

        seed = [
            self.tokenizer.bos_token_id,
            self.tokenizer.the_id,
            self.tokenizer.capital_id,
            self.tokenizer.of_id,
            self.tokenizer.pakistan_id,
            self.tokenizer.is_id,
            self.tokenizer.lol_id,
            self.tokenizer.eos_token_id,
        ]
        seed_tensor = torch.tensor([seed])
        x = self.embedder.embed_ids(seed_tensor)

        results = hybrid.generate_text(
            x=x,
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            seed_ids=[seed],
            max_iterations=2,
            dp1=x,
            unroll_subwords=False,  # test direct unified generation path
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], str)


if __name__ == "__main__":
    unittest.main()
