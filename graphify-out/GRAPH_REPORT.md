# Graph Report - ExDLLM (DSB Hybrid & Diffusion Pretraining)

## Summary
- 596 nodes · 1327 edges · 32 communities
- Extraction: Updated with DSB, DSBHybrid, score networks, and pretraining pipelines.

## God Nodes (most connected - your core abstractions)
1. `dllm_dsb_hybrid` - 128 edges
2. `scripts_train_dsb_hybrid` - 76 edges
3. `DLLMDataset` - 74 edges
4. `scripts_visualize_dsb_bridge` - 68 edges
5. `scripts_generate_dsb_hybrid` - 52 edges
6. `scripts_train_dsb_pretrain` - 44 edges
7. `scripts_train_dsb` - 38 edges
8. `dllm_dsb` - 37 edges
9. `DLLMTokenizer` - 34 edges
10. `DiffSchrodingerBridge` - 28 edges

## Surprising Connections (cross-domain and hybrid bridges)
- `DSBHybrid` --bridges--> `DiffSchrodingerBridge` and `TaggerHead` / `GenHead` [EXTRACTED]
  Continuous Brownian bridge in `dllm/dsb.py` directly interfaces with discrete edit operations in `dllm/dsb_hybrid.py`.
- `corrupt_fixed` --diverges_from--> `corrupt_full` [STRUCTURAL]
  `corrupt_fixed` only outputs KEEP/REPLACE while `corrupt_full` handles Levenshtein grammar, causing class imbalance between training schemes.
- `TransformerScoreNet` --couples_with--> `router (gated drift)` [EXTRACTED]
  Self-attention score network directly predicts discrete edit tags via routing logits to gate continuous drift.
- `TextEmbedder` --bridges--> `AutoModelForMaskedLM` and `DiffSchrodingerBridge` [EXTRACTED]
  Frozen contextual Layer 12 states serve as the boundary condition endpoints (DP1, DP2) for the continuous SDE transport.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_
- **Why does `DSBHybrid` connect the continuous SDE score network to discrete edit token heads, and what causes the gradient bottleneck between them?**
- **How does the `corrupt_fixed` procedure isolate `DSBHybrid` from learning the `DELETE`, `INSERT`, and `EXPAND` edit trajectories?**
- **Why is `TransformerScoreNet` routing gating the drift on `KEEP` slots when contextual neighbor shifts require continuous SDE adjustment?**
- **What is the exact mathematical interface between the continuous x_t intermediate state and the `GenHead` vocabulary projection?**
