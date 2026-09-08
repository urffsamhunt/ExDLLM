# Graph Report - ExDLLM  (2026-09-08)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 729 nodes · 1313 edges · 38 communities (30 shown, 8 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `149b91e3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- visualize_dsb_bridge.py
- set_seed
- Model Architecture Wiki
- DLLMDataset
- DLLMTrainer
- DiffSchrodingerBridge
- DLLMTokenizer
- Graphify Skill
- visualizer/app.js
- static/app.js
- ForwardCorruptor
- benchmark.py
- DLLMInference
- train_dsb_hybrid.py
- train_dsb_pretrain.py
- DSBHybrid
- dsb_hybrid.py
- ARLMTokenizer
- ARLMTrainer
- Tensor
- prepare_translation_data.py
- torch
- TestDSBHybridSampling
- ARLMInference
- Discrete Diffusion Language Model (DLLM)
- DummyEmbedder
- .train
- ARLMDataset
- ._get_random_noise_token
- ._sample_topk
- TextEmbedder
- dllm/__init__.py
- .build_edit_loss
- .discrete_targets
- Module
- no_grad
- Tensor

## God Nodes (most connected - your core abstractions)
1. `DiffSchrodingerBridge` - 30 edges
2. `DLLMTokenizer` - 29 edges
3. `set_seed()` - 28 edges
4. `DSBHybrid` - 27 edges
5. `DLLMDataset` - 27 edges
6. `ARLMTokenizer` - 21 edges
7. `MLPScoreNet` - 20 edges
8. `ForwardCorruptor` - 20 edges
9. `resolve_device()` - 19 edges
10. `DLLM` - 18 edges

## Surprising Connections (you probably didn't know these)
- `DLLM Discrete Edit-Based Diffusion LM` --semantically_similar_to--> `Model Architecture Wiki`  [INFERRED] [semantically similar]
  paper.pdf → wiki/Architecture.md
- `Edit Tokenizer Extension` --semantically_similar_to--> `Edit Operations (KEEP/DELETE/REPLACE/INSERT/EXPAND)`  [INFERRED] [semantically similar]
  wiki/Architecture.md → paper.pdf
- `Levenshtein Alignment` --semantically_similar_to--> `Forward Corruption Process`  [INFERRED] [semantically similar]
  README.md → paper.pdf
- `Prompt-Response Canvas Corruption` --semantically_similar_to--> `Forward Corruption Process`  [INFERRED] [semantically similar]
  wiki/Data-and-Corruption.md → paper.pdf
- `Generator Head` --semantically_similar_to--> `Dual-Head Design`  [INFERRED] [semantically similar]
  wiki/Architecture.md → paper.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Translation Config Family** — configs_translation, configs_translation_kaggle, configs_translation_kaggle2 [EXTRACTED 1.00]
- **DLLM Architecture** — readme_dual_head_model, readme_forward_corruptor, readme_edit_special_tokens, readme_non_autoregressive_generation [INFERRED 0.85]
- **Edit-Based Diffusion Paradigm** — paper_pdf_dllm, wiki_architecture_edit_tokens, wiki_data_and_corruption_corruptor, wiki_training_and_inference_inference [INFERRED 0.85]
- **Graphify Extraction Pipeline** — _agents_skills_graphify_skill_ast_extraction, _agents_skills_graphify_skill_semantic_extraction, _agents_skills_graphify_skill_knowledge_graph, _agents_skills_graphify_skill_community_detection [INFERRED 0.85]
- **Graphify Reference Documentation** — _agents_skills_graphify_references_add_watch, _agents_skills_graphify_references_exports, _agents_skills_graphify_references_extraction_spec, _agents_skills_graphify_references_github_and_merge, _agents_skills_graphify_references_hooks, _agents_skills_graphify_references_query, _agents_skills_graphify_references_transcribe, _agents_skills_graphify_references_update [INFERRED 0.85]
- **Sub-Iteration Training Pipeline** — configs_default_training, wiki_data_and_corruption_trajectory, wiki_training_and_inference_sub_iteration [INFERRED 0.85]

## Communities (38 total, 8 thin omitted)

### Community 0 - "visualize_dsb_bridge.py"
Cohesion: 0.05
Nodes (45): MLPScoreNet, Diffusion Schrödinger Bridge (DSB) for score-based training. We model a Markov…, A small transformer score network: self-attention across positions. Unlike…, A time-conditioned MLP score network. Maps (x, t) -> s_theta(x, t), an estimate…, TransformerScoreNet, Resolve the best available compute device. Preference order: CUDA GPU > Intel…, resolve_device(), build_bridge() (+37 more)

### Community 1 - "set_seed"
Cohesion: 0.05
Nodes (46): DLLM, load_dllm_state(), Module, Tensor, DLLM Model: A two-headed bidirectional Transformer for edit-based diffusion.…, Tie the generator's final linear layer to the backbone's input token embeddings…, Build and register the tag ID mapping buffer eagerly., Convert tokenizer tag IDs (keep_id, delete_id, etc.) to internal 0-based… (+38 more)

### Community 2 - "Model Architecture Wiki"
Cohesion: 0.05
Nodes (58): DLLM Default Configuration, Default Corruption Config, Default Inference Config, Default Model Config (RoBERTa), Default Training Config, English-Hindi Translation Config, English-Hindi Parallel Data Config, Kaggle T4 Translation Config (+50 more)

### Community 3 - "DLLMDataset"
Cohesion: 0.05
Nodes (35): collate_fn(), DLLMDataset, Dataset, Tensor, DLLM Dataset: Loads text data and applies forward corruption on the fly.…, Args: tokenizer: DLLMTokenizer instance. corruptor: ForwardCorruptor instance.…, Load text from a local file and parse chunks or dialogue pairs., Download and load tiny_shakespeare, extracting dialogue pairs if in… (+27 more)

### Community 4 - "DLLMTrainer"
Cohesion: 0.07
Nodes (23): DataLoader, DLLMTrainer, Module, no_grad, Tensor, Initialize AdamW optimizer with proper weight decay grouping., Linear warmup followed by linear decay., Main training loop. Args: train_loader: DataLoader for training data.… (+15 more)

### Community 5 - "DiffSchrodingerBridge"
Cohesion: 0.12
Nodes (18): cosine_beta_schedule(), DiffSchrodingerBridge, linear_beta_schedule(), Module, no_grad, Tensor, A diffusion Schrödinger bridge from an input datapoint DP1 to an output…, Linearly interpolate a schedule buffer at continuous t in [0, 1]. (+10 more)

### Community 6 - "DLLMTokenizer"
Cohesion: 0.07
Nodes (15): DLLMTokenizer, Tensor, DLLM Tokenizer: Extended RoBERTa tokenizer with edit operation tokens. Adds the…, Encode a text string into token IDs (no padding/truncation)., Decode token IDs back to a string., Tokenize text(s) with padding and truncation. Returns a dict with 'input_ids'…, Check if a token ID corresponds to one of the edit operation tokens., Check if a token ID is a regular vocabulary token (not a special edit token). (+7 more)

### Community 7 - "Graphify Skill"
Cohesion: 0.09
Nodes (28): Add URL and Watch Folder Reference, Exports and Benchmark Reference, Extraction Subagent Spec, GitHub Clone and Cross-Repo Merge Reference, Commit Hook and CLAUDE.md Integration Reference, Query, Path, Explain Reference, Transcribe Video and Audio Reference, Incremental Update and Cluster-Only Reference (+20 more)

### Community 8 - "visualizer/app.js"
Cohesion: 0.16
Nodes (22): buildChart(), buildTimeline(), chipClass(), confToColor(), escHtml(), gotoStep(), hide(), highlightTimelineColumn() (+14 more)

### Community 9 - "static/app.js"
Cohesion: 0.16
Nodes (22): buildChart(), buildTimeline(), chipClass(), confToColor(), escHtml(), gotoStep(), hide(), highlightTimelineColumn() (+14 more)

### Community 10 - "ForwardCorruptor"
Cohesion: 0.12
Nodes (13): ForwardCorruptor, Perturb a token into a spelling typo or grammatical/inflection variant., Corrupt a clean token sequence and produce tag labels. Args: clean_ids: List of…, Applies synthetic edits to token sequences and computes ground-truth edit…, Randomly replace some tokens with noise tokens or morphological variants., Args: tokenizer: DLLMTokenizer instance for token ID access. replace_ratio:…, Randomly delete some tokens., Insert random noise tokens at random positions. (+5 more)

### Community 11 - "benchmark.py"
Cohesion: 0.15
Nodes (14): ARLM, load_arlm_state(), Module, Tensor, ARLM Model: standard autoregressive language model. This is a thin wrapper…, Return parameter groups with weight decay applied only to weights (not biases…, Load an ARLM state dict, unwrapping DataParallel if present. Returns a message…, Autoregressive language model wrapper. Usage: model = ARLM("roberta-base") loss… (+6 more)

### Community 12 - "DLLMInference"
Cohesion: 0.12
Nodes (13): make_bleu_evaluator(), BLEU evaluation callback for the training loop. Translates a deterministic…, Build a callable that returns {'bleu': score} on a fixed subset of a test TSV…, DLLMInference, no_grad, Tensor, DLLM Inference: Iterative denoising with full 5-op canvas evolution. Canvas…, Top-k + nucleus sampling from generator logits. Returns one token per position. (+5 more)

### Community 13 - "train_dsb_hybrid.py"
Cohesion: 0.20
Nodes (17): apply_collapse_drift(), corrupt_full(), EditConditionedScoreNet, Interpolated Collapse Drift for continuous Diffusion Schrödinger Bridge. For…, Corrupt a clean token sequence with the FULL edit grammar (via…, Score network that conditions the continuous drift on discrete edit tags.…, batch_clean_ids_from_texts(), batch_stream() (+9 more)

### Community 14 - "train_dsb_pretrain.py"
Cohesion: 0.16
Nodes (14): batch_stream(), corrupt_ids(), iter_lines(), main(), parse_args(), Yield non-empty stripped lines from a file, one at a time (lazy)., Yield batches of lines from a file, streaming and with a finite shuffle buffer.…, Yield up to `max_batches` batches (no shuffle) for streaming eval. (+6 more)

### Community 15 - "DSBHybrid"
Cohesion: 0.17
Nodes (10): DSBHybrid, Generate an output embedding via the reverse SDE, then run the discrete heads…, Comprehensive interpretability diagnostics: 1. Continuous SDE: baseline, signal…, Decode continuous embeddings x to token IDs by finding the nearest neighbor in…, Two-phase generation: reverse the SDE to an output embedding, then read off the…, Turn the SDE output embedding into literal text by porting the DLLM iterative…, Apply KEEP/DELETE/REPLACE/INSERT/EXPAND, ported from DLLM._execute_edits., True variable-length iterative refinement decode (DLLM-style, ported). Each… (+2 more)

### Community 16 - "dsb_hybrid.py"
Cohesion: 0.18
Nodes (15): Any, align_to_fixed(), corrupt_fixed(), corrupt_multiroute(), _is_syntax_token(), _perturb_word_morph(), _perturb_word_typo(), Auxiliary-head hybrid: Diffusion Schrödinger Bridge + discrete edit heads.… (+7 more)

### Community 17 - "ARLMTokenizer"
Cohesion: 0.13
Nodes (9): ARLMTokenizer, Tensor, Wrap a HuggingFace causal-LM tokenizer with a DLLM-compatible surface., Args: base_model: HuggingFace model identifier for the base tokenizer.…, Encode a text string into token IDs (no padding/truncation by default)., Decode token IDs back to a string., Tokenize text(s) with padding and truncation., Access the underlying HuggingFace tokenizer. (+1 more)

### Community 18 - "ARLMTrainer"
Cohesion: 0.17
Nodes (9): ARLMTrainer, Module, Serialize and write a checkpoint on a background thread., Trainer for the autoregressive language model. Usage: trainer =…, Load model, optimizer, and training state from checkpoint., Initialize AdamW optimizer with proper weight-decay grouping., Linear warmup followed by linear decay (identical to DLLM)., main() (+1 more)

### Community 19 - "Tensor"
Cohesion: 0.19
Nodes (8): DiffSchrodingerBridge, GenHead, x: (B, S, D) or (B, D); t: (B,); tag_ids: (B, S) int64 (default: sentinel);…, Predict the edit op (NUM_TAGS classes) at every position. Conditions on the…, Predict the clean token at REPLACE positions (sparse projection). Same t + DP1…, TaggerHead, Module, Tensor

### Community 20 - "prepare_translation_data.py"
Cohesion: 0.20
Nodes (14): append_word_dict(), hindi_fraction(), iitb_iter(), is_word(), main(), normalize(), quality_filter(), Fraction of characters in `s` that fall in the Devanagari range. (+6 more)

### Community 21 - "torch"
Cohesion: 0.23
Nodes (9): build_labels(), make_collate(), ARLM Dataset: teacher-forced (prompt + response) sequences for an…, Return a collate function that pads a batch to the longest sequence., Build shifted next-token labels for teacher forcing. The label at position i is…, ARLM Inference: standard autoregressive (left-to-right) generation. The public…, ARLM Tokenizer: thin wrapper around a HuggingFace causal LM tokenizer. Unlike…, ARLM Trainer: standard teacher-forced autoregressive training loop. The… (+1 more)

### Community 22 - "TestDSBHybridSampling"
Cohesion: 0.18
Nodes (7): MockModule, Test GenHead returns aligned features and supports op conditioning., Test that an INSERT op prevents generating the anchor token itself., Test lm_blend_weight blends MLM head logits on REPLACE slots., Test progressive_fill resolves multi-token spans without error., Test that legacy state dicts without op_emb load cleanly with strict=False., TestDSBHybridSampling

### Community 23 - "ARLMInference"
Cohesion: 0.19
Nodes (8): ARLMInference, no_grad, Tensor, Sample tokens from an (N, V) logits tensor (used by the benchmark)., Extract the newly generated portion of `full_text` (the continuation after the…, Autoregressive sampling for the ARLM model. Usage: inference =…, Generate a continuation for the given prompt via autoregressive decoding. Args:…, Apply top-k and nucleus filtering to a (1, vocab) logits tensor.

### Community 24 - "Discrete Diffusion Language Model (DLLM)"
Cohesion: 0.24
Nodes (10): DLLM Wiki Skill, DLLM Wiki, DLLM README, Discrete Diffusion Language Model (DLLM), Dual-Head Model, Edit Special Tokens, Non-Autoregressive Generation, Progressive Trajectory Training (+2 more)

### Community 26 - ".train"
Cohesion: 0.22
Nodes (6): no_grad, Main training loop. Args: train_loader: DataLoader yielding {'input_ids',…, Run evaluation on the validation set. Returns averaged metrics., Save model, optimizer, and training state. State dicts are snapshotted to CPU…, Recursively move all tensors in a nested structure to CPU (detached). Used to…, to_cpu()

### Community 27 - "ARLMDataset"
Cohesion: 0.33
Nodes (4): ARLMDataset, Dataset, Tensor, PyTorch Dataset of teacher-forced prompt+response sequences. Args: tokenizer:…

### Community 28 - "._get_random_noise_token"
Cohesion: 0.33
Nodes (3): Sample a random noise token from the noise pool, avoiding exclude_tok., Builds an interleaved canvas with a noise level t skewed toward 1: For each…, Build a progressive denoising trajectory for a single (prompt, response) pair.…

### Community 29 - "._sample_topk"
Cohesion: 0.33
Nodes (3): Vectorized distance-gated top-k / top-p sampling over vocab logits -> token ids…, Test _sample_topk prunes tokens using aligned feat_states and decoder_weight., Test repetition penalty respects exempt_tokens and local windows.

## Knowledge Gaps
- **31 isolated node(s):** `TAG_COLOR`, `TAG_ORDER`, `trajectory`, `TAG_COLOR`, `TAG_ORDER` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DSBHybrid` connect `DSBHybrid` to `.build_edit_loss`, `.discrete_targets`, `visualize_dsb_bridge.py`, `train_dsb_hybrid.py`, `dsb_hybrid.py`, `Tensor`, `TestDSBHybridSampling`, `._sample_topk`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `DLLMDataset` connect `DLLMDataset` to `set_seed`, `benchmark.py`, `ARLMTokenizer`, `torch`, `dllm/__init__.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `set_seed()` connect `set_seed` to `visualize_dsb_bridge.py`, `DLLMDataset`, `benchmark.py`, `train_dsb_hybrid.py`, `train_dsb_pretrain.py`, `ARLMTrainer`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `DiffSchrodingerBridge` (e.g. with `sample()` and `TestDSBHybridSampling`) actually correct?**
  _`DiffSchrodingerBridge` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `DLLMTokenizer` (e.g. with `.__init__()` and `load_dllm()`) actually correct?**
  _`DLLMTokenizer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `TAG_COLOR`, `TAG_ORDER`, `trajectory` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `visualize_dsb_bridge.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05048076923076923 - nodes in this community are weakly interconnected._