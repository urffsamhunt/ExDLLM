# Graph Report - ExDLLM  (2026-09-08)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 725 nodes · 1315 edges · 50 communities (43 shown, 7 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 114 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `149b91e3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DLLMDataset
- DiffSchrodingerBridge
- Graphify Skill
- set_seed
- torch
- DLLMTrainer
- visualizer/app.js
- static/app.js
- ARLMTokenizer
- ForwardCorruptor
- DLLMTokenizer
- DLLMInference
- DLLM
- MLPScoreNet
- train_dsb_pretrain.py
- ARLMTrainer
- dsb_hybrid.py
- train_dsb_hybrid.py
- visualize_dsb_bridge.py
- prepare_translation_data.py
- resolve_device
- Tensor
- TestDSBHybridSampling
- Training & Inference Wiki
- ._sample_topk
- Discrete Diffusion Language Model (DLLM)
- DSBHybrid
- load_dllm_state
- app.py
- test_dsb_hybrid_sampling.py
- ARLMDataset
- DLLM Discrete Edit-Based Diffusion LM
- TextEmbedder
- build_labels
- Model Architecture Wiki
- ._sample_filters
- English-Hindi Translation Config
- Data & Corruption Pipeline Wiki
- clean_lines
- ._get_random_noise_token
- .decode
- Python Dependencies
- TextEmbedder
- PairDataset
- TextEmbedder
- DLLM Inference Visualizer
- dllm/__init__.py
- .build_edit_loss
- .discrete_targets

## God Nodes (most connected - your core abstractions)
1. `DiffSchrodingerBridge` - 33 edges
2. `DLLMTokenizer` - 29 edges
3. `DSBHybrid` - 28 edges
4. `set_seed()` - 28 edges
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
- `Generator Head` --semantically_similar_to--> `Dual-Head Design`  [INFERRED] [semantically similar]
  wiki/Architecture.md → paper.pdf
- `Tagger Head` --semantically_similar_to--> `Dual-Head Design`  [INFERRED] [semantically similar]
  wiki/Architecture.md → paper.pdf
- `Levenshtein Alignment` --semantically_similar_to--> `Forward Corruption Process`  [INFERRED] [semantically similar]
  README.md → paper.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Translation Config Family** — configs_translation, configs_translation_kaggle, configs_translation_kaggle2 [EXTRACTED 1.00]
- **DLLM Architecture** — readme_dual_head_model, readme_forward_corruptor, readme_edit_special_tokens, readme_non_autoregressive_generation [INFERRED 0.85]
- **Edit-Based Diffusion Paradigm** — paper_pdf_dllm, wiki_architecture_edit_tokens, wiki_data_and_corruption_corruptor, wiki_training_and_inference_inference [INFERRED 0.85]
- **Graphify Extraction Pipeline** — _agents_skills_graphify_skill_ast_extraction, _agents_skills_graphify_skill_semantic_extraction, _agents_skills_graphify_skill_knowledge_graph, _agents_skills_graphify_skill_community_detection [INFERRED 0.85]
- **Graphify Reference Documentation** — _agents_skills_graphify_references_add_watch, _agents_skills_graphify_references_exports, _agents_skills_graphify_references_extraction_spec, _agents_skills_graphify_references_github_and_merge, _agents_skills_graphify_references_hooks, _agents_skills_graphify_references_query, _agents_skills_graphify_references_transcribe, _agents_skills_graphify_references_update [INFERRED 0.85]
- **Sub-Iteration Training Pipeline** — configs_default_training, wiki_data_and_corruption_trajectory, wiki_training_and_inference_sub_iteration [INFERRED 0.85]

## Communities (50 total, 7 thin omitted)

### Community 0 - "DLLMDataset"
Cohesion: 0.05
Nodes (35): collate_fn(), DLLMDataset, Dataset, Tensor, DLLM Dataset: Loads text data and applies forward corruption on the fly.…, Args: tokenizer: DLLMTokenizer instance. corruptor: ForwardCorruptor instance.…, Load text from a local file and parse chunks or dialogue pairs., Download and load tiny_shakespeare, extracting dialogue pairs if in… (+27 more)

### Community 1 - "DiffSchrodingerBridge"
Cohesion: 0.12
Nodes (18): cosine_beta_schedule(), DiffSchrodingerBridge, linear_beta_schedule(), Module, no_grad, Tensor, A diffusion Schrödinger bridge from an input datapoint DP1 to an output…, Linearly interpolate a schedule buffer at continuous t in [0, 1]. (+10 more)

### Community 2 - "Graphify Skill"
Cohesion: 0.09
Nodes (28): Add URL and Watch Folder Reference, Exports and Benchmark Reference, Extraction Subagent Spec, GitHub Clone and Cross-Repo Merge Reference, Commit Hook and CLAUDE.md Integration Reference, Query, Path, Explain Reference, Transcribe Video and Audio Reference, Incremental Update and Cluster-Only Reference (+20 more)

### Community 3 - "set_seed"
Cohesion: 0.13
Nodes (20): ARLM, load_arlm_state(), Module, Tensor, ARLM Model: standard autoregressive language model. This is a thin wrapper…, Return parameter groups with weight decay applied only to weights (not biases…, Load an ARLM state dict, unwrapping DataParallel if present. Returns a message…, Autoregressive language model wrapper. Usage: model = ARLM("roberta-base") loss… (+12 more)

### Community 4 - "torch"
Cohesion: 0.11
Nodes (19): ARLM Trainer: standard teacher-forced autoregressive training loop. The…, DLLM Trainer: Training loop with combined Tagger + Generator loss. Handles: -…, compute_edit_accuracy(), delete_positions(), _expand_1d(), expand_sequence(), JSONMetricsLogger, Tensor (+11 more)

### Community 5 - "DLLMTrainer"
Cohesion: 0.11
Nodes (15): DataLoader, DLLMTrainer, Module, no_grad, Tensor, Initialize AdamW optimizer with proper weight decay grouping., Linear warmup followed by linear decay., Main training loop. Args: train_loader: DataLoader for training data.… (+7 more)

### Community 6 - "visualizer/app.js"
Cohesion: 0.16
Nodes (22): buildChart(), buildTimeline(), chipClass(), confToColor(), escHtml(), gotoStep(), hide(), highlightTimelineColumn() (+14 more)

### Community 7 - "static/app.js"
Cohesion: 0.16
Nodes (22): buildChart(), buildTimeline(), chipClass(), confToColor(), escHtml(), gotoStep(), hide(), highlightTimelineColumn() (+14 more)

### Community 8 - "ARLMTokenizer"
Cohesion: 0.11
Nodes (13): ARLM Dataset: teacher-forced (prompt + response) sequences for an…, ARLMInference, ARLM Inference: standard autoregressive (left-to-right) generation. The public…, Extract the newly generated portion of `full_text` (the continuation after the…, Autoregressive sampling for the ARLM model. Usage: inference =…, ARLMTokenizer, Tensor, ARLM Tokenizer: thin wrapper around a HuggingFace causal LM tokenizer. Unlike… (+5 more)

### Community 9 - "ForwardCorruptor"
Cohesion: 0.12
Nodes (13): ForwardCorruptor, Perturb a token into a spelling typo or grammatical/inflection variant., Corrupt a clean token sequence and produce tag labels. Args: clean_ids: List of…, Applies synthetic edits to token sequences and computes ground-truth edit…, Randomly replace some tokens with noise tokens or morphological variants., Args: tokenizer: DLLMTokenizer instance for token ID access. replace_ratio:…, Randomly delete some tokens., Insert random noise tokens at random positions. (+5 more)

### Community 10 - "DLLMTokenizer"
Cohesion: 0.09
Nodes (10): DLLMTokenizer, Encode a text string into token IDs (no padding/truncation)., Check if a token ID corresponds to one of the edit operation tokens., Check if a token ID is a regular vocabulary token (not a special edit token)., Get the string name of an edit tag token., Access the underlying HuggingFace tokenizer., Wraps a RobertaTokenizer with additional edit-operation tokens. Usage: tok =…, Args: base_model: HuggingFace model identifier for the base tokenizer.… (+2 more)

### Community 11 - "DLLMInference"
Cohesion: 0.12
Nodes (13): make_bleu_evaluator(), BLEU evaluation callback for the training loop. Translates a deterministic…, Build a callable that returns {'bleu': score} on a fixed subset of a test TSV…, DLLMInference, no_grad, Tensor, DLLM Inference: Iterative denoising with full 5-op canvas evolution. Canvas…, Top-k + nucleus sampling from generator logits. Returns one token per position. (+5 more)

### Community 12 - "DLLM"
Cohesion: 0.14
Nodes (11): DLLM, Tensor, Tie the generator's final linear layer to the backbone's input token embeddings…, Build and register the tag ID mapping buffer eagerly., Convert tokenizer tag IDs (keep_id, delete_id, etc.) to internal 0-based…, Convert internal tag index (0-4) to tokenizer tag ID., Forward pass through the two-headed model. Args: noisy_ids: Corrupted token…, Discrete Diffusion Language Model with a bidirectional Transformer backbone and… (+3 more)

### Community 13 - "MLPScoreNet"
Cohesion: 0.17
Nodes (14): corrupt_fixed(), Corrupt a clean token sequence in place (same length returned). Each real…, MLPScoreNet, Diffusion Schrödinger Bridge (DSB) for score-based training. We model a Markov…, A small transformer score network: self-attention across positions. Unlike…, A time-conditioned MLP score network. Maps (x, t) -> s_theta(x, t), an estimate…, TransformerScoreNet, build_bridge() (+6 more)

### Community 14 - "train_dsb_pretrain.py"
Cohesion: 0.16
Nodes (14): batch_stream(), corrupt_ids(), iter_lines(), main(), parse_args(), Yield non-empty stripped lines from a file, one at a time (lazy)., Yield batches of lines from a file, streaming and with a finite shuffle buffer.…, Yield up to `max_batches` batches (no shuffle) for streaming eval. (+6 more)

### Community 15 - "ARLMTrainer"
Cohesion: 0.15
Nodes (11): make_collate(), Return a collate function that pads a batch to the longest sequence., ARLMTrainer, Module, Serialize and write a checkpoint on a background thread., Trainer for the autoregressive language model. Usage: trainer =…, Load model, optimizer, and training state from checkpoint., Initialize AdamW optimizer with proper weight-decay grouping. (+3 more)

### Community 16 - "dsb_hybrid.py"
Cohesion: 0.17
Nodes (14): Any, Forward Corruptor: Corrupts clean text sequences and computes Levenshtein edit…, align_to_fixed(), corrupt_multiroute(), _is_syntax_token(), _perturb_word_morph(), _perturb_word_typo(), Auxiliary-head hybrid: Diffusion Schrödinger Bridge + discrete edit heads.… (+6 more)

### Community 17 - "train_dsb_hybrid.py"
Cohesion: 0.23
Nodes (15): apply_collapse_drift(), corrupt_full(), Interpolated Collapse Drift for continuous Diffusion Schrödinger Bridge. For…, Corrupt a clean token sequence with the FULL edit grammar (via…, batch_clean_ids_from_texts(), batch_stream(), build_batch_labels(), evaluate() (+7 more)

### Community 18 - "visualize_dsb_bridge.py"
Cohesion: 0.19
Nodes (14): EditConditionedScoreNet, Score network that conditions the continuous drift on discrete edit tags.…, build_interactive_html(), build_static_png(), generate_variations(), inverse_project_pca(), load_hybrid_model(), main() (+6 more)

### Community 19 - "prepare_translation_data.py"
Cohesion: 0.20
Nodes (14): append_word_dict(), hindi_fraction(), iitb_iter(), is_word(), main(), normalize(), quality_filter(), Fraction of characters in `s` that fall in the Devanagari range. (+6 more)

### Community 20 - "resolve_device"
Cohesion: 0.25
Nodes (12): Resolve the best available compute device. Preference order: CUDA GPU > Intel…, resolve_device(), _cached_embed(), embed_pairs(), main(), parse_args(), no_grad, Mean-pool the last hidden state of a frozen RoBERTa encoder. (+4 more)

### Community 21 - "Tensor"
Cohesion: 0.23
Nodes (7): GenHead, Module, Tensor, x: (B, S, D) or (B, D); t: (B,); tag_ids: (B, S) int64 (default: sentinel);…, Predict the edit op (NUM_TAGS classes) at every position. Conditions on the…, Predict the clean token at REPLACE positions (sparse projection). Same t + DP1…, TaggerHead

### Community 22 - "TestDSBHybridSampling"
Cohesion: 0.18
Nodes (7): MockModule, Test GenHead returns aligned features and supports op conditioning., Test that an INSERT op prevents generating the anchor token itself., Test lm_blend_weight blends MLM head logits on REPLACE slots., Test progressive_fill resolves multi-token spans without error., Test that legacy state dicts without op_emb load cleanly with strict=False., TestDSBHybridSampling

### Community 23 - "Training & Inference Wiki"
Cohesion: 0.24
Nodes (12): DLLM Default Configuration, Default Corruption Config, Default Inference Config, Default Training Config, English-Hindi Parallel Data Config, Translation Training Config, DLLM Wiki Home, DLLM Project Overview (+4 more)

### Community 24 - "._sample_topk"
Cohesion: 0.33
Nodes (3): Vectorized distance-gated top-k / top-p sampling over vocab logits -> token ids…, Test _sample_topk prunes tokens using aligned feat_states and decoder_weight., Test repetition penalty respects exempt_tokens and local windows.

### Community 25 - "Discrete Diffusion Language Model (DLLM)"
Cohesion: 0.22
Nodes (11): DLLM Wiki Skill, DLLM Wiki, DLLM README, Discrete Diffusion Language Model (DLLM), Dual-Head Model, Edit Special Tokens, Forward Corruptor, Non-Autoregressive Generation (+3 more)

### Community 26 - "DSBHybrid"
Cohesion: 0.17
Nodes (10): DSBHybrid, no_grad, Generate an output embedding via the reverse SDE, then run the discrete heads…, Comprehensive interpretability diagnostics: 1. Continuous SDE: baseline, signal…, Decode continuous embeddings x to token IDs by finding the nearest neighbor in…, Two-phase generation: reverse the SDE to an output embedding, then read off the…, Turn the SDE output embedding into literal text by porting the DLLM iterative…, Apply KEEP/DELETE/REPLACE/INSERT/EXPAND, ported from DLLM._execute_edits. (+2 more)

### Community 27 - "load_dllm_state"
Cohesion: 0.27
Nodes (9): load_dllm_state(), Module, Load a DLLM state dict, tolerating older checkpoints: - checkpoints from the…, load_dllm(), main(), parse_args(), load_model(), main() (+1 more)

### Community 28 - "app.py"
Cohesion: 0.24
Nodes (7): route, DSBInferenceWrapper, generate(), index(), load_model(), DLLM Visualizer — Flask backend. Usage: python visualizer/app.py --checkpoint…, status()

### Community 30 - "ARLMDataset"
Cohesion: 0.20
Nodes (6): ARLMDataset, Dataset, Tensor, PyTorch Dataset of teacher-forced prompt+response sequences. Args: tokenizer:…, Access the underlying HuggingFace tokenizer., PreTrainedTokenizer

### Community 31 - "DLLM Discrete Edit-Based Diffusion LM"
Cohesion: 0.27
Nodes (10): DLLM Technical Report, DiffusER, DLLM Discrete Edit-Based Diffusion LM, DreamOn, Edit Operations (KEEP/DELETE/REPLACE/INSERT/EXPAND), EXPAND Operation, Dynamic Length Control, LLaDA (+2 more)

### Community 32 - "TextEmbedder"
Cohesion: 0.20
Nodes (5): Frozen or trainable RoBERTa embedder producing per-token embeddings., Return PER-TOKEN embeddings (B, S, D), not pooled., Decode Layer 12 contextual hidden states (..., D) -> (..., V) vocabulary logits., Mean-pooled document embedding (B, D) for nearest-neighbour search., TextEmbedder

### Community 33 - "build_labels"
Cohesion: 0.25
Nodes (6): build_labels(), Build shifted next-token labels for teacher forcing. The label at position i is…, no_grad, Main training loop. Args: train_loader: DataLoader yielding {'input_ids',…, Run evaluation on the validation set. Returns averaged metrics., Save model, optimizer, and training state. State dicts are snapshotted to CPU…

### Community 34 - "Model Architecture Wiki"
Cohesion: 0.42
Nodes (9): Dual-Head Design, Model Architecture Wiki, RoBERTa Backbone, Edit Tokenizer Extension, Generator Head, Joint Loss Function, Length Head, Prompt Protection Principle (+1 more)

### Community 35 - "._sample_filters"
Cohesion: 0.29
Nodes (5): no_grad, Tensor, Sample tokens from an (N, V) logits tensor (used by the benchmark)., Generate a continuation for the given prompt via autoregressive decoding. Args:…, Apply top-k and nucleus filtering to a (1, vocab) logits tensor.

### Community 36 - "English-Hindi Translation Config"
Cohesion: 0.33
Nodes (7): Default Model Config (RoBERTa), English-Hindi Translation Config, Kaggle T4 Translation Config, Kaggle 2xT4 Translation Config, Kaggle 2xT4 Training Config, Kaggle Training Config (batch 8), XLM-RoBERTa Translation Model Config

### Community 37 - "Data & Corruption Pipeline Wiki"
Cohesion: 0.48
Nodes (7): Forward Corruption Process, Levenshtein Alignment, Data & Corruption Pipeline Wiki, Prompt-Response Canvas Corruption, DLLMDataset, Shortage Training, Progressive Trajectory Corruption

### Community 38 - "clean_lines"
Cohesion: 0.43
Nodes (6): clean_lines(), devanagari_fraction(), main(), normalize_ws(), NFKC normalize + collapse all whitespace runs., Split article text into cleaned, content-bearing lines.

### Community 39 - "._get_random_noise_token"
Cohesion: 0.33
Nodes (3): Sample a random noise token from the noise pool, avoiding exclude_tok., Builds an interleaved canvas with a noise level t skewed toward 1: For each…, Build a progressive denoising trajectory for a single (prompt, response) pair.…

### Community 40 - ".decode"
Cohesion: 0.33
Nodes (4): Tensor, Decode token IDs back to a string., Tokenize text(s) with padding and truncation. Returns a dict with 'input_ids'…, or

### Community 41 - "Python Dependencies"
Cohesion: 0.33
Nodes (6): Python Dependencies, HuggingFace Datasets, python-Levenshtein, PyYAML, PyTorch, HuggingFace Transformers

### Community 45 - "DLLM Inference Visualizer"
Cohesion: 0.33
Nodes (6): DLLM Inference Visualizer, Canvas View, Iteration Timeline, DLLM Inference Visualizer (static), Iterative Reverse Denoising Inference, Refinement Loop

## Knowledge Gaps
- **31 isolated node(s):** `TAG_COLOR`, `TAG_ORDER`, `trajectory`, `TAG_COLOR`, `TAG_ORDER` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DLLMDataset` connect `DLLMDataset` to `ARLMTokenizer`, `dllm/__init__.py`, `set_seed`, `ARLMDataset`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `DSBHybrid` connect `DSBHybrid` to `DiffSchrodingerBridge`, `MLPScoreNet`, `.build_edit_loss`, `dsb_hybrid.py`, `.discrete_targets`, `train_dsb_hybrid.py`, `visualize_dsb_bridge.py`, `Tensor`, `TestDSBHybridSampling`, `._sample_topk`, `test_dsb_hybrid_sampling.py`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `DiffSchrodingerBridge` connect `DiffSchrodingerBridge` to `MLPScoreNet`, `train_dsb_pretrain.py`, `dsb_hybrid.py`, `train_dsb_hybrid.py`, `visualize_dsb_bridge.py`, `resolve_device`, `Tensor`, `TestDSBHybridSampling`, `DSBHybrid`, `test_dsb_hybrid_sampling.py`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `DiffSchrodingerBridge` (e.g. with `DSBHybrid` and `sample()`) actually correct?**
  _`DiffSchrodingerBridge` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `DLLMTokenizer` (e.g. with `.__init__()` and `load_dllm()`) actually correct?**
  _`DLLMTokenizer` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `DSBHybrid` (e.g. with `DiffSchrodingerBridge` and `TestDSBHybridSampling`) actually correct?**
  _`DSBHybrid` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `TAG_COLOR`, `TAG_ORDER`, `trajectory` to the rest of the system?**
  _31 weakly-connected nodes found - possible documentation gaps or missing edges._