# Graph Report - DLLM  (2026-08-21)

## Corpus Check
- 55 files · ~53,032 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 483 nodes · 814 edges · 22 communities (21 shown, 1 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 92 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8e07545d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DLLM
- Model Architecture Wiki
- DLLMDataset
- DLLMTrainer
- ForwardCorruptor
- DLLMTokenizer
- Graphify Skill
- visualizer/app.js
- static/app.js
- prepare_translation_data.py
- ARLM
- Discrete Diffusion Language Model (DLLM)
- clean_lines
- _expand_1d
- torch
- .decode
- ARLMTokenizer
- arlm/dataset.py
- ARLMTrainer
- ._sample_filters
- ARLMDataset

## God Nodes (most connected - your core abstractions)
1. `DLLMTokenizer` - 34 edges
2. `DLLMDataset` - 28 edges
3. `ARLMTokenizer` - 21 edges
4. `DLLM` - 21 edges
5. `DLLMInference` - 19 edges
6. `ForwardCorruptor` - 18 edges
7. `set_seed()` - 18 edges
8. `Graphify Skill` - 18 edges
9. `load_dllm_state()` - 14 edges
10. `DLLMTrainer` - 14 edges

## Surprising Connections (you probably didn't know these)
- `DLLM Discrete Edit-Based Diffusion LM` --semantically_similar_to--> `Model Architecture Wiki`  [INFERRED] [semantically similar]
  paper.pdf → wiki/Architecture.md
- `Edit Tokenizer Extension` --semantically_similar_to--> `Edit Operations (KEEP/DELETE/REPLACE/INSERT/EXPAND)`  [INFERRED] [semantically similar]
  wiki/Architecture.md → paper.pdf
- `Prompt-Response Canvas Corruption` --semantically_similar_to--> `Forward Corruption Process`  [INFERRED] [semantically similar]
  wiki/Data-and-Corruption.md → paper.pdf
- `Levenshtein Alignment` --semantically_similar_to--> `Forward Corruption Process`  [INFERRED] [semantically similar]
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

## Communities (22 total, 1 thin omitted)

### Community 0 - "DLLM"
Cohesion: 0.05
Nodes (37): BLEU evaluation callback for the training loop. Translates a deterministic…, DLLMInference, no_grad, Tensor, DLLM Inference: Iterative denoising with full 5-op canvas evolution. Canvas…, Top-k + nucleus sampling from generator logits. Returns one token per position., Apply predicted edit operations to the canvas, enforcing positional rules: BOS…, Iterative denoising inference for the DLLM model. Usage: inference =… (+29 more)

### Community 1 - "Model Architecture Wiki"
Cohesion: 0.06
Nodes (57): DLLM Default Configuration, Default Corruption Config, Default Inference Config, Default Model Config (RoBERTa), Default Training Config, English-Hindi Translation Config, English-Hindi Parallel Data Config, Kaggle T4 Translation Config (+49 more)

### Community 2 - "DLLMDataset"
Cohesion: 0.09
Nodes (18): DLLMDataset, Dataset, Tensor, Load text from a local file and parse chunks or dialogue pairs., Download and load tiny_shakespeare, extracting dialogue pairs if in…, Split long text into chunks suitable for tokenization. Merges lines until each…, Extract consecutive dialogue turn pairs (Turn K -> Turn K+1) from text. For…, Number of text chunks or dialogue pairs available. (+10 more)

### Community 3 - "DLLMTrainer"
Cohesion: 0.09
Nodes (21): DataLoader, make_bleu_evaluator(), Build a callable that returns {'bleu': score} on a fixed subset of a test TSV…, collate_fn(), DLLM Dataset: Loads text data and applies forward corruption on the fly.…, Custom collate function; filters out dummy samples and stacks tensors. Handles…, DLLMTrainer, Module (+13 more)

### Community 4 - "ForwardCorruptor"
Cohesion: 0.08
Nodes (18): ForwardCorruptor, Forward Corruptor: Corrupts clean text sequences and computes Levenshtein edit…, Builds an interleaved canvas with a noise level t skewed toward 1: For each…, Applies synthetic edits to token sequences and computes ground-truth edit…, Build a progressive denoising trajectory for a single (prompt, response) pair.…, Randomly replace some tokens with noise tokens., Randomly delete some tokens., Insert random noise tokens at random positions. (+10 more)

### Community 5 - "DLLMTokenizer"
Cohesion: 0.06
Nodes (24): DLLMTokenizer, Tensor, Encode a text string into token IDs (no padding/truncation)., Decode token IDs back to a string., Tokenize text(s) with padding and truncation. Returns a dict with 'input_ids'…, Check if a token ID corresponds to one of the edit operation tokens., Check if a token ID is a regular vocabulary token (not a special edit token)., Get the string name of an edit tag token. (+16 more)

### Community 6 - "Graphify Skill"
Cohesion: 0.09
Nodes (28): Add URL and Watch Folder Reference, Exports and Benchmark Reference, Extraction Subagent Spec, GitHub Clone and Cross-Repo Merge Reference, Commit Hook and CLAUDE.md Integration Reference, Query, Path, Explain Reference, Transcribe Video and Audio Reference, Incremental Update and Cluster-Only Reference (+20 more)

### Community 7 - "visualizer/app.js"
Cohesion: 0.16
Nodes (22): buildChart(), buildTimeline(), chipClass(), confToColor(), escHtml(), gotoStep(), hide(), highlightTimelineColumn() (+14 more)

### Community 8 - "static/app.js"
Cohesion: 0.16
Nodes (22): buildChart(), buildTimeline(), chipClass(), confToColor(), escHtml(), gotoStep(), hide(), highlightTimelineColumn() (+14 more)

### Community 9 - "prepare_translation_data.py"
Cohesion: 0.20
Nodes (14): append_word_dict(), hindi_fraction(), iitb_iter(), is_word(), main(), normalize(), quality_filter(), Fraction of characters in `s` that fall in the Devanagari range. (+6 more)

### Community 10 - "ARLM"
Cohesion: 0.22
Nodes (7): ARLM, Tensor, Return parameter groups with weight decay applied only to weights (not biases…, Autoregressive language model wrapper. Usage: model = ARLM("roberta-base") loss…, Args: tokenizer: ARLMTokenizer instance. backbone_name: HuggingFace causal-LM…, Standard causal-LM forward pass. Args: input_ids: Token IDs. attention_mask:…, Compute the next-token cross-entropy loss. Returns: (loss, loss_dict) where…

### Community 11 - "Discrete Diffusion Language Model (DLLM)"
Cohesion: 0.20
Nodes (12): DLLM Wiki Skill, DLLM Wiki, DLLM README, Discrete Diffusion Language Model (DLLM), Dual-Head Model, Edit Special Tokens, Forward Corruptor, Levenshtein Alignment (+4 more)

### Community 12 - "clean_lines"
Cohesion: 0.43
Nodes (6): clean_lines(), devanagari_fraction(), main(), normalize_ws(), NFKC normalize + collapse all whitespace runs., Split article text into cleaned, content-bearing lines.

### Community 13 - "_expand_1d"
Cohesion: 0.25
Nodes (9): compute_edit_accuracy(), delete_positions(), _expand_1d(), expand_sequence(), Tensor, Compute accuracy of the Tagger predictions., Deterministically expand a sequence by replacing each EXPAND-tagged position…, Helper: expand a 1D sequence. (+1 more)

### Community 15 - "torch"
Cohesion: 0.16
Nodes (18): load_arlm_state(), Module, ARLM Model: standard autoregressive language model. This is a thin wrapper…, Load an ARLM state dict, unwrapping DataParallel if present. Returns a message…, DLLM Utilities: Helper functions for tensor manipulation, logging, etc.…, Create a human-readable visualization of the edit operations. Returns a multi-…, Set random seed across torch, numpy, and Python random., set_seed() (+10 more)

### Community 17 - "ARLMTokenizer"
Cohesion: 0.12
Nodes (12): ARLMInference, ARLM Inference: standard autoregressive (left-to-right) generation. The public…, Extract the newly generated portion of `full_text` (the continuation after the…, Autoregressive sampling for the ARLM model. Usage: inference =…, ARLMTokenizer, ARLM Tokenizer: thin wrapper around a HuggingFace causal LM tokenizer. Unlike…, Wrap a HuggingFace causal-LM tokenizer with a DLLM-compatible surface., Args: base_model: HuggingFace model identifier for the base tokenizer.… (+4 more)

### Community 18 - "arlm/dataset.py"
Cohesion: 0.18
Nodes (8): build_labels(), ARLM Dataset: teacher-forced (prompt + response) sequences for an…, Build shifted next-token labels for teacher forcing. The label at position i is…, no_grad, ARLM Trainer: standard teacher-forced autoregressive training loop. The…, Main training loop. Args: train_loader: DataLoader yielding {'input_ids',…, Run evaluation on the validation set. Returns averaged metrics., Save model, optimizer, and training state.

### Community 19 - "ARLMTrainer"
Cohesion: 0.18
Nodes (10): make_collate(), Return a collate function that pads a batch to the longest sequence., ARLMTrainer, Module, Load model, optimizer, and training state from checkpoint., Trainer for the autoregressive language model. Usage: trainer =…, Initialize AdamW optimizer with proper weight-decay grouping., Linear warmup followed by linear decay (identical to DLLM). (+2 more)

### Community 23 - "._sample_filters"
Cohesion: 0.29
Nodes (5): no_grad, Tensor, Sample tokens from an (N, V) logits tensor (used by the benchmark)., Generate a continuation for the given prompt via autoregressive decoding. Args:…, Apply top-k and nucleus filtering to a (1, vocab) logits tensor.

### Community 25 - "ARLMDataset"
Cohesion: 0.29
Nodes (4): ARLMDataset, Dataset, Tensor, PyTorch Dataset of teacher-forced prompt+response sequences. Args: tokenizer:…

## Knowledge Gaps
- **32 isolated node(s):** `trajectory`, `TAG_COLOR`, `TAG_ORDER`, `trajectory`, `TAG_COLOR` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DLLMDataset` connect `DLLMDataset` to `DLLMTrainer`, `ForwardCorruptor`, `DLLMTokenizer`, `torch`, `arlm/dataset.py`, `ARLMDataset`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `DLLMTokenizer` connect `DLLMTokenizer` to `DLLM`, `DLLMTrainer`, `ForwardCorruptor`, `torch`, `arlm/dataset.py`, `ARLMDataset`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Why does `ForwardCorruptor` connect `ForwardCorruptor` to `DLLMTrainer`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `ARLMTokenizer` (e.g. with `ARLMDataset` and `ARLMInference`) actually correct?**
  _`ARLMTokenizer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `trajectory`, `TAG_COLOR`, `TAG_ORDER` to the rest of the system?**
  _32 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DLLM` be split into smaller, more focused modules?**
  _Cohesion score 0.05384150030248034 - nodes in this community are weakly interconnected._
- **Should `Model Architecture Wiki` be split into smaller, more focused modules?**
  _Cohesion score 0.05513784461152882 - nodes in this community are weakly interconnected._