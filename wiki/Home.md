# DLLM — Discrete Diffusion Language Model Wiki

Welcome to the **DLLM (Discrete Diffusion Language Model)** project wiki. This repository contains a PyTorch & HuggingFace implementation of a non-autoregressive, edit-based discrete diffusion language model.

---

## 📌 Repository Overview

Unlike standard autoregressive language models (which generate text left-to-right token by token), **DLLM** processes text iteratively in parallel. Starting from an empty or masked token canvas, DLLM applies discrete edit operations (such as keeping, deleting, replacing, inserting, or expanding tokens) across multiple refinement steps until the text reaches a high-confidence clean state.

### Key Architecture Highlights
* **Bidirectional Encoder Backbone**: Built on `roberta-base`, enabling all tokens to attend to left and right context simultaneously.
* **Extended Edit Vocabulary**: Includes 6 edit control tokens (`<KEEP>`, `<DELETE>`, `<REPLACE>`, `<INSERT>`, `<EXPAND>`, `<MASK>`).
* **Dual Heads**:
  1. **Tagger Head**: Predicts edit operations for every position.
  2. **Generator Head**: Predicts vocabulary tokens at positions marked for replacement or insertion, conditioned on tag embeddings.
* **Levenshtein Forward Corruptor**: Computes optimal Levenshtein edit paths between synthetically corrupted text and ground-truth text during training.

---

## 📚 Wiki Directory

* 📖 **[Home](file:///home/sameer/Research/DLLM/wiki/Home.md)**: Repository overview, structure, and quickstart guide.
* 🏗️ **[Architecture](file:///home/sameer/Research/DLLM/wiki/Architecture.md)**: Detailed description of the dual-head RoBERTa model, tag embeddings, and joint loss function.
* 🔄 **[Data & Corruption](file:///home/sameer/Research/DLLM/wiki/Data-and-Corruption.md)**: Forward corruption process, Levenshtein alignment algorithm, and dataset batching.
* 🚀 **[Training & Inference](file:///home/sameer/Research/DLLM/wiki/Training-and-Inference.md)**: Training configuration, gradient accumulation, iterative reverse denoising inference, and trajectory sampling.

---

## 📂 Project Structure

```
DLLM/
├── checkpoints_v2/           # Saved model checkpoints (.pt)
│   └── best_model.pt          # Only the best checkpoint is kept (no periodic saves)
├── configs/
│   └── default.yaml          # Master configuration file
├── data/
│   ├── tiny_shakespeare.txt  # Default dataset cache
│   └── en_hi_*.tsv           # IITB en-hi + MUSE dict English<->Hindi parallel data (translation.yaml)
├── dllm/                     # Core Python module
│   ├── __init__.py           # Package exports
│   ├── corruptor.py          # Forward corruption & Levenshtein alignment
│   ├── dataset.py            # PyTorch Dataset & collate_fn
│   ├── inference.py          # Iterative reverse denoising engine
│   ├── model.py              # Dual-head RoBERTa PyTorch module
│   ├── tokenizer.py          # Extended RoBERTa tokenizer
│   ├── trainer.py            # AdamW training loop with LR scheduler
│   └── utils.py              # Sequence expansion/contraction & visualization helpers
├── scripts/
│   ├── baseline_ar.py         # Autoregressive baseline (same backbone, next-token objective)
│   ├── generate.py            # Text generation script
│   ├── prepare_translation_data.py  # IITB en-hi + MUSE dict -> TSV pairs (both directions)
│   ├── train.py               # Model training entrypoint
│   └── translate.py           # Translate / BLEU-eval with a trained checkpoint
├── wiki/                     # Repository documentation wiki
├── .skills/
│   └── dllm-wiki/            # Reusable AI Agent Wiki Skill
├── requirements.txt          # Python dependencies
└── training.log              # Sample training logs
```

---

## ⚡ Quickstart

### 1. Requirements & Setup
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Training a Model
To launch training using the default configuration (`configs/default.yaml`):
```bash
python scripts/train.py --config configs/default.yaml --save_dir ./checkpoints_v2
```

### 3. Generating Text
To generate text from a trained checkpoint using iterative reverse denoising:
```bash
python scripts/generate.py --checkpoint checkpoints_v2/best_model.pt --prompt "The king" --show_trajectory
```
