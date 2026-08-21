# Discrete Diffusion Language Model (DLLM)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)

**DLLM** is a PyTorch implementation of a **non‑autoregressive, edit‑based discrete diffusion language model** that generates text through iterative parallel refinement. Unlike standard left‑to‑right autoregressive models, DLLM starts from an empty or masked token canvas and applies discrete edit operations (keep, delete, replace, insert, expand) across multiple refinement steps until a high‑confidence clean state is reached.

The model uses a bidirectional RoBERTa/XLM‑RoBERTa backbone with dual heads (tagger + generator) and is trained via a forward corruption process that computes optimal Levenshtein edit paths between corrupted and clean text.

---

## Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [Architecture Overview](#-architecture-overview)
- [Data & Corruption](#-data--corruption)
- [Training](#-training)
- [Inference](#-inference)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Visualizer](#-visualizer)
- [Autoregressive Baseline](#-autoregressive-baseline)
- [Wiki](#-wiki)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

## ✨ Key Features

- **Non‑autoregressive generation**: Parallel edit prediction over the whole sequence, enabling faster inference and better global coherence.
- **Bidirectional conditioning**: Full context attention from both left and right tokens (RoBERTa encoder).
- **Extended edit vocabulary**: Six special edit tokens (`<KEEP>`, `<DELETE>`, `<REPLACE>`, `<INSERT>`, `<EXPAND>`, `<MASK>`) for fine‑grained control.
- **Levenshtein‑based forward corruptor**: Synthetically degrades clean text and provides ground‑truth edit labels for training.
- **Prompt‑response mode**: Preserves prompt tokens as uncorrupted conditioning context; only the response region is refined.
- **Progressive trajectory training**: Aligns training with inference by generating a trajectory of canvases at decreasing noise levels (multi‑stage training).
- **Interactive visualizer**: Flask‑based web UI to explore the denoising trajectory step‑by‑step.
- **Translation support**: Configurable for English‑Hindi (or any parallel) translation using XLM‑RoBERTa.
- **Autoregressive baseline**: Included for comparison (causal RoBERTa trained on same data).

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/DLLM.git
cd DLLM
pip install -r requirements.txt
```

**Dependencies**: PyTorch (≥2.0), Transformers (≥4.30), Datasets, python‑Levenshtein, NumPy, PyYAML, tqdm.

**Optional**: The visualizer requires Flask (`pip install flask`).

---

## 📊 Datasets

- **Tiny Shakespeare**: The default dataset (`data/tiny_shakespeare.txt`) is automatically loaded via Hugging Face Datasets (`datasets` library). The dataset is split into dialogue pairs (prompt‑response) for training.
- **Custom parallel data**: For translation tasks, you can prepare a TSV file with source‑target pairs (one per line). Use `scripts/prepare_translation_data.py` to build the training set from the IIT Bombay English‑Hindi corpus plus the MUSE/ARRIVAL word dictionary.

---

## ⚡ Quick Start

### 1. Train a model (Tiny Shakespeare)

```bash
python scripts/train.py --config configs/default.yaml --save_dir ./checkpoints_v2
```

The default configuration uses `roberta‑base`, `prompt_response` mode on the `tiny_shakespeare` dataset. Only the best validation checkpoint is saved (`best_model.pt`).

### 2. Generate text from a trained checkpoint

```bash
python scripts/generate.py \
  --checkpoint checkpoints_v2/best_model.pt \
  --prompt "The king" \
  --show_trajectory
```

The `--show_trajectory` flag prints the intermediate canvas states at each refinement step.

### 3. English ↔ Hindi translation

Prepare parallel data (see `scripts/prepare_translation_data.py`) and train with:

```bash
python scripts/train.py --config configs/translation.yaml --save_dir ./checkpoints_trans
```

Evaluate BLEU with:

```bash
python scripts/translate.py \
  --checkpoint checkpoints_trans/best_model.pt \
  --config configs/translation.yaml \
  --test_path data/en_hi_test.tsv
```

### 4. Monolingual pretraining (English / Hindi)

Before translation/ARLM benchmark fine-tuning, the XLM‑RoBERTa backbone is adapted
on clean, monolingual text so it learns the DLLM **unconditional denoising**
objective (reversing length-changing corruption on raw text) — no
prompt/response pairing. The model thereby acquires the edit vocabulary
(`KEEP`/`DELETE`/`REPLACE`/`INSERT`/`EXPAND`) and the parallel-denoising
dynamics that masked-LM pretraining cannot express.

#### Step 1 — Prepare the data

`scripts/prepare_pretrain_data.py` streams the HuggingFace `wikimedia/wikipedia`
dataset (already stripped of wiki markup) for a single language, applies light
cleaning, and writes one cleaned text chunk per line:

```bash
# English
python scripts/prepare_pretrain_data.py --lang en --max_docs 200000 --out data/pretrain_en.txt

# Hindi (gated to predominantly Devanagari text)
python scripts/prepare_pretrain_data.py --lang hi --max_docs 200000 --out data/pretrain_hi.txt
```

Cleaning applied on top of the HF preprocessing:

- NFKC unicode normalization + whitespace collapse.
- Drop section-heading boilerplate (`References`, `See also`, etc.).
- Drop URL / template / code-fence residue.
- Drop very short stub lines.
- **Hindi only:** keep lines that are ≥70% Devanagari, so English / Hinglish /
  Romanized text is excluded from Hindi grammar training. (This threshold is
  the `devanagari_frac` value in the script's `LANG_DEFAULTS`; it is not exposed
  as a CLI flag.)
- Exact-line dedup (disable with `--no_dedup`).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--lang` | *(required)* | `en` or `hi`. |
| `--config` | `20231101.<lang>` | `wikimedia/wikipedia` snapshot; override if HF adds a newer date. |
| `--split` | `train` | Dataset split to stream. |
| `--max_docs` | `200000` | Max articles to stream (`None` = no cap). |
| `--out` | *(required)* | Output plain-text file. |
| `--no_dedup` | off | Disable exact-line dedup. |

> **Verify `--config`**: Wikipedia snapshots are date-suffixed and can change;
> check the live `wikimedia/wikipedia` dataset card for the current date string
> (or any other valid language config) before running.

#### Step 2 — Train

```bash
python scripts/train.py --config configs/pretrain_en.yaml --save_dir ./checkpoints_pretrain_en
python scripts/train.py --config configs/pretrain_hi.yaml --save_dir ./checkpoints_pretrain_hi
```

The pretraining configs set `mode: unconditional` (no length-head supervision,
`scheduled_sampling_prob: 0`), use the `xlm-roberta-base` backbone, and point
`data.dataset_name` at the file written in Step 1. To keep the runs small/clean
(per the "no very large datasets" preference), cap `--max_docs` rather than
streaming full Wikipedia.

After pretraining, load the checkpoint and continue with task fine-tuning
(e.g. `configs/translation.yaml` for en↔hi).

---

## 🏗️ Architecture Overview

DLLM consists of three main components:

1. **Tokenizer extension** (`dllm/tokenizer.py`): Wraps RoBERTa tokenizer and adds six edit‑specific special tokens.
2. **Dual‑head RoBERTa model** (`dllm/model.py`):
   - **Tagger head**: Predicts edit operation (KEEP/DELETE/REPLACE/INSERT/EXPAND) for every position.
   - **Generator head**: Predicts vocabulary tokens at REPLACE/INSERT positions, conditioned on tag embeddings.
   - **Length head**: Predicts the response length from the prompt (prompt‑only pooling).
3. **Forward corruptor** (`dllm/corruptor.py`): Synthetically corrupts clean text and computes Levenshtein edit paths (or uses a slot‑based corruption for prompt‑response mode) to produce training labels.

**Prompt protection**: During both training and inference, prompt tokens are never deleted or modified; they are preserved as uncorrupted conditioning context.

**Loss function**: Joint cross‑entropy over tag, generator, and length predictions, with optional per‑class weighting and length‑smoothing.

---

## 🔄 Data & Corruption

- **Dynamic corruption**: Each epoch presents varied noisy versions of the same text.
- **Levenshtein alignment**: For unconditional mode, token sequences are aligned using Levenshtein edit operations.
- **Slot‑based corruption (prompt_response)**: The response region is split into exactly `M` slots (where `M` is the answer length). Each slot is corrupted with a noise level `t` sampled from a skewed distribution (biased toward full masking).
- **Shortage training**: With probability `shortage_prob`, the canvas is built with 1‑2 fewer slots than the answer, forcing the model to learn INSERT/EXPAND operations.
- **Progressive trajectory**: For multi‑stage training, a canonical order of slot revelation is fixed, and K canvases at decreasing noise levels are generated, directly mimicking the inference denoising trajectory.

---

## 🚀 Training

The trainer (`dllm/trainer.py`) supports:

- **AdamW with weight‑decay grouping** (no decay for biases/LayerNorm).
- **Linear warmup + decay** learning rate schedule.
- **Gradient accumulation** and clipping.
- **Scheduled sampling**: With probability `scheduled_sampling_prob`, noise tokens at REPLACE positions are replaced by the model’s own sampled tokens (self‑draft training).
- **Multi‑stage training**: When `sub_iterations` is set, each sample produces K canvases; loss is weighted across stages (higher weight for fully‑masked stages).
- **BLEU evaluation**: For translation tasks, BLEU can be computed on a held‑out test set at regular intervals.

**Configuration**: All hyperparameters are defined in YAML files (`configs/`). Adjust batch size, learning rate, corruption ratios, noise levels, etc.

**Metrics logging**: During training, the trainer appends per-step metrics (loss components, learning rate) as newline-delimited JSON to `training_metrics.jsonl` inside the save directory. Validation metrics are logged with a `val_` prefix. This file is designed for plotting loss-vs-step/epoch curves; disable it by passing `metrics_path=None` to `trainer.train(...)`.

---

## 🧠 Inference

The inference engine (`dllm/inference.py`) performs iterative reverse denoising:

1. **Length prediction**: Forward pass over a prompt‑only canvas yields expected response length `M`.
2. **Canvas construction**: Build interleaved canvas with `<MASK>` tokens beside each prompt token and `M` response slot pairs.
3. **Refinement loop** (up to `max_iterations` steps):
   - Forward pass → tag logits + generator logits.
   - Sample tokens for REPLACE/INSERT positions (temperature, top‑k, top‑p).
   - Execute edits sequentially (keep, delete, replace, insert, expand).
   - Stop early if all response positions predict KEEP or canvas unchanged.
4. **Trajectory recording**: Intermediate canvas states can be saved for analysis.

---

## ⚙️ Configuration

Example configuration (see `configs/default.yaml`):

```yaml
model:
  backbone: "roberta-base"
  tag_weights: [0.7, 0.8, 1.0, 2.0, 4.0]   # KEEP, DELETE, REPLACE, INSERT, EXPAND
  len_smoothing: 0.15

training:
  batch_size: 8
  learning_rate: 5e-5
  warmup_steps: 500
  max_steps: 25000
  sub_iterations: [1.0, 0.8, 0.6, 0.4, 0.2]   # multi‑stage noise levels
  sub_iteration_weights: [2.0, 1.5, 1.0, 0.75, 0.5]

data:
  mode: "prompt_response"
  max_prompt_length: 48
  max_response_length: 48
  corruption:
    replace_ratio: 0.10
    delete_ratio: 0.05
    insert_ratio: 0.05
    expand_ratio: 0.05
    mask_ratio: 0.80
    t_skew: 2.0
    shortage_prob: 0.35
```

For translation, switch the backbone to `xlm‑roberta‑base` and adjust batch size (see `configs/translation.yaml`).

---

## 📂 Project Structure

```
DLLM/
├── checkpoints_v2/               # Saved model checkpoints (.pt)
├── configs/                      # YAML configuration files
│   ├── default.yaml
│   ├── translation.yaml
│   └── translation_kaggle.yaml
├── data/                         # Datasets (tiny_shakespeare, en‑hi TSV files)
├── dllm/                         # Core DLLM module (discrete diffusion)
│   ├── __init__.py
│   ├── corruptor.py              # Forward corruption & Levenshtein alignment
│   ├── dataset.py                # PyTorch Dataset & collate_fn
│   ├── inference.py              # Iterative reverse denoising engine
│   ├── model.py                  # Dual-head RoBERTa model
│   ├── tokenizer.py              # Extended RoBERTa tokenizer
│   ├── trainer.py                # AdamW training loop with LR scheduler
│   └── utils.py                  # Sequence expansion/contraction & helpers
├── arlm/                         # Core ARLM module (autoregressive baseline)
│   ├── __init__.py
│   ├── dataset.py                # Teacher-forced prompt+response sequences
│   ├── inference.py              # Left-to-right autoregressive generation
│   ├── model.py                  # Causal LM wrapper (RobertaForCausalLM / GPT2)
│   ├── tokenizer.py              # Standard causal-LM tokenizer wrapper
│   └── trainer.py                # Teacher-forced next-token training loop
├── scripts/
│   ├── train.py                  # DLLM training entrypoint
│   ├── generate.py               # DLLM text generation script
│   ├── translate.py              # Translation / BLEU evaluation
│   ├── baseline_ar.py            # Legacy autoregressive baseline (single script)
│   ├── train_arlm.py             # ARLM training entrypoint
│   ├── generate_arlm.py          # ARLM text generation script
│   ├── benchmark.py              # DLLM vs ARLM benchmark on held-out prompts
│   ├── prepare_translation_data.py
│   └── prepare_pretrain_data.py  # Monolingual Wikipedia pretraining data (en/hi)
├── wiki/                         # Detailed documentation (architecture, data, training, inference)
├── visualizer/                   # Flask web UI for interactive denoising
│   ├── app.py
│   └── static/
├── requirements.txt
└── README.md                     # This file
```

---

## 🌐 Visualizer

A Flask‑based web UI is included to interactively explore the denoising trajectory:

```bash
python visualizer/app.py --checkpoint checkpoints_v2/best_model.pt
```

Open `http://127.0.0.1:5000` in your browser, enter a prompt, and watch the model refine the canvas step‑by‑step.

---

## 🔁 Autoregressive Baseline (ARLM)

For comparison, the repository includes a full autoregressive baseline module
(`arlm/`) that trains a standard causal LM (default: causal-masked RoBERTa via
`RobertaForCausalLM`, reusing the pretrained LM head) with a teacher-forced
next-token objective on the **same dialogue-pair data** as the DLLM. The
optimizer, LR schedule, gradient accumulation, and checkpoint format mirror
the DLLM trainer so the two are directly comparable.

Train it:

```bash
python scripts/train_arlm.py --config configs/arlm.yaml --save_dir ./models_ar
```

Generate samples:

```bash
python scripts/generate_arlm.py --checkpoint ./models_ar/best_model.pt --prompt "MENENIUS:"
```

### Benchmarking DLLM vs ARLM

Once you have trained both a DLLM checkpoint and an ARLM checkpoint, run the
benchmark harness to compare them on the same held-out prompts:

```bash
python scripts/benchmark.py \
  --dllm_checkpoint ./checkpoints_v2/best_model.pt \
  --arlm_checkpoint ./models_ar/best_model.pt \
  --config configs/default.yaml \
  --arlm_config configs/arlm.yaml \
  --n 50
```

The benchmark reports per-model parameter counts, generation wall-time
(tokens/sec), BLEU / chrF against the reference responses (when `sacrebleu` is
installed), and side-by-side sample generations for qualitative inspection.

A legacy single-script autoregressive baseline also exists at
`scripts/baseline_ar.py`; the `arlm/` module is the maintained, structured
replacement.

---

## 📚 Wiki

Detailed documentation is available in the `wiki/` directory:

- [Home](wiki/Home.md) – Repository overview and quickstart.
- [Architecture](wiki/Architecture.md) – Model architecture, tokenizer, dual heads, loss function.
- [Data & Corruption](wiki/Data-and-Corruption.md) – Forward corruption, Levenshtein alignment, dataset batching.
- [Training & Inference](wiki/Training-and-Inference.md) – Training configuration, iterative denoising, sub‑iteration training.

---

## 📄 License

This project is released under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgements

- Hugging Face `transformers` and `datasets` libraries.
- RoBERTa and XLM‑RoBERTa models.
- The `python‑Levenshtein` package for edit‑path computation.
- The Tiny Shakespeare dataset (used for demonstration).

---

## 📧 Contact

For questions, please open an issue on the GitHub repository.