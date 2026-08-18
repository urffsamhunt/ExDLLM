---
name: dllm-wiki
description: Reference and maintenance instructions for the Discrete Diffusion Language Model (DLLM) codebase wiki and architecture. Activate when inspecting, modifying, or explaining DLLM modules, data pipelines, model heads, loss functions, training, or inference strategies.
---

# DLLM Project Wiki Skill

This skill equips AI agents with direct reference guides and operational instructions for navigating, understanding, and maintaining the **DLLM (Discrete Diffusion Language Model)** codebase.

---

## 📌 Quick Reference Map

| Topic | File / Path | Wiki Reference |
| :--- | :--- | :--- |
| **Project Overview** | [configs/default.yaml](file:///home/sameer/Research/DLLM/configs/default.yaml) | [wiki/Home.md](file:///home/sameer/Research/DLLM/wiki/Home.md) |
| **Edit Tokenizer** | [dllm/tokenizer.py](file:///home/sameer/Research/DLLM/dllm/tokenizer.py) | [wiki/Architecture.md](file:///home/sameer/Research/DLLM/wiki/Architecture.md#1-edit-tokenizer-extension-dllmtokenizerpy) |
| **Dual-Head Model** | [dllm/model.py](file:///home/sameer/Research/DLLM/dllm/model.py) | [wiki/Architecture.md](file:///home/sameer/Research/DLLM/wiki/Architecture.md#3-dual-heads) |
| **Forward Corruptor (single-stage)** | [dllm/corruptor.py](file:///home/sameer/Research/DLLM/dllm/corruptor.py) | [wiki/Data-and-Corruption.md](file:///home/sameer/Research/DLLM/wiki/Data-and-Corruption.md#1-forward-corruptor-dllmcorruptorpy) |
| **Trajectory Corruptor (multi-stage)** | [dllm/corruptor.py — corrupt_prompt_response_trajectory](file:///home/sameer/Research/DLLM/dllm/corruptor.py) | [wiki/Data-and-Corruption.md — §4](file:///home/sameer/Research/DLLM/wiki/Data-and-Corruption.md#4-progressive-trajectory-corruption-corrupt_prompt_response_trajectory) |
| **Prompt-Response Canvas** | [dllm/corruptor.py](file:///home/sameer/Research/DLLM/dllm/corruptor.py#L98-L151) | [wiki/Data-and-Corruption.md](file:///home/sameer/Research/DLLM/wiki/Data-and-Corruption.md#prompt-response-canvas-corruption-corrupt_prompt_response) |
| **Dataset & Collate** | [dllm/dataset.py](file:///home/sameer/Research/DLLM/dllm/dataset.py) | [wiki/Data-and-Corruption.md](file:///home/sameer/Research/DLLM/wiki/Data-and-Corruption.md#3-dataset--batch-collation-dllmdatasetpy) |
| **Trainer Loop** | [dllm/trainer.py](file:///home/sameer/Research/DLLM/dllm/trainer.py) | [wiki/Training-and-Inference.md](file:///home/sameer/Research/DLLM/wiki/Training-and-Inference.md#1-training-setup-dllmtrainerpy--scriptstrainpy) |
| **Sub-Iteration Training** | [dllm/trainer.py — sub-iter loop](file:///home/sameer/Research/DLLM/dllm/trainer.py) | [wiki/Training-and-Inference.md — §5](file:///home/sameer/Research/DLLM/wiki/Training-and-Inference.md#5-sub-iteration-training-loop) |
| **Prompt Protection / Refinement** | [dllm/inference.py](file:///home/sameer/Research/DLLM/dllm/inference.py#L82-L150) | [wiki/Training-and-Inference.md](file:///home/sameer/Research/DLLM/wiki/Training-and-Inference.md#canvas-initialization--prompt-protection) |
| **Training Entrypoint** | [scripts/train.py](file:///home/sameer/Research/DLLM/scripts/train.py) | [wiki/Home.md](file:///home/sameer/Research/DLLM/wiki/Home.md#2-training-a-model) |
| **Generation Script** | [scripts/generate.py](file:///home/sameer/Research/DLLM/scripts/generate.py) | [wiki/Home.md](file:///home/sameer/Research/DLLM/wiki/Home.md#3-generating-text) |

---

## 🛠️ Instructions for AI Agents

When working in this repository:

### 1. Codebase Search & Inspection Rules
* Always refer to source files in `dllm/` before proposing modifications.
* Check [configs/default.yaml](file:///home/sameer/Research/DLLM/configs/default.yaml) when adjusting hyperparameters or corruption ratios.

### 2. Modifying Model Architecture
* If adding new edit operations or special tokens, update:
  1. `EDIT_SPECIAL_TOKENS` in [dllm/tokenizer.py](file:///home/sameer/Research/DLLM/dllm/tokenizer.py#L21-L28).
  2. `num_edit_tags` and `_init_tag_mapping` in [dllm/model.py](file:///home/sameer/Research/DLLM/dllm/model.py#L42-L123).
  3. `_execute_edits` in [dllm/inference.py](file:///home/sameer/Research/DLLM/dllm/inference.py#L207-L262).
  4. Corresponding Wiki pages in [wiki/Architecture.md](file:///home/sameer/Research/DLLM/wiki/Architecture.md).

### 3. Modifying Sub-Iteration Training
* To change the number of stages or noise schedule, edit `sub_iterations` in [configs/default.yaml](file:///home/sameer/Research/DLLM/configs/default.yaml).
* To adjust stage loss weighting, edit `sub_iteration_weights` (must be same length as `sub_iterations`).
* To disable sub-iteration training entirely, set `sub_iterations: null` in the config — the trainer and dataset automatically fall back to single-stage random-`t` mode.
* The validation dataset always uses single-stage mode regardless of config; do not pass `sub_iterations` to val `DLLMDataset`.
* Refer to [wiki/Data-and-Corruption.md §4](file:///home/sameer/Research/DLLM/wiki/Data-and-Corruption.md#4-progressive-trajectory-corruption) and [wiki/Training-and-Inference.md §5](file:///home/sameer/Research/DLLM/wiki/Training-and-Inference.md#5-sub-iteration-training-loop) for the full design.

### 4. Updating Documentation
* If altering dataset mechanisms, model heads, loss calculation, or inference logic, update the corresponding markdown file in `wiki/`.
* Maintain GitHub-style markdown file links using the `file:///` protocol.

---

## 🧪 Verification Commands

To verify setup and codebase health:
```bash
# 1. Run quick training check
python scripts/train.py --config configs/default.yaml --save_dir ./checkpoints

# 2. Test generation from checkpoint
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "The king" --show_trajectory
```
