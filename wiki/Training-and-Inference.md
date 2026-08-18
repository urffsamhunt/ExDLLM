# 🚀 Training & Inference Wiki

This page describes the training infrastructure and iterative reverse denoising inference engine for **DLLM**.

---

## 1. Training Setup (`dllm/trainer.py` & `scripts/train.py`)

Training is driven by [DLLMTrainer](file:///home/sameer/Research/DLLM/dllm/trainer.py#L24-L293).

### Key Training Parameters (`configs/default.yaml`)

```yaml
training:
  batch_size: 8
  learning_rate: 0.00005
  weight_decay: 0.01
  warmup_steps: 500
  max_steps: 50000
  gradient_accumulation_steps: 4
  log_every: 10
  eval_every: 500
```

Only the best validation checkpoint is stored (`best_model.pt`) — there are no periodic checkpoints, so the save directory stays small.

> **Note**: `inference.keep_threshold` and `inference.confidence_threshold` are declared in `configs/default.yaml` but are not consumed by `DLLMInference`; edit decisions are made purely by `argmax` over the tagger logits.

### Optimizer & Parameter Grouping
Uses `AdamW` optimizer. To avoid over-regularization, weight decay is applied only to 2D matrix weights, while biases and LayerNorm parameters are assigned `weight_decay = 0.0`:

```python
no_decay = ["bias", "LayerNorm.weight"]
```

### Learning Rate Schedule
* **Warmup**: Linear warmup from $0.0 \to \text{lr}$ over `warmup_steps` (500 steps).
* **Decay**: Linear decay from $\text{lr} \to 0.0$ over remaining steps up to `max_steps`.

### Gradient Accumulation & Clipping
* Gradients are accumulated over `gradient_accumulation_steps` before executing `optimizer.step()`.
* Norm clipping (`torch.nn.utils.clip_grad_norm_`) caps gradient norm at `1.0`.

### Scheduled Sampling (self-draft training)
Training is single-pass (one corruption, one forward/backward — the noise level `t` plays the role of the inference step index). To close the gap between training corruption (uniform random noise) and the model's *own* decode errors (structured, e.g. repetition), with prob `scheduled_sampling_prob` (0.3) the trainer replaces the noise/mask tokens at `REPLACE` positions with the model's own top-k/top-p sampled tokens (a no-grad eval-mode forward, exactly like inference). The labels are unchanged, so the model learns to apply the `REPLACE` edit toward the clean token on inputs that look like its own decoding output. Prompt positions are never touched.

---

## 2. Iterative Reverse Denoising Inference (`dllm/inference.py`)

At inference time, text generation is performed via multi-step iterative refinement using [DLLMInference](file:///home/sameer/Research/DLLM/dllm/inference.py#L43-L359).

### Canvas Initialization & Prompt Protection
Given a text prompt (e.g. `"KING EDWARD:"`), generation proceeds in two stages:

1. **Length prediction**: a first forward pass over a *prompt-only* canvas yields the length-head logits; the response length `M` is set to `round(E[length])` — the expectation over the length distribution, which is smoother than `argmax` and less mode-locked by the imbalanced length classes (override with `--target_length` if desired).
2. **Canvas construction**: `<MASK>` tokens are interleaved beside every prompt token, followed by exactly `M` response slot pairs of `<MASK>` tokens, and a single trailing `</s>` (`eos_id`):
```
Canvas: [ <s>, <MASK>, "KING", <MASK>, "EDWARD", <MASK>, ":", <MASK>, <MASK>, ..., <MASK>, </s> ]
         |-------- prompt section --------| |-- M response slot pairs --|
```
* **Predicted Length, Not a Cutoff**: `M` is the *initial* number of response masks; `INSERT`/`EXPAND` can grow the canvas beyond it, and nothing is pruned — the model fills what it is given.
* **Fully Interleaved Masks**: `<MASK>` tokens are injected beside every token in the prompt and response region.
* **Single Trailing EOS**: `</s>` is placed strictly at the very end of the sequence.
* **Prompt Protection**: Position types `bos`, `eos`, and `prompt_imask` are hard-locked to `KEEP` during edit execution; a `KEEP` prediction on a `prompt_tok` position is forced to `REPLACE` with itself. Prompt tokens are never deleted or modified.

---

## 3. The Refinement Loop

For each iteration $t = 1 \dots N_{\text{max}}$:

```
                          ┌────────────────────────┐
                          │   Current Token Canvas │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  Forward Pass DLLM     │
                          └─────┬────────────┬─────┘
                                │            │
                                ▼            ▼
                        Tag Predictions   Generator Logits
                         (KEEP/DEL/REP/     (Top-k / Top-p
                           INS/EXPAND)        Sampling)
                                │            │
                                └─────┬──────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │     Execute Edits      │
                          │   (Canvas Mutation)    │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   Updated Token Canvas │
                          └────────────────────────┘
```

### 1. Model Forward Pass
Input tensor is passed through DLLM in evaluation mode (`model.eval()`).
* `tag_logits`: Determines predicted edit index ($0..4$) per token.
* `gen_logits`: Provides probability distribution over vocabulary.

### 2. Token Sampling (`_sample_tokens`)
At generation positions (`REPLACE`, `INSERT`), tokens are sampled using:
* **Temperature**: Controls entropy ($T = 1.0$).
* **Top-k Filtering**: Retains top $k$ candidates ($k = 50$).
* **Top-p (Nucleus) Filtering**: Retains candidates within cumulative probability $p$ ($p = 0.9$).

### 3. Deterministic Edit Execution (`_execute_edits`)
Edits are executed sequentially to mutate the canvas:

| Tag Index | Edit Action | Canvas Mutation |
| :--- | :--- | :--- |
| `0` (KEEP) | Keep token as-is | `result.append(tok_id)` |
| `1` (DELETE) | Drop token | *(skipped)* |
| `2` (REPLACE) | Substitute token | `result.append(gen_tok)` |
| `3` (INSERT) | Insert sampled token | `result.append(gen_tok); result.append(tok_id)` |
| `4` (EXPAND) | Expand into two `<MASK>` tokens | `result.append(<MASK>); result.append(<MASK>)` |

*Note*: A `KEEP` prediction on a `<MASK>` position leaves the `<MASK>` in the canvas; any remaining `<MASK>` tokens are stripped by `_decode_canvas(clean=True)` when rendering the final text.

---

## 4. Convergence & Trajectory

### Convergence Criteria
The refinement loop terminates early (from the second iteration onward) when either:
1. All positions of type `response_*` predict `KEEP`, or
2. The canvas is unchanged from the previous iteration (no edits were executed).

### Trajectory Inspection
When `return_trajectory=True` or `--show_trajectory` flag is passed to `scripts/generate.py`, intermediate canvas states at each step are recorded for analysis.

---

## 5. Sub-Iteration Training Loop

To align training with the iterative denoising trajectory at inference, each (prompt, response) pair produces K canonically ordered canvases (default K=5 at noise levels `[1.0, 0.8, 0.6, 0.4, 0.2]`). The trainer iterates through all K stages per batch before stepping the optimizer.

### Configuration (`configs/default.yaml`)

```yaml
training:
  sub_iterations: [1.0, 0.8, 0.6, 0.4, 0.2]   # noise levels per stage
  sub_iteration_weights: [2.0, 1.5, 1.0, 0.75, 0.5]  # loss weights per stage
  max_steps: 25000               # halved from 50000 (5 sub-iters ≈ 2.5× more FLOPs/step)
  gradient_accumulation_steps: 2 # halved from 4 to maintain compute budget
```

### Inner Training Loop (per batch)

```
For each batch B:
  is_multistage = (noisy_ids.dim() == 3)  # (B, K, S) vs (B, S)
  K = batch.shape[1] if is_multistage else 1

  # Scheduled sampling decided ONCE per batch (not per stage)
  do_self_draft = random() < scheduled_sampling_prob

  For k = 0 .. K-1:
    sub_batch = batch[:, k, :]   # (B, S) slice; attention_mask/resp_length shared
    noisy_ids = self_draft(sub_batch) if do_self_draft else sub_batch.noisy_ids
    outputs = model(noisy_ids, ...)
    loss_k = compute_loss(outputs, ...)
    scaled = loss_k × w_k / gradient_accumulation_steps
    scaled.backward()   # independent graph per stage; no retain_graph needed

  accumulated_steps += 1
  if accumulated_steps >= gradient_accumulation_steps:
    clip_grad + optimizer.step() + scheduler.step() + zero_grad
    global_step += 1
```

### Stage Loss Weighting

Weights `w_k` are normalized (sum to 1.0) before use. The total gradient magnitude across all K stages equals one unweighted backward pass. By default, stage 0 (t=1.0, fully masked) receives 4× more weight than stage 4 (t=0.2, nearly clean), preserving the `t_skew` curriculum from the original single-stage training.

### Throughput Budget

With K=5 sub-iterations, each optimizer step processes ~5× as many forward/backward passes as single-stage training. The default config compensates by halving `max_steps` (50000 → 25000) and `gradient_accumulation_steps` (4 → 2), keeping total training FLOPs approximately constant.

> **Note**: Validation always uses a **single-stage** dataset (no `sub_iterations`) for fast checkpointing. The multi-stage regime applies only to the training loader.

