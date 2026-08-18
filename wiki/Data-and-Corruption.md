# 🔄 Data & Corruption Pipeline Wiki

The training of **DLLM** relies on synthetically degrading clean text and computing ground-truth edit paths to train the model to reverse degradation.

---

### Prompt-Response Canvas Corruption (`corrupt_prompt_response`)

In `prompt_response` mode, the corruptor builds an interleaved canvas that preserves the prompt while corrupting only the response portion, using a noise level `t` sampled from `1 - U(0,1)^t_skew` (skewed toward full masking so the model trains on near-fully-masked canvases like the inference start):

1. **Prompt Section**: `[BOS] <MASK> p1 <MASK> p2 ... <MASK>` — prompt tokens are kept clean and uncorrupted. Each interleaving `<MASK>` is tagged `<KEEP>`, and each prompt token is tagged `<REPLACE>` with **itself** as the generator target (self-replacement).
2. **Response Section**: A canvas of slot pairs `(r-slot, iMask)` with exactly `M = answer-length` slots — the canvas is sized to the response, with **no capacity padding and no pruning**. M is the *initial* mask count only: at inference it is predicted from the prompt by the model's length head, and `INSERT`/`EXPAND` can still grow the canvas beyond it. For each slot with clean target token `r_i` and a random `roll ~ Uniform(0, 1)`:
   * `roll > t` → the clean token `r_i` is placed in the slot, tagged `<KEEP>`.
   * `roll < t * mask_ratio` → the slot holds `<MASK>`, tagged `<EXPAND>` (with prob `expand_prob`) or `<REPLACE>` (target `r_i`).
   * otherwise → the slot holds a random noise token, tagged `<REPLACE>` (target `r_i`).
   The structural interleaving mask after each slot is tagged `<INSERT>` (target `r_{i+1}`, with prob `insert_prob`) or `<KEEP>`. Because the canvas holds exactly the answer, `<DELETE>` never appears in training labels — the tagger learns to fill, never to prune.
3. **Shortage Training (growth)**: with prob `shortage_prob`, the canvas is built with 1-2 *fewer* slots than the answer. The last iMask is then tagged `<INSERT>` (target = the first overflow token) and/or the last slot tagged `<EXPAND>`, so the model learns *when to grow* — the tagger sees a canvas smaller than the prompt's expected answer and must insert/expand to fit the overflow.
4. **Trailing EOS**: `[EOS]` closes the canvas, tagged `<KEEP>`.
5. **Generator Targets**: The corruptor returns a `pos_to_clean` map recording the exact clean token for every `REPLACE`/`INSERT` position, so `dataset.py` can build generator labels without heuristics.

The skewed noise level produces a curriculum across samples: high `t` gives near-fully-masked canvases (scratch generation), low `t` gives clean slots (verification), and intermediate `t` gives a mixture of masks, noise tokens, and clean tokens (in-place refinement).

---

## 2. Levenshtein Alignment (`_align`)

After corrupting a clean sequence $C$ into a noisy sequence $N$, the corruptor computes the minimum-edit alignment from $N \to C$ using `Levenshtein.editops`.

> **Note**: This path is used by the generic `corrupt()` method, which serves the `unconditional` dataset mode. In `prompt_response` mode the tags are derived directly from the slot corruption scheme above (no Levenshtein alignment).

### Token-to-Unicode Character Mapping
Standard Levenshtein string algorithms operate on characters. To align token ID sequences:
1. Each unique token ID is mapped to a unique Unicode character in the Private Use Area (`U+E000` onward).
2. `Levenshtein.editops(noisy_str, clean_str)` returns edit operations: `'replace'`, `'delete'`, `'insert'`.
3. Operations are mapped back to token positions in $N$:

```
Noisy Sequence (N):   [ "The", "cat", "bad", "sat" ]
Clean Target   (C):   [ "The", "black", "cat", "sat" ]
                                  │
                                  ▼ Levenshtein Alignment
Tag Targets:          [  KEEP,   INSERT,  DELETE,  KEEP  ]
                        (·)       (I)       (D)     (·)
```

### Special Token Target Overrides
* **`<EXPAND>`**: Any position containing `<EXPAND>` is explicitly tagged as `EXPAND` (4).
* **`<MASK>`**: Any position containing `<MASK>` is explicitly tagged as `REPLACE` (2) to ensure `<MASK>` tokens are filled by the Generator head.

---

## 3. Dataset & Batch Collation (`dllm/dataset.py`)

[DLLMDataset](file:///home/sameer/Research/DLLM/dllm/dataset.py#L32-L389) handles text loading, chunking, and batching.

### Dynamic Corruption
Corruption is applied dynamically inside `__getitem__`. This means each epoch presents the model with varied noisy sequence variations for the same underlying text.

### Returned Tensors per Sample
* `noisy_ids`: `(max_length,)` — Token IDs of the corrupted sequence (padded with `pad_id`).
* `attention_mask`: `(max_length,)` — Binary mask (`1` for real tokens, `0` for padding).
* `tag_labels`: `(max_length,)` — Target edit tags (`0..4`, padded with `-100`).
* `gen_labels`: `(max_length,)` — Target vocabulary token IDs for `REPLACE`/`INSERT` positions (padded with `-100`).
* `gen_mask`: `(max_length,)` — Boolean mask (`True` where `gen_labels` is active).
* `prompt_mask`: `(max_length,)` — Boolean mask of prompt positions (feeds the length head's prompt-only pooling).
* `resp_length`: scalar — Target response length `M` for the length head (`-100` in unconditional mode, or when the answer was truncated at `max_response_length` and has no true length).

### Collate Function (`collate_fn`)
Filters out empty/invalid samples and stacks tensors into batch dimensions.

- **Single-stage batches**: stacks into `(B, S)` — existing behavior unchanged.
- **Multi-stage batches** (when `sub_iterations` is set on the training dataset): stacks `(K, S)` per-sample tensors into `(B, K, S)`. `attention_mask` (`(B, S)`) and `resp_length` (`(B,)`) are never stage-duplicated — they are shared across all K stages of the same sample.

---

## 4. Progressive Trajectory Corruption (`corrupt_prompt_response_trajectory`)

To align training with inference, the corruptor can generate an ordered **trajectory** of K canvases for a single (prompt, response) pair, mimicking the progressive denoising that occurs at inference time (step 0: all masked → step K: nearly clean).

### Algorithm

1. **Sample `shortage_k` once** (same logic as single-stage): `shortage_k ∈ {0, 1, 2}` with prob `shortage_prob`. This is shared across all K stages, guaranteeing every stage has the same `n_slots = n_resp - shortage_k` and therefore the same canvas length — a hard requirement for stacking into `(K, S)` tensors.

2. **Build a single permutation `π`** over `[0, ..., n_slots - 1]`. This permutation defines the **order in which response slots are revealed** as clean across stages.

3. **For each noise level `t` in `stages`** (e.g. `[1.0, 0.8, 0.6, 0.4, 0.2]`):
   - `n_revealed = round((1 - t) × n_slots)` — number of slots that should appear clean.
   - `revealed_set = set(π[:n_revealed])` — cumulative prefix of the permutation.
   - Build the canvas:
     - **Slots in `revealed_set`**: place the clean token, tag = `KEEP`.
     - **Remaining slots**: corrupt with `<MASK>` (`REPLACE`/`EXPAND`) or noise token (`REPLACE`) — using `mask_ratio` to split between the two, without a separate per-slot roll against `t_noise` (corruption is now determined entirely by set membership).
   - Prompt section and shortage INSERT/EXPAND logic are identical to `corrupt_prompt_response`.

```
Example trajectory for n_slots=5, π=[2,0,4,1,3], stages=[1.0, 0.8, 0.6, 0.4, 0.2]:

Stage  t    n_revealed  revealed_set  Canvas (slot states)
─────────────────────────────────────────────────────────
  0   1.0      0         {}           [M, M, M, M, M]   ← fully masked
  1   0.8      1         {2}          [M, M, c, M, M]   ← slot 2 clean
  2   0.6      2         {2,0}        [c, M, c, M, M]   ← slots 0,2 clean
  3   0.4      3         {2,0,4}      [c, M, c, M, c]   ← slots 0,2,4 clean
  4   0.2      4         {2,0,4,1}   [c, c, c, M, c]   ← slots 0,1,2,4 clean
```

The revealed set grows monotonically across stages: if a slot is clean at stage k, it remains clean at stage k+1. This directly mirrors inference, where a filled token is never re-masked.

### Stage Loss Weighting

To preserve the `t_skew` curriculum (which originally biased training toward high-masking canvases), each stage's loss contribution is scaled by a configurable weight before averaging:

$$\mathcal{L}_{\text{total}} = \frac{\sum_k w_k \cdot \mathcal{L}_k}{\sum_k w_k}$$

Default weights `[2.0, 1.5, 1.0, 0.75, 0.5]` upweight the fully-masked stage (where cold-start generation is hardest) and downweight the nearly-clean stages.

