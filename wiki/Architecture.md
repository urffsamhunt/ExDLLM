# 🏗️ Model Architecture Wiki

The Discrete Diffusion Language Model (**DLLM**) uses a bidirectional Transformer encoder backbone paired with two specialized heads for non-autoregressive text refinement.

---

## 🔍 Architecture Overview

```
                      ┌──────────────────────────────────────────┐
                      │          Input Interleaved Canvas        │
                      │ <s> Prompt [EOS] [MASK] r1 [MASK] </s>   │
                      └────────────────────┬─────────────────────┘
                                           │
                                           ▼
                      ┌──────────────────────────────────────────┐
                      │          RoBERTa Backbone                │
                      │   (Full Bidirectional Conditioning)      │
                      └────────────────────┬─────────────────────┘
                                           │  (hidden_states)
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
         ┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
         │     Tagger Head     │  │    Length Head       │  │   Generator Head    │
         │  (Predicts edits on │  │ (Prompt-only pooled  │  │ (Hidden + Tag Embed │
         │   Response region)  │  │  hidden -> 1..max)   │  │  -> Vocab Projection│
         └──────────┬──────────┘  └──────────┬───────────┘  └──────────┬──────────┘
                    │                       │                          │
                    ▼                       ▼                          ▼
            Tag Logits (5)            Length Logits (M)          Vocab Logits
         [KEEP, DELETE, REPLACE,      [P(M | prompt)]            [P(w | prompt, canvas)]
          INSERT, EXPAND]
```

### Prompt Protection Principle
Prompt tokens are strictly preserved as uncorrupted conditioning context.

* **Training**: the canvas is built as `[BOS] <MASK> p1 <MASK> p2 ... <MASK> [response] [EOS]`. Each interleaving `<MASK>` in the prompt section is tagged `<KEEP>`, while each prompt token is tagged `<REPLACE>` with **itself** as the generator target (self-replacement), teaching the model to reconstruct the prompt.
* **Inference**: positions of type `bos`, `eos`, and `prompt_imask` are hard-locked to `<KEEP>` during edit execution; a `<KEEP>` prediction on a `prompt_tok` position is forced to `<REPLACE>` with itself. The prompt is never deleted or modified.

---

## 1. Edit Tokenizer Extension (`dllm/tokenizer.py`)

[DLLMTokenizer](file:///home/sameer/Research/DLLM/dllm/tokenizer.py#L31-L187) wraps `transformers.RobertaTokenizer` (`roberta-base`) and registers 6 edit-specific special tokens:

| Token | ID | Description |
| :--- | :--- | :--- |
| `<KEEP>` | `50265` | Token is correct; no change required. |
| `<DELETE>` | `50266` | Token is noisy/spurious; should be removed from sequence. |
| `<REPLACE>` | `50267` | Token should be substituted with a new vocabulary token. |
| `<INSERT>` | `50268` | A new token should be inserted prior to this position. |
| `<EXPAND>` | `50269` | Token should be expanded into two `<MASK>` placeholder tokens. |
| `<MASK>` | `50270` | Placeholder token whose content must be generated. |

---

## 2. Model Backbone (`dllm/model.py`)

The backbone is `transformers.RobertaModel`. Because RoBERTa uses full bidirectional attention without a lower-triangular causal mask, every token in the sequence receives contextual information from both preceding and following tokens.

```python
self.backbone = RobertaModel.from_pretrained(backbone_name, config=config)
self.backbone.resize_token_embeddings(self.vocab_size)  # 50,271 tokens
```

---

## 3. Dual Heads

### Head 1: Tagger (`tagger_head`)
Predicts the discrete edit operation for each position in the input sequence.

* **Input**: Backbone last hidden state $H \in \mathbb{R}^{B \times S \times d}$
* **Architecture**:
  $$\text{Tagger}(H) = \text{Linear}_{d \to d} \to \text{GELU} \to \text{LayerNorm}_d \to \text{Linear}_{d \to 5}$$
* **Output**: Logits over 5 tag indices:
  * `0`: KEEP
  * `1`: DELETE
  * `2`: REPLACE
  * `3`: INSERT
  * `4`: EXPAND

### Head 2: Generator (`generator_head`)
Predicts vocabulary tokens for positions tagged as `REPLACE` or `INSERT`.

* **Tag Conditioning**: To inform the generator of the edit decision, tag embeddings are added directly to the backbone hidden states:
  $$H_{\text{cond}} = H + \text{Embedding}_{\text{tag}}(\text{tag\_indices})$$
  *(Teacher forcing uses target tag labels during training; predicted tags during inference)*
* **Architecture**:
  $$\text{Generator}(H_{\text{cond}}) = \text{Linear}_{d \to d} \to \text{GELU} \to \text{LayerNorm}_d \to \text{Linear}_{d \to V_{\text{orig}}}$$
* **Weight Tying**: The weight matrix of the final projection layer is tied with the input token embeddings of the RoBERTa backbone for the original vocabulary ($V_{\text{orig}} = 50,265$).

### Head 3: Length (`length_head`)
Predicts the response length $M$ (number of response slot pairs) from the prompt. The backbone hidden states are mean-pooled **over prompt positions only** (`prompt_mask`), and a linear layer maps the pooled vector to logits over lengths $1 \dots \text{max\_response\_length}$:

$$\text{Length}(H) = \text{Linear}_{d \to M_{\max}}\left( \frac{1}{|P|} \sum_{i \in P} H_i \right)$$

$M$ is the *initial* mask count — the canvas is built with exactly $M$ response slots and is never pruned; `INSERT`/`EXPAND` can still grow it. At inference the length is predicted in a first forward pass over a prompt-only canvas.

---

## 4. Loss Function (`compute_loss`)

Training optimizes a joint cross-entropy loss combining the Tagger, Generator, and Length outputs:

$$L_{\text{total}} = L_{\text{tag}} + L_{\text{gen}} + L_{\text{len}}$$

### Tag Loss ($L_{\text{tag}}$)
Cross-entropy over all valid positions (ignoring padding positions marked `-100`):
$$L_{\text{tag}} = \text{CrossEntropy}(\text{tag\_logits}, \text{tag\_indices})$$

The tag loss supports optional per-class weights (ordered `[KEEP, DELETE, REPLACE, INSERT, EXPAND]`) to counter the majority-class bias — e.g. `configs/default.yaml` uses `[0.7, 0.8, 1.0, 2.0, 4.0]`, upweighting the rare `INSERT`/`EXPAND`. (`DELETE` never appears in training labels in `prompt_response` mode, since the canvas holds exactly the answer.)

### Generator Loss ($L_{\text{gen}}$)
Cross-entropy computed **only** at positions flagged by `gen_mask` (positions requiring generation, i.e., `REPLACE` or `INSERT`):
$$L_{\text{gen}} = \frac{1}{N_{\text{gen}}} \sum_{i \in \text{gen\_mask}} \text{CrossEntropy}(\text{gen\_logits}_i, \text{gen\_labels}_i)$$

If $N_{\text{gen}} = 0$, $L_{\text{gen}} = 0$.

### Length Loss ($L_{\text{len}}$)
Cross-entropy between the length-head logits and the true answer length (clamped to $1 \dots M_{\max}$; samples with an invalid target — including answers truncated at `max_response_length`, which have no true length — are ignored):
$$L_{\text{len}} = \text{CrossEntropy}(\text{length\_logits}, \text{clamp}(M, 1, M_{\max}) - 1)$$

Two mechanisms counter the answer-length class imbalance (short answers are rare; truncated answers create a fake mass at $M_{\max}$):
1. **Inverse-frequency class weights** (from `DLLMDataset.length_weights()`, smoothed with a square root).
2. **Neighbor smoothing** (`len_smoothing`, default 0.15): a fraction of the target mass is spread to the adjacent length classes, since length is ordinal — a near miss should be penalized less than a far miss.
