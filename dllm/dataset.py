"""
DLLM Dataset: Loads text data and applies forward corruption on the fly.

Supports:
- Local text files (e.g., tiny_shakespeare.txt)
- HuggingFace datasets
- Automatic download of tiny_shakespeare raw text

Each __getitem__ call:
1. Fetches a clean text sequence
2. Tokenizes it
3. Applies the ForwardCorruptor to produce (noisy, tag_labels)
4. Returns tensors ready for training
"""

from __future__ import annotations

import os
import collections
import requests
from typing import Dict, Optional, List, Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


# URL for the raw tiny_shakespeare dataset
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


class DLLMDataset(Dataset):
    """
    PyTorch Dataset for training the DLLM.

    Loads text data (e.g., tiny_shakespeare) and applies corruption
    dynamically so the model sees different corruptions each epoch.
    """

    def __init__(
        self,
        tokenizer,          # DLLMTokenizer
        corruptor,          # ForwardCorruptor
        dataset_name: str = "tiny_shakespeare",
        split: str = "train",
        max_length: int = 128,
        mode: str = "prompt_response",
        max_prompt_length: int = 48,
        max_response_length: int = 48,
        seed: Optional[int] = None,
        text_file: Optional[str] = None,
        cache_dir: str = "./data",
        sub_iterations: Optional[List[float]] = None,
    ):
        """
        Args:
            tokenizer: DLLMTokenizer instance.
            corruptor: ForwardCorruptor instance.
            dataset_name: HuggingFace dataset name or path to local text file.
            split: Dataset split ('train' or 'validation' or 'test').
            max_length: Maximum total sequence length for tokenization.
            mode: Dataset mode ('prompt_response' or 'unconditional').
            max_prompt_length: Max tokens for prompt context in prompt_response mode.
            max_response_length: Max tokens for target response in prompt_response mode.
            seed: Random seed for shuffling.
            text_file: Explicit path to a local text file.
            cache_dir: Directory to cache downloaded datasets.
            sub_iterations: Optional list of noise levels for trajectory training
                (e.g. [1.0, 0.8, 0.6, 0.4, 0.2]). When set, __getitem__ returns
                tensors of shape (K, max_length) instead of (max_length,). Use
                only for the training dataset; set to None for validation.
        """
        self.tokenizer = tokenizer
        self.corruptor = corruptor
        self.max_length = max_length
        self.mode = mode
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.sub_iterations = sub_iterations  # None → single-stage; List → trajectory mode
        self._len_hist = None

        # Load text data
        self._text_lines: List[str] = []

        if text_file is not None and os.path.isfile(text_file):
            self._text_lines = self._load_text_file(text_file)
        elif dataset_name == "tiny_shakespeare":
            self._text_lines = self._load_tiny_shakespeare(cache_dir)
        elif os.path.isfile(dataset_name):
            self._text_lines = self._load_text_file(dataset_name)
        else:
            # An explicit local path that does not exist is a hard error —
            # never silently fall back to tiny_shakespeare.
            if os.path.sep in dataset_name or dataset_name.endswith((".tsv", ".txt", ".jsonl")):
                raise FileNotFoundError(
                    f"dataset_name '{dataset_name}' is not a readable file. "
                    "Fix the path (check symlinks/dataset versions on Kaggle); "
                    "refusing to fall back to tiny_shakespeare."
                )
            # Try loading as HuggingFace dataset
            try:
                self._hf_dataset = load_dataset(dataset_name, split=split, streaming=True)
                self._is_hf = True
                return
            except Exception:
                # Fallback: try tiny_shakespeare download
                self._text_lines = self._load_tiny_shakespeare(cache_dir)

        self._is_hf = False

    # ── Data Loading Methods ──────────────────────────────────────────

    def _load_text_file(self, path: str) -> List:
        """Load text from a local file and parse chunks or dialogue pairs."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if self.mode == "prompt_response":
            # Tab-separated parallel corpus (e.g. translation data):
            # each line is "source<TAB>target", loaded directly as prompt-response pairs.
            first_line = text.split("\n", 1)[0]
            if "\t" in first_line:
                pairs = []
                for line in text.splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                        pairs.append((parts[0].strip(), parts[1].strip()))
                return pairs
            return self._extract_dialogue_pairs(text)
        return self._chunk_text(text)

    def _load_tiny_shakespeare(self, cache_dir: str) -> List:
        """Download and load tiny_shakespeare, extracting dialogue pairs if in prompt_response mode."""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "tiny_shakespeare.txt")

        if not os.path.isfile(cache_path):
            print(f"Downloading tiny_shakespeare to {cache_path}...")
            response = requests.get(TINY_SHAKESPEARE_URL)
            response.raise_for_status()
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("Download complete.")

        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()

        if self.mode == "prompt_response":
            return self._extract_dialogue_pairs(text)
        return self._chunk_text(text)

    def _chunk_text(self, text: str, min_tokens: int = 32) -> List[str]:
        """
        Split long text into chunks suitable for tokenization.
        Merges lines until each chunk has at least min_tokens.
        """
        # Split on newlines
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Merge lines into chunks of at least min_tokens
        chunks = []
        buffer = ""
        for line in lines:
            candidate = f"{buffer} {line}".strip() if buffer else line
            tokens = self.tokenizer.encode(candidate)
            if len(tokens) >= min_tokens:
                chunks.append(candidate)
                buffer = ""
            else:
                buffer = candidate

        # Flush remaining buffer into last chunk or as standalone
        if buffer:
            if chunks:
                chunks[-1] = f"{chunks[-1]} {buffer}".strip()
            else:
                chunks.append(buffer)

        # Filter out chunks that are still too short after encoding
        chunks = [c for c in chunks if len(self.tokenizer.encode(c)) >= 8]

        return chunks

    def _extract_dialogue_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract consecutive dialogue turn pairs (Turn K -> Turn K+1) from text.

        For example in Shakespeare / dialogue scripts:
            Turn 1 (Prompt):   "First Citizen: Before we proceed any further, hear me speak."
            Turn 2 (Response): "All: Speak, speak."
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        turns = []
        current_speaker = ""
        current_speech = []

        for line in lines:
            # Check if line indicates a speaker heading (e.g., "First Citizen:", "KING EDWARD:")
            if (line.endswith(":") and len(line.split()) <= 4 and line[:-1].isupper()) or \
               (line.endswith(":") and len(line) < 30 and any(c.isupper() for c in line)):
                if current_speaker and current_speech:
                    speech_text = " ".join(current_speech)
                    turns.append(f"{current_speaker} {speech_text}".strip())
                    current_speech = []
                current_speaker = line
            else:
                current_speech.append(line)

        # Flush final turn
        if current_speaker and current_speech:
            speech_text = " ".join(current_speech)
            turns.append(f"{current_speaker} {speech_text}".strip())

        # If no speaker turns identified, fallback to sentence splitting
        if len(turns) < 2:
            chunks = self._chunk_text(text, min_tokens=16)
            pairs = []
            for i in range(len(chunks) - 1):
                pairs.append((chunks[i], chunks[i + 1]))
            return pairs

        # Build consecutive (Turn_K, Turn_K+1) prompt-response pairs
        pairs = []
        for i in range(len(turns) - 1):
            prompt_turn = turns[i]
            response_turn = turns[i + 1]
            pairs.append((prompt_turn, response_turn))

        return pairs

    # ── Dataset API ───────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of text chunks or dialogue pairs available."""
        if self._is_hf:
            try:
                return len(self._hf_dataset)
            except TypeError:
                return 100000
        return len(self._text_lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch a single training sample."""
        if self.mode == "prompt_response" and not self._is_hf:
            idx = idx % len(self._text_lines)
            item = self._text_lines[idx]
            if isinstance(item, tuple):
                prompt_text, response_text = item
            else:
                prompt_text = str(item)
                response_text = ""

            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)[:self.max_prompt_length]
            raw_response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            resp_truncated = len(raw_response_ids) > self.max_response_length
            clean_response_ids = raw_response_ids[:self.max_response_length]

            if len(prompt_ids) == 0 or len(clean_response_ids) == 0:
                return self._dummy_item()

            # Target response length for the length head (M = number of slots);
            # -100 when the answer was truncated at max_response_length.
            resp_length = -100 if resp_truncated else len(clean_response_ids)

            # ── Multi-stage trajectory mode ──────────────────────────────────
            if self.sub_iterations is not None:
                stage_tuples = self.corruptor.corrupt_prompt_response_trajectory(
                    prompt_ids=prompt_ids,
                    clean_response_ids=clean_response_ids,
                    stages=self.sub_iterations,
                )

                all_noisy    = []
                all_tags     = []
                all_gen_lbl  = []
                all_gen_mask = []
                all_pmask    = []

                for noisy_ids_k, tag_labels_k, prompt_mask_k, pos_to_clean_k in stage_tuples:
                    gen_labels_k, gen_mask_k = self._build_gen_labels_from_map(
                        noisy_ids_k, tag_labels_k, pos_to_clean_k
                    )
                    all_noisy.append(self._pad_or_truncate(noisy_ids_k,    self.max_length, self.tokenizer.pad_id))
                    all_tags.append( self._pad_or_truncate(tag_labels_k,   self.max_length, -100))
                    all_gen_lbl.append(self._pad_or_truncate(gen_labels_k, self.max_length, -100))
                    all_gen_mask.append(self._pad_or_truncate_bool(gen_mask_k,   self.max_length, False))
                    all_pmask.append(  self._pad_or_truncate_bool(prompt_mask_k, self.max_length, False))

                # attention_mask is identical across stages (only masks padding, not content);
                # compute once from stage-0 noisy_ids for efficiency.
                attn_mask = [1 if tid != self.tokenizer.pad_id else 0 for tid in all_noisy[0]]

                # Stack stages into (K, max_length) tensors.
                return {
                    "noisy_ids":      torch.tensor(all_noisy,    dtype=torch.long),   # (K, S)
                    "attention_mask": torch.tensor(attn_mask,    dtype=torch.long),   # (S,)
                    "tag_labels":     torch.tensor(all_tags,     dtype=torch.long),   # (K, S)
                    "gen_labels":     torch.tensor(all_gen_lbl,  dtype=torch.long),   # (K, S)
                    "gen_mask":       torch.tensor(all_gen_mask, dtype=torch.bool),   # (K, S)
                    "prompt_mask":    torch.tensor(all_pmask,    dtype=torch.bool),   # (K, S)
                    "resp_length":    torch.tensor(resp_length,  dtype=torch.long),   # scalar
                }

            # ── Single-stage mode (default) ──────────────────────────────────
            # Corrupt response with interleaved canvas while keeping prompt pristine.
            noisy_ids, tag_labels, prompt_mask, pos_to_clean = self.corruptor.corrupt_prompt_response(
                prompt_ids=prompt_ids,
                clean_response_ids=clean_response_ids,
            )

            # Build generator labels using exact position map (no heuristic)
            gen_labels, gen_mask = self._build_gen_labels_from_map(noisy_ids, tag_labels, pos_to_clean)
        else:
            # Unconditional mode or HF dataset
            if self._is_hf:
                text = self._get_hf_item(idx)
            else:
                idx = idx % len(self._text_lines)
                text = str(self._text_lines[idx])

            # Tokenize
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            clean_ids = encoded["input_ids"][0].tolist()
            attention_mask = encoded["attention_mask"][0]

            # Remove padding for corruption
            real_length = attention_mask.sum().item()
            clean_ids = clean_ids[:real_length]

            # Skip short sequences
            if len(clean_ids) < 8:
                return self._dummy_item()

            # Apply forward corruption
            noisy_ids, tag_labels = self.corruptor.corrupt(clean_ids)

            # Build generator labels
            gen_labels, gen_mask = self._build_gen_labels(noisy_ids, tag_labels, clean_ids)

            # Unconditional mode: no length head target, no prompt mask
            resp_length = -100
            prompt_mask = [False] * len(noisy_ids)

        # Pad/truncate
        noisy_ids = self._pad_or_truncate(noisy_ids, self.max_length, self.tokenizer.pad_id)
        tag_labels = self._pad_or_truncate(tag_labels, self.max_length, -100)
        gen_labels = self._pad_or_truncate(gen_labels, self.max_length, -100)
        gen_mask = self._pad_or_truncate_bool(gen_mask, self.max_length, False)
        prompt_mask = self._pad_or_truncate_bool(prompt_mask, self.max_length, False)

        # Attention mask based on noisy_ids
        attn_mask = [1 if tid != self.tokenizer.pad_id else 0 for tid in noisy_ids]

        return {
            "noisy_ids": torch.tensor(noisy_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "tag_labels": torch.tensor(tag_labels, dtype=torch.long),
            "gen_labels": torch.tensor(gen_labels, dtype=torch.long),
            "gen_mask": torch.tensor(gen_mask, dtype=torch.bool),
            "prompt_mask": torch.tensor(prompt_mask, dtype=torch.bool),
            "resp_length": torch.tensor(resp_length, dtype=torch.long),
        }

    def _get_hf_item(self, idx: int) -> str:
        """Get text from HuggingFace dataset."""
        if hasattr(self._hf_dataset, "__getitem__"):
            item = self._hf_dataset[idx]
        else:
            item = None
            for i, row in enumerate(self._hf_dataset):
                if i == idx:
                    item = row
                    break
            if item is None:
                return ""
        return item.get("text", item.get("content", ""))

    # ── Length Statistics (for rebalancing the length head) ──────────

    def response_length_hist(self) -> Dict[int, int]:
        """
        Histogram of response lengths for non-truncated samples, used to
        rebalance the length-head classes (short answers are rare).
        """
        if self._len_hist is None:
            hist = collections.Counter()
            for item in self._text_lines:
                if isinstance(item, tuple):
                    _, r = item
                else:
                    r = str(item)
                rid = self.tokenizer.encode(r, add_special_tokens=False)
                if 0 < len(rid) <= self.max_response_length:
                    hist[len(rid)] += 1
            self._len_hist = dict(hist)
        return self._len_hist

    def length_weights(self, exponent: float = 0.5) -> List[float]:
        """
        Inverse-frequency weights over length classes 1..max_response_length
        (smoothed with sqrt to avoid extreme weights).
        """
        hist = self.response_length_hist()
        return [(hist.get(l, 0) + 1) ** (-exponent)
                for l in range(1, self.max_response_length + 1)]

    def _dummy_item(self) -> Dict[str, torch.Tensor]:
        """Return a zero-filled dummy (will be filtered by collate_fn)."""
        return {
            "noisy_ids": torch.zeros(self.max_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "tag_labels": torch.full((self.max_length,), -100, dtype=torch.long),
            "gen_labels": torch.full((self.max_length,), -100, dtype=torch.long),
            "gen_mask": torch.zeros(self.max_length, dtype=torch.bool),
            "prompt_mask": torch.zeros(self.max_length, dtype=torch.bool),
            "resp_length": torch.tensor(-100, dtype=torch.long),
        }

    # ── Generator Labels ──────────────────────────────────────────────

    def _build_gen_labels_from_map(
        self,
        noisy_ids: list,
        tag_labels: list,
        pos_to_clean: dict,  # {canvas_position: clean_token_id}
    ) -> tuple:
        """
        Build generator labels and gen_mask using the exact position-to-clean-token
        map returned by the corruptor. No heuristic alignment needed.
        """
        gen_labels = [-100] * len(noisy_ids)
        gen_mask = [False] * len(noisy_ids)

        replace_id = self.tokenizer.replace_id
        insert_id = self.tokenizer.insert_id

        for i, tag in enumerate(tag_labels):
            if tag in (replace_id, insert_id) and i in pos_to_clean:
                gen_labels[i] = pos_to_clean[i]
                gen_mask[i] = True

        return gen_labels, gen_mask

    def _build_gen_labels(
        self,
        noisy_ids: list,
        tag_labels: list,
        clean_ids: list,
    ) -> tuple:
        """Fallback: build generator labels for unconditional mode."""
        gen_labels = [-100] * len(noisy_ids)
        gen_mask = [False] * len(noisy_ids)

        replace_id = self.tokenizer.replace_id
        insert_id = self.tokenizer.insert_id

        for i, (nid, tag) in enumerate(zip(noisy_ids, tag_labels)):
            if tag in (replace_id, insert_id):
                gen_labels[i] = self._find_aligned_clean(i, noisy_ids, tag_labels, clean_ids)
                gen_mask[i] = True

        return gen_labels, gen_mask

    def _find_aligned_clean(
        self,
        noisy_pos: int,
        noisy_ids: list,
        tag_labels: list,
        clean_ids: list,
    ) -> int:
        """Heuristic: find clean token aligned with noisy position."""
        clean_idx = 0
        for i in range(noisy_pos):
            if i < len(tag_labels) and tag_labels[i] != self.tokenizer.delete_id:
                clean_idx += 1

        if clean_idx < len(clean_ids):
            return clean_ids[clean_idx]
        return 4  # fallback to common token

    # ── Padding Utilities ──────────────────────────────────────────────

    def _pad_or_truncate(self, seq: list, target_len: int, pad_value: int) -> list:
        if len(seq) > target_len:
            return seq[:target_len]
        return seq + [pad_value] * (target_len - len(seq))

    def _pad_or_truncate_bool(self, seq: list, target_len: int, pad_value: bool) -> list:
        if len(seq) > target_len:
            return seq[:target_len]
        return seq + [pad_value] * (target_len - len(seq))


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Custom collate function; filters out dummy samples and stacks tensors.

    Handles both single-stage batches (tensors of shape (S,)) and multi-stage
    trajectory batches (tensors of shape (K, S)). In multi-stage mode:
      • noisy_ids, tag_labels, gen_labels, gen_mask, prompt_mask → (B, K, S)
      • attention_mask, resp_length → (B, S) / (B,)  [shared across K stages]
    """
    # Filter out dummy/empty samples (attention_mask always has shape (..., S))
    batch = [b for b in batch if b["attention_mask"].sum() > 0]
    if len(batch) == 0:
        return {}

    # Detect multi-stage mode by checking noisy_ids dimensionality
    is_multistage = batch[0]["noisy_ids"].dim() == 2  # (K, S) per sample

    if is_multistage:
        return {
            # (B, K, S) — stage dimension inserted between batch and sequence
            "noisy_ids":      torch.stack([b["noisy_ids"]   for b in batch]),
            "tag_labels":     torch.stack([b["tag_labels"]  for b in batch]),
            "gen_labels":     torch.stack([b["gen_labels"]  for b in batch]),
            "gen_mask":       torch.stack([b["gen_mask"]    for b in batch]),
            "prompt_mask":    torch.stack([b["prompt_mask"] for b in batch]),
            # (B, S) — attention_mask is identical across all K stages
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            # (B,) — resp_length is a shared scalar per sample
            "resp_length":    torch.stack([b["resp_length"] for b in batch]),
        }

    # Single-stage: original behavior
    return {
        "noisy_ids":      torch.stack([b["noisy_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "tag_labels":     torch.stack([b["tag_labels"]     for b in batch]),
        "gen_labels":     torch.stack([b["gen_labels"]     for b in batch]),
        "gen_mask":       torch.stack([b["gen_mask"]       for b in batch]),
        "prompt_mask":    torch.stack([b["prompt_mask"]    for b in batch]),
        "resp_length":    torch.stack([b["resp_length"]    for b in batch]),
    }
