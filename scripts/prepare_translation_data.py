#!/usr/bin/env python3
"""
Prepare English <-> Hindi parallel data for DLLM training.

Default dataset: ai4bharat/samanantar (en-hi, ~10M crawled parallel sentences —
much larger and more diverse than OPUS-100). Rows are quality-filtered and
deduplicated, and the training set is capped at --max_pairs. Since the HF
Samanantar mirror has no val/test splits, they are carved from the filtered
train stream (deterministic: first rows of the stream).

Also supports the old opus100 source (which has real train/validation/test).

Writes (same names, so the translation configs are unchanged):
    data/en_hi_both_train.tsv   en->hi AND hi->en pairs (training)
    data/en_hi_both_val.tsv     validation (both directions)
    data/en_hi_test.tsv         en->hi only (for BLEU evaluation)
    data/en_hi_rev_test.tsv     hi->en only (for BLEU evaluation)

Usage:
    python scripts/prepare_translation_data.py                          # Samanantar, 2M pairs
    python scripts/prepare_translation_data.py --max_pairs 3000000      # bigger cap
    python scripts/prepare_translation_data.py --dataset opus100 --max_pairs 500000
"""

import argparse
import hashlib
import os
import unicodedata
import urllib.request

from datasets import load_dataset

MAX_LEN_EN = 120
MAX_LEN_HI = 180

# MUSE/ARRIVAL ground-truth bilingual dictionary: word-level en-hi pairs.
WORD_DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.txt"


def is_word(s: str) -> bool:
    """
    True for single tokens made of letters and combining marks only
    (Devanagari vowel signs are marks, so plain isalpha() wrongly rejects
    most Hindi words). Rejects digits, punctuation, and symbols.
    """
    for c in s:
        cat = unicodedata.category(c)
        if c.isdigit() or c == "_" or cat[0] in "PS":
            return False
        if not (c.isalpha() or cat in ("Mn", "Mc", "Me")):
            return False
    return True


def quality_filter(src: str, tgt: str) -> bool:
    """Light quality gate: reasonable lengths, sane length ratio, not identical."""
    if not src or not tgt:
        return False
    if src == tgt:
        return False
    n_en = len(src.split())
    n_hi = len(tgt.split())
    if n_en < 1 or n_en > MAX_LEN_EN or n_hi < 1 or n_hi > MAX_LEN_HI:
        return False
    ratio = n_en / max(1, n_hi)
    if ratio < 0.25 or ratio > 2.5:
        return False
    return True


def samanantar_iter(split: str):
    ds = load_dataset("ai4bharat/samanantar", "hi", split=split, streaming=True)
    for item in ds:
        yield item["src"].strip(), item["tgt"].strip()


def opus100_iter(split: str):
    ds = load_dataset("opus100", "en-hi", split=split, streaming=True)
    for item in ds:
        tr = item["translation"]
        yield tr["en"].strip(), tr["hi"].strip()


def write_tsv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for src, tgt in rows:
            f.write(f"{src}\t{tgt}\n")
    print(f"wrote {len(rows)} pairs -> {path}")


def append_word_dict(train_path: str, out_dir: str):
    """
    Append MUSE/ARRIVAL word-level en<->hi pairs (one-to-one per word) to the
    training file. Pairs with multi-token or non-alphabetic sides are dropped
    so every row is a genuine single-word translation (both directions).
    """
    dict_path = os.path.join(out_dir, "en_hi_dict.txt")
    if not os.path.isfile(dict_path):
        print(f"Downloading word dictionary -> {dict_path}")
        urllib.request.urlretrieve(WORD_DICT_URL, dict_path)

    count = 0
    skipped = 0
    with open(train_path, "a", encoding="utf-8") as f:
        for line in open(dict_path, encoding="utf-8"):
            parts = line.split()
            if len(parts) != 2:
                skipped += 1
                continue
            en, hi = parts
            if en == hi or not is_word(en) or not is_word(hi):
                skipped += 1
                continue
            f.write(f"{en}\t{hi}\n")
            f.write(f"{hi}\t{en}\n")
            count += 1
    print(f"appended {count} word pairs (both directions) to train; skipped {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="ai4bharat/samanantar",
                        choices=["ai4bharat/samanantar", "opus100"])
    parser.add_argument("--max_pairs", type=int, default=2000000,
                        help="Max filtered source pairs for training (None = no cap)")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable exact-source deduplication")
    parser.add_argument("--no_words", action="store_true",
                        help="Do not append the MUSE word-level dictionary to train")
    args = parser.parse_args()

    it = samanantar_iter if args.dataset == "ai4bharat/samanantar" else opus100_iter
    print(f"Dataset: {args.dataset} (filtered, dedup={'off' if args.no_dedup else 'on'}, "
          f"cap={args.max_pairs})")

    heldout = []          # first 8000 accepted rows -> val + test sets
    seen = set()
    written = 0
    skipped = 0
    train_path = os.path.join(args.out_dir, "en_hi_both_train.tsv")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f_train:
        for src, tgt in it("train"):
            if not quality_filter(src, tgt):
                skipped += 1
                continue
            if not args.no_dedup:
                h = int.from_bytes(hashlib.md5(src.encode("utf-8")).digest()[:8], "big")
                if h in seen:
                    skipped += 1
                    continue
                seen.add(h)
            if len(heldout) < 8000:
                heldout.append((src, tgt))
                continue
            if args.max_pairs is not None and written >= args.max_pairs:
                break
            f_train.write(f"{src}\t{tgt}\n")
            f_train.write(f"{tgt}\t{src}\n")
            written += 1
            if written % 200000 == 0:
                print(f"  {written} train pairs...", flush=True)

    print(f"train pairs: {written} (both directions); filtered out: {skipped}")

    # Carve val/test from the held-out rows (disjoint sentences per set)
    val = []
    for src, tgt in heldout[4000:8000]:
        val.append((src, tgt))
        val.append((tgt, src))
    write_tsv(os.path.join(args.out_dir, "en_hi_both_val.tsv"), val)
    write_tsv(os.path.join(args.out_dir, "en_hi_test.tsv"), heldout[0:2000])
    write_tsv(os.path.join(args.out_dir, "en_hi_rev_test.tsv"),
              [(t, s) for s, t in heldout[2000:4000]])

    if not args.no_words:
        append_word_dict(train_path, args.out_dir)

    print(f"\nDone. Train pairs (both directions): {written * 2}" + ("" if args.no_words else " + word pairs"))


if __name__ == "__main__":
    main()
