#!/usr/bin/env python3
"""
Prepare English <-> Hindi parallel data for DLLM training from the
IIT Bombay English-Hindi Parallel Corpus (cfilt/iitb-english-hindi) plus the
MUSE/ARRIVAL ground-truth word-level bilingual dictionary.

Why IITB instead of Samanantar?
  * IITB is a curated corpus (built from existing sources + IIT Bombay data),
    so it needs far less aggressive cleaning than crawled Samanantar text.
  * It ships with real train/validation/test splits on HuggingFace
    (1.66M / 520 / 2.51k rows), so we no longer carve held-out rows from the
    stream head.
  * It still produces both en->hi and hi->en pairs for a single-direction loss.

The MUSE/ARRIVAL word dictionary is appended to the training set to give the
model clean single-word anchors and help the short-answer / length-head regime
(short answers are otherwise rare).

Writes (same names, so the translation configs are unchanged):
    data/en_hi_both_train.tsv   en->hi AND hi->en pairs (training)
    data/en_hi_both_val.tsv     validation (both directions)
    data/en_hi_test.tsv         en->hi only (for BLEU evaluation)
    data/en_hi_rev_test.tsv     hi->en only (for BLEU evaluation)

Usage:
    python scripts/prepare_translation_data.py                       # full IITB train (both directions)
    python scripts/prepare_translation_data.py --max_pairs 1000000   # cap the training set
    python scripts/prepare_translation_data.py --no_words            # skip the MUSE word dict
"""

import argparse
import hashlib
import html
import os
import re
import unicodedata
import urllib.request

from datasets import load_dataset

MAX_LEN_EN = 120
MAX_LEN_HI = 180

# MUSE/ARRIVAL ground-truth bilingual dictionary: word-level en-hi pairs.
WORD_DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.txt"

# Devanagari Unicode range (used for the light Hindi language check).
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# Patterns for cleaning up IITB's software-localization artifacts (see below).
#   (1) A parenthesized group containing an underscore is a mnemonic accelerator
#       hint like "Acti _ on ( _ M)" or "(C _)" -> drop the whole group.
#   (2) A mid-word underscore surrounded by word chars is an accelerator:
#       "Co _ mponent" -> "Component".
_ACCEL_PAREN = re.compile(r"\([^()]*_[^()]*\)")
_ACCEL_MID = re.compile(r"(\w)\s*_\s*(?=\w)")
_ACCEL_LEAD = re.compile(r"(^|\s)_\s*(?=\w)")
_TAG = re.compile(r"<[^>]*>")
# Mangled / double-escaped entity tokens: matches an entity name (lt, gt, amp,
# quot, apos, nbsp, bgt) with optional leading & / < / / and trailing / ; >,
# whether space-separated ("& lt;") or contiguous ("&lt;").
_MANGLED_ENTITY = re.compile(
    r"[&</]?\s*(?:lt|gt|amp|quot|apos|nbsp|bgt)\s*/?\s*;?\s*[>]?",
    re.IGNORECASE,
)


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


def normalize(s: str) -> str:
    """Normalize a line: NFKC unicode + strip localization markup + collapse whitespace.

    The IITB corpus is dominated by GNOME/dogtail UI localization strings, which
    inject presentational artifacts that are not real language content and would
    otherwise teach the model to emit stray underscores/brackets at inference:

      1. Mnemonic accelerators (underscore marks the Alt-key):
           "Co _ mponent" -> "Component"
           "_ File"       -> "File"
           "Acti _ on ( _ M)" -> "Action"
      2. Mangled HTML markup from the source's double-escaping:
           "& lt; bgt; Table lt;/bgt;" -> "Table"
    """
    s = unicodedata.normalize("NFKC", s)

    # (2) Markup: unescape real entities, strip remaining tags, then remove the
    # leftover 'lt;'/'gt;'/'bgt;' fragments that remain after the corpus was
    # double-escaped.
    s = html.unescape(s)
    s = _TAG.sub(" ", s)
    s = _MANGLED_ENTITY.sub(" ", s)
    # Stray delimiters left behind after entity/tag removal (&, <, >).
    s = s.replace("&", " ").replace("<", " ").replace(">", " ")

    # (1) Accelerators, in order: whole parenthesized hints, mid-word joins,
    # leading hints, then any residual underscores.
    s = _ACCEL_PAREN.sub(" ", s)
    s = _ACCEL_MID.sub(lambda m: m.group(1), s)
    s = _ACCEL_LEAD.sub(lambda m: m.group(1), s)
    s = s.replace("_", " ")

    return " ".join(s.split())


def hindi_fraction(s: str) -> float:
    """Fraction of characters in `s` that fall in the Devanagari range."""
    if not s:
        return 0.0
    dev = sum(1 for c in s if DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)
    return dev / len(s)


def quality_filter(src: str, tgt: str) -> bool:
    """Quality gate: reasonable lengths, sane length ratio, not identical."""
    if not src or not tgt:
        return False
    if src == tgt:
        return False
    # Reject URL/template noise that survives normalization. Angle-bracket /
    # entity markup is already stripped in normalize(), so it need not be re-tested.
    for bad in ("http", "www.", "mailto:", "{{", "}}"):
        if bad in src or bad in tgt:
            return False
    n_en = len(src.split())
    n_hi = len(tgt.split())
    if n_en < 1 or n_en > MAX_LEN_EN or n_hi < 1 or n_hi > MAX_LEN_HI:
        return False
    ratio = n_en / max(1, n_hi)
    if ratio < 0.25 or ratio > 2.5:
        return False
    return True


def iitb_iter(split: str):
    """
    Yield normalized (en, hi) pairs from the IITB corpus split.
    Rows without an 'en' or 'hi' key are skipped.
    """
    ds = load_dataset("cfilt/iitb-english-hindi", split=split, streaming=True)
    for item in ds:
        trans = item.get("translation") or item
        en = trans.get("en", "").strip()
        hi = trans.get("hi", "").strip()
        en = normalize(en)
        hi = normalize(hi)
        if en and hi:
            yield en, hi


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
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max filtered source pairs for training (None = no cap)")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable exact-source deduplication")
    parser.add_argument("--no_words", action="store_true",
                        help="Do not append the MUSE word-level dictionary to train")
    parser.add_argument("--require_hindi", action="store_true",
                        help="Drop rows whose target is not predominantly Devanagari")
    args = parser.parse_args()

    print(f"Dataset: cfilt/iitb-english-hindi (IITB en-hi) "
          f"+ MUSE word dict (words={'off' if args.no_words else 'on'}, "
          f"dedup={'off' if args.no_dedup else 'on'}, cap={args.max_pairs})")

    # IITB has real splits, so we read each split directly instead of carving
    # held-out rows from the training stream.
    train_path = os.path.join(args.out_dir, "en_hi_both_train.tsv")
    os.makedirs(args.out_dir, exist_ok=True)

    seen = set()
    train_written = 0
    train_skipped = 0
    with open(train_path, "w", encoding="utf-8") as f_train:
        for src, tgt in iitb_iter("train"):
            if not quality_filter(src, tgt):
                train_skipped += 1
                continue
            if args.require_hindi and hindi_fraction(tgt) < 0.6:
                train_skipped += 1
                continue
            if not args.no_dedup:
                h = int.from_bytes(hashlib.md5(src.encode("utf-8")).digest()[:8], "big")
                if h in seen:
                    train_skipped += 1
                    continue
                seen.add(h)
            if args.max_pairs is not None and train_written >= args.max_pairs:
                break
            f_train.write(f"{src}\t{tgt}\n")
            f_train.write(f"{tgt}\t{src}\n")
            train_written += 1
            if train_written % 200000 == 0:
                print(f"  {train_written} train pairs...", flush=True)

    print(f"train pairs: {train_written} (both directions); filtered out: {train_skipped}")

    # Validation and test sets: IITB's own held-out splits. Write both
    # directions for validation (mirrors the training file format), but keep
    # the test files single-direction (en->hi and hi->en) for BLEU.
    val = []
    for src, tgt in iitb_iter("validation"):
        if not quality_filter(src, tgt):
            continue
        val.append((src, tgt))
        val.append((tgt, src))
    write_tsv(os.path.join(args.out_dir, "en_hi_both_val.tsv"), val)

    test_forward = []
    for src, tgt in iitb_iter("test"):
        if not quality_filter(src, tgt):
            continue
        test_forward.append((src, tgt))
    write_tsv(os.path.join(args.out_dir, "en_hi_test.tsv"), test_forward)
    write_tsv(os.path.join(args.out_dir, "en_hi_rev_test.tsv"),
              [(t, s) for s, t in test_forward])

    if not args.no_words:
        append_word_dict(train_path, args.out_dir)

    print(f"\nDone. Train pairs (both directions): {train_written * 2}"
          + ("" if args.no_words else " + word pairs"))


if __name__ == "__main__":
    main()
