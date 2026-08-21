#!/usr/bin/env python3
"""
Prepare monolingual Wikipedia pretraining data for DLLM.

Downloads (streams) the cleaned HuggingFace `wikimedia/wikipedia` dataset for a
single language, applies light cleaning, optionally gates non-Devanagari text
out of the Hindi split, dedups, and writes one cleaned text chunk per line to a
plain-text file. The DLLM `unconditional` mode then reads this file directly.

Why Wikipedia for pretraining?
  * It is already preprocessed by HuggingFace (wiki markup, internal links,
    templates, and infoboxes are stripped), so very little cleaning is needed.
  * It is canonical, grammatical, and permissively licensed (CC BY-SA 4.0).
  * The DLLM objective reproduces clean text, so "clean in = grammatical out".

Cleaning applied here (on top of the HF preprocessing):
  1. NFKC unicode normalization + whitespace collapse.
  2. Drop section-header / boilerplate lines (References, See also, etc.).
  3. Drop very short lines / stubs.
  4. Drop lines with URL / template / code-fence residue.
  5. Hindi only: drop lines that are not predominantly Devanagari (this is the
     same script-purity gate as prepare_translation_data.py, reused to keep
     English / Hinglish / Romanized text out of Hindi grammar pretraining).
  6. Exact-line dedup within the run.

Usage:
    # English
    python scripts/prepare_pretrain_data.py --lang en --max_docs 200000 \
        --out data/pretrain_en.txt

    # Hindi
    python scripts/prepare_pretrain_data.py --lang hi --max_docs 200000 \
        --out data/pretrain_hi.txt

Notes:
  * The dataset is streamed, so only `--max_docs` documents are materialized.
  * The exact `--config` string (e.g. "20231101.en") should be verified against
    the live `wikimedia/wikipedia` card; it is exposed as a CLI flag for that
    reason and may need updating as HF adds snapshot dates.
"""

import argparse
import re
import unicodedata

from datasets import load_dataset


# ── Language configs ──────────────────────────────────────────────────────

LANG_DEFAULTS = {
    "en": {
        "config": "20231101.en",
        "min_line_chars": 40,
        "require_devanagari": False,
    },
    "hi": {
        "config": "20231101.hi",
        "min_line_chars": 30,
        "require_devanagari": True,
        "devanagari_frac": 0.7,
    },
}

# Section headings that survive HF extraction as standalone lines. We drop
# these so the model never learns to reproduce them as "content".
_BOILERPLATE = re.compile(
    r"^(references|see also|external links|further reading|notes|categories|"
    r"bibliography|sources|works cited|footnotes|index)\s*$",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://|www\.|\.com\b|\.org\b|\.net\b", re.IGNORECASE)
_TEMPLATE = re.compile(r"\{\{|\}\}|\[\[|\]\]|<ref|</ref>|```")

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F


def normalize_ws(s: str) -> str:
    """NFKC normalize + collapse all whitespace runs."""
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.split())


def devanagari_fraction(s: str) -> float:
    if not s:
        return 0.0
    dev = sum(1 for c in s if DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)
    return dev / len(s)


def clean_lines(text: str, lang: str) -> list:
    """Split article text into cleaned, content-bearing lines."""
    cfg = LANG_DEFAULTS[lang]
    out = []
    for raw in text.split("\n"):
        line = normalize_ws(raw)
        if len(line) < cfg["min_line_chars"]:
            continue
        if _BOILERPLATE.match(line):
            continue
        if _URL.search(line) or _TEMPLATE.search(line):
            continue
        if cfg.get("require_devanagari") and devanagari_fraction(line) < cfg.get("devanagari_frac", 0.7):
            continue
        out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia pretraining data")
    parser.add_argument("--lang", type=str, required=True, choices=["en", "hi"])
    parser.add_argument("--config", type=str, default=None,
                        help="Override the wikimedia/wikipedia snapshot config "
                             "(default: per-language default)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_docs", type=int, default=200000,
                        help="Max articles to stream (None = no cap)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output plain-text file")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Disable exact-line dedup")
    args = parser.parse_args()

    config = args.config or LANG_DEFAULTS[args.lang]["config"]
    print(f"Streaming wikimedia/wikipedia (config={config}, split={args.split}, "
          f"max_docs={args.max_docs})...")

    ds = load_dataset("wikimedia/wikipedia", config, split=args.split, streaming=True)

    seen = set()
    written = 0
    skipped_docs = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            if args.max_docs is not None and i >= args.max_docs:
                break

            text = item.get("text", "") or ""
            if not text.strip():
                skipped_docs += 1
                continue

            for line in clean_lines(text, args.lang):
                if not args.no_dedup:
                    if line in seen:
                        continue
                    seen.add(line)
                f.write(line + "\n")
                written += 1

            if (i + 1) % 20000 == 0:
                print(f"  {i + 1} docs / {written} lines...", flush=True)

    print(f"\nDone. Wrote {written} lines -> {args.out} "
          f"(skipped {skipped_docs} empty docs, "
          f"dedup={'off' if args.no_dedup else 'on'})")


if __name__ == "__main__":
    main()
