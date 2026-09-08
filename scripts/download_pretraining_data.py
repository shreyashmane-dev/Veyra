#!/usr/bin/env python3
"""Downloads open, high-quality pretraining text datasets for VEYRA-LM.

Supported datasets:
  - tinystories: High-quality narrative text, excellent for grammar, reasoning, and vocabulary in 125M models.
  - wikitext: Clean Wikipedia articles for factual and encyclopedic knowledge.
  - python_corpus: Python algorithms, data structures, and standard library code examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request


DATASET_URLS = {
    "tinystories_sample": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "wikitext2": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "simple_wiki": "https://raw.githubusercontent.com/LG-AI-EXAONE/EXAONE-1.0/main/examples/sample_pretrain.txt",
}


def download_file(url: str, dest_path: Path, max_bytes: int | None = None) -> Path:
    """Downloads file with streaming and progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Connecting to: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VEYRA-LM Dataset Downloader/1.0"},
    )

    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        total_read = 0
        chunk_size = 1024 * 64  # 64KB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            total_read += len(chunk)
            mb_read = total_read / (1024 * 1024)
            print(f"\r  Downloaded: {mb_read:.2f} MB...", end="", flush=True)
            if max_bytes and total_read >= max_bytes:
                print(f"\n  Reached requested size limit ({max_bytes / (1024*1024):.1f} MB).")
                break

    print(f"\n  Successfully saved to: {dest_path.resolve()} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretraining datasets for VEYRA-LM")
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "wikitext2", "all"],
        default="tinystories",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Target output directory for raw files",
    )
    parser.add_argument(
        "--max-mb",
        type=int,
        default=50,
        help="Maximum megabytes to download (default: 50MB)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    max_bytes = args.max_mb * 1024 * 1024

    if args.dataset in ("tinystories", "all"):
        print("\n--> Downloading TinyStories Pretraining Corpus...")
        dest = out_dir / "tinystories_pretrain.txt"
        download_file(DATASET_URLS["tinystories_sample"], dest, max_bytes=max_bytes)

    if args.dataset in ("wikitext2", "all"):
        print("\n--> Downloading WikiText-2 Pretraining Corpus...")
        dest = out_dir / "wikitext2_pretrain.txt"
        download_file(DATASET_URLS["wikitext2"], dest, max_bytes=max_bytes)

    print("\nDataset preparation finished! You can now run the preprocessing and sharding pipeline.")


if __name__ == "__main__":
    main()
