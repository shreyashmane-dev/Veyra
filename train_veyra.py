"""Placeholder entry point for VEYRA-LM training.

This intentionally does not download data or train a fake 'ChatGPT clone'.
The next phase will add the tokenizer, dataset cleaner, Transformer model,
checkpointing, evaluation, and a real training loop.
"""
from pathlib import Path

DATA = Path("data")


def main() -> None:
    print("VEYRA-LM training pipeline")
    print(f"Dataset root: {DATA.resolve()}")
    print("Status: tokenizer/model/training loop will be added in the next phase.")
    print("Put legally usable source datasets under data/raw/ before training.")


if __name__ == "__main__":
    main()
