#!/usr/bin/env python3
"""Run Supervised Fine-Tuning (SFT) to produce VEYRA-INSTRUCT.

This turns the raw language model into a helpful, conversational, reasoning assistant
like models running in Ollama (Llama-3-Instruct / Mistral-Instruct).

Usage:
    !python scripts/run_sft_pipeline.py --base-checkpoint checkpoints/veyra_125m/latest.pt --dataset openhermes --steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import zipfile
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veyra.checkpoints.manager import CheckpointManager
from veyra.config.model_config import ModelConfig
from veyra.data.instruction import InstructionDataIngester, InstructionDataset
from veyra.model.model import VeyraLM
from veyra.tokenizer.bpe import VeyraTokenizer
from veyra.training.sft import SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="VEYRA-INSTRUCT: Supervised Fine-Tuning Pipeline")
    parser.add_argument("--base-checkpoint", default=None, help="Path to pretrained base checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer", help="Path to tokenizer directory")
    parser.add_argument(
        "--dataset",
        choices=["all_mix", "openhermes", "code_alpaca", "gsm8k"],
        default="all_mix",
        help="Instruction dataset (all_mix combines OpenHermes + CodeAlpaca + GSM8K)",
    )
    parser.add_argument("--samples", type=int, default=6000, help="Number of instruction pairs to use")
    parser.add_argument("--steps", type=int, default=1500, help="SFT training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for SFT")
    parser.add_argument("--output-dir", default="/kaggle/working/checkpoints/veyra_instruct_125m")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  VEYRA-INSTRUCT: SUPERVISED FINE-TUNING PIPELINE")
    print("  (Transforming Base Model into an Ollama-Grade Assistant)")
    print("=" * 70)

    # 1. Load Tokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"Error: Tokenizer not found at {tok_path}.")
        sys.exit(1)
    tokenizer = VeyraTokenizer.load(tok_path)
    print(f"  Tokenizer Loaded (Vocab: {tokenizer.vocab_size:,})")

    # 2. Load Base Model
    if args.base_checkpoint and Path(args.base_checkpoint).exists():
        print(f"  Loading Pretrained Base Model from: {args.base_checkpoint}")
        model, _ = CheckpointManager.load_checkpoint(args.base_checkpoint)
    else:
        print("  [INFO] No base checkpoint specified or found. Initializing 125M architecture...")
        cfg = ModelConfig.from_yaml("configs/models/veyra_125m.yaml")
        cfg.vocab_size = tokenizer.vocab_size
        model = VeyraLM(cfg)

    # 3. Ingest Instruction Dataset
    ingester = InstructionDataIngester()
    samples: list[dict[str, str]] = []
    try:
        if args.dataset == "all_mix":
            print(f"  Ingesting All-Mix: OpenHermes (Conversational/Poems) + CodeAlpaca (Coding) + GSM8K (Math)...")
            oh_count = int(args.samples * 0.60)
            code_count = int(args.samples * 0.20)
            math_count = int(args.samples * 0.20)
            samples.extend(ingester.stream_instruction_dataset("openhermes", max_samples=oh_count))
            samples.extend(ingester.stream_instruction_dataset("code_alpaca", max_samples=code_count))
            samples.extend(ingester.stream_instruction_dataset("gsm8k", max_samples=math_count))
        else:
            samples = ingester.stream_instruction_dataset(dataset_key=args.dataset, max_samples=args.samples)
    except Exception as e:
        print(f"  [WARN] Could not stream from HF ({e}). Using built-in instruction seeds...")
        samples = [
            {"prompt": "Write a beautiful poem about artificial intelligence and stars.", "response": "In silicon veins the embers gleam,\nA mind awakening from a dream,\nBeneath the cold celestial light,\nWe seek the truths beyond the night."},
            {"prompt": "Explain how an operating system manages virtual memory.", "response": "Virtual memory creates an illusion of a large, uniform memory space using paging, page tables, and MMU hardware translation."},
            {"prompt": "Write a Python function to check if a string is a palindrome.", "response": "def is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]"},
            {"prompt": "What is 25 multiplied by 16?", "response": "25 multiplied by 16 is 400. (25 * 16 = 400)"},
        ] * 150

    dataset = InstructionDataset(samples, tokenizer=tokenizer, max_length=512)
    print(f"  Prepared {len(dataset):,} prompt-masked training sequences.")

    # 4. Run SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=4,
    )
    results = trainer.train()
    print(f"  SFT Training Finished! Checkpoint saved to: {results['checkpoint']}")

    # 5. Export clean inference weights & package for download
    output_dir = Path(args.output_dir)
    release_dir = output_dir / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    ckpt_mgr = CheckpointManager(output_dir)
    clean_model_path = ckpt_mgr.export_inference_model(
        model=model,
        model_config=model.config,
        export_path=release_dir / "model.pt",
        precision="bfloat16" if torch.cuda.is_available() else "float32",
    )
    model.config.to_yaml(release_dir / "model_config.yaml")

    rel_tok_dir = release_dir / "tokenizer"
    rel_tok_dir.mkdir(parents=True, exist_ok=True)
    for tf in tok_path.glob("*"):
        if tf.is_file():
            shutil.copy2(tf, rel_tok_dir / tf.name)

    output_base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("checkpoints")
    zip_path = output_base / "veyra_instruct_125m_model.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in release_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(release_dir))

    alias_zip = output_base / "zonareth_instruct_125m_model.zip"
    shutil.copy2(zip_path, alias_zip)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print("  VEYRA-INSTRUCT MODEL PACKAGED FOR DOWNLOAD!")
    print(f"  File : {zip_path.resolve()} ({size_mb:.2f} MB)")
    print("  Ready to download from Kaggle Output tab and run in app.py!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
