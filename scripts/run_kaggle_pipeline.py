#!/usr/bin/env python3
"""Automated Kaggle GPU Pipeline Runner for VEYRA-LM.

Usage inside Kaggle Notebook:
    !python scripts/run_kaggle_pipeline.py --model 125m --steps 2000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import zipfile
import torch

from veyra.checkpoints.manager import CheckpointManager
from veyra.config.dataset_config import DatasetMixConfig, DatasetSourceConfig
from veyra.config.model_config import ModelConfig
from veyra.config.training_config import TrainingConfig
from veyra.data.cleaner import TextCleaner
from veyra.data.kaggle import KaggleDatasetAdapter
from veyra.data.pipeline import DatasetPipeline, ShardedTokenDataset
from veyra.model.model import VeyraLM
from veyra.tokenizer.bpe import VeyraTokenizer
from veyra.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="VEYRA-LM Kaggle GPU Training Pipeline")
    parser.add_argument("--model", choices=["tiny", "125m", "350m"], default="125m", help="Target model architecture")
    parser.add_argument("--steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size for tokenizer")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  VEYRA-LM: KAGGLE GPU TRAINING PIPELINE")
    print("=" * 70)

    # 1. GPU Check
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Please enable GPU Accelerator in Kaggle Notebook settings (T4 x2 or P100).")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  PyTorch CUDA Version: {torch.version.cuda}")

    # 2. Working paths
    is_kaggle = KaggleDatasetAdapter.is_running_on_kaggle()
    root_dir = Path("/kaggle/working/veyra") if is_kaggle and Path("/kaggle/working/veyra").exists() else Path(".")
    output_base = Path("/kaggle/working") if is_kaggle else Path("checkpoints")
    output_dir = output_base / f"veyra_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Discover Kaggle input datasets
    adapter = KaggleDatasetAdapter(raw_data_dir=root_dir / "data" / "raw")
    discovered = adapter.discover_kaggle_input_datasets() if is_kaggle else {}
    raw_files = list((root_dir / "data" / "raw").glob("*.txt")) + list((root_dir / "data" / "raw").glob("*.jsonl"))

    print(f"\n[1/5] Preparing Datasets...")
    if discovered:
        print(f"  Discovered {len(discovered)} Kaggle Input Datasets:")
        for name, p in discovered.items():
            print(f"    - {name}: {p}")
            try:
                staged = adapter.stage_kaggle_dataset(name, f"{name}.txt")
                raw_files.append(staged)
                print(f"      Staged to: {staged}")
            except Exception as e:
                print(f"      Could not auto-stage {name}: {e}")
    else:
        print(f"  Using existing raw datasets in {root_dir / 'data' / 'raw'}")

    if not raw_files:
        print("  No raw files found. Creating bootstrap corpus...")
        boot_path = root_dir / "data" / "raw" / "bootstrap_corpus.txt"
        boot_path.parent.mkdir(parents=True, exist_ok=True)
        boot_path.write_text("VEYRA is an artificial intelligence system.\n" * 100, encoding="utf-8")
        raw_files.append(boot_path)

    # 4. Train or load Tokenizer
    tok_dir = root_dir / "data" / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/5] Building VeyraTokenizer...")
    target_vocab = args.vocab_size if args.model != "tiny" else 1024

    all_texts = []
    for rf in raw_files[:5]:
        try:
            with open(rf, "r", encoding="utf-8", errors="replace") as f:
                all_texts.append(f.read())
        except Exception:
            pass

    tokenizer = VeyraTokenizer(vocab_size=target_vocab)
    print(f"  Training tokenizer from {len(all_texts)} source documents (target vocab: {target_vocab})...")
    tokenizer.train_from_texts(all_texts, min_frequency=2)
    tokenizer.save(tok_dir)
    print(f"  VeyraTokenizer trained successfully! Final vocab size: {tokenizer.vocab_size}")

    # 5. Preprocess & Shard
    print(f"\n[3/5] Preprocessing and Sharding Datasets...")
    shards_dir = root_dir / "data" / "shards"
    sources = [
        DatasetSourceConfig(
            name=rf.stem,
            source_type="local",
            path_or_identifier=str(rf),
            file_format="txt",
        )
        for rf in raw_files
    ]
    mix_cfg = DatasetMixConfig(
        tokenizer_path=str(tok_dir),
        max_sequence_length=128 if args.model == "tiny" else 512,
        sources=sources,
    )
    pipeline = DatasetPipeline(tokenizer)
    stats = pipeline.process_and_shard(mix_cfg, shards_dir)
    print(f"  Sharding complete: {stats['train_sequences']} train sequences, {stats['val_sequences']} val sequences.")

    # 6. Load Training Config & Model
    print(f"\n[4/5] Initializing VEYRA-LM Architecture ({args.model})...")
    model_cfg_path = root_dir / "configs" / "models" / f"veyra_{args.model}.yaml"
    model_cfg = ModelConfig.from_yaml(model_cfg_path)
    # Align vocab size with tokenizer
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.context_length = mix_cfg.max_sequence_length

    train_cfg_path = root_dir / "configs" / "training" / (
        f"pretrain_{args.model}_kaggle.yaml"
        if (root_dir / "configs" / "training" / f"pretrain_{args.model}_kaggle.yaml").exists()
        else f"pretrain_{args.model}.yaml"
    )
    train_cfg = TrainingConfig.from_yaml(train_cfg_path)
    train_cfg.output_dir = str(output_dir)
    train_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.steps:
        train_cfg.max_steps = args.steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size

    model = VeyraLM(model_cfg)
    print(f"  Model Parameters: {model.get_num_params():,}")

    train_shards = list((shards_dir / "train").glob("*.pt"))
    val_shards = list((shards_dir / "val").glob("*.pt"))
    train_ds = ShardedTokenDataset(train_shards, seq_length=model_cfg.context_length)
    val_ds = ShardedTokenDataset(val_shards, seq_length=model_cfg.context_length) if val_shards else None

    # 7. Execute Training
    print(f"\n[5/5] Launching Training Loop on {train_cfg.device.upper()}...")
    trainer = Trainer(
        model=model,
        training_config=train_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    train_results = trainer.train()

    # 8. Package Artifacts into ZIP for direct download
    print("\n" + "=" * 70)
    print("  PACKAGING ARTIFACTS FOR DOWNLOAD")
    print("=" * 70)
    zip_path = output_base / f"veyra_{args.model}_artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add checkpoints
        for ckpt in output_dir.glob("*"):
            zf.write(ckpt, arcname=f"checkpoints/{ckpt.name}")
        # Add tokenizer
        for tok_file in tok_dir.glob("*"):
            zf.write(tok_file, arcname=f"tokenizer/{tok_file.name}")
        # Add model config
        model_cfg.to_yaml(output_dir / "model_config.yaml")
        zf.write(output_dir / "model_config.yaml", arcname="model_config.yaml")

    print(f"  Downloadable Archive Created: {zip_path.resolve()}")
    print(f"  Size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("  You can now download this file directly from the Kaggle Output tab!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
