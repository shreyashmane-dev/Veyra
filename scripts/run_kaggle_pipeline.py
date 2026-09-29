#!/usr/bin/env python3
"""Automated Kaggle GPU Pipeline Runner for ZONARETH-LM.

Usage inside Kaggle Notebook:
    !python scripts/run_kaggle_pipeline.py --model 125m --steps 2500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
import torch

# Ensure repository root is in sys.path regardless of execution directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from zonareth.checkpoints.manager import CheckpointManager
    from zonareth.config.dataset_config import DatasetMixConfig, DatasetSourceConfig
    from zonareth.config.model_config import ModelConfig
    from zonareth.config.training_config import TrainingConfig
    from zonareth.data.cleaner import TextCleaner
    from zonareth.data.kaggle import KaggleDatasetAdapter
    from zonareth.data.pipeline import DatasetPipeline, ShardedTokenDataset
    from zonareth.model.model import ZonarethLM
    from zonareth.tokenizer.bpe import ZonarethTokenizer
    from zonareth.training.trainer import Trainer
except ImportError:
    from veyra.checkpoints.manager import CheckpointManager
    from veyra.config.dataset_config import DatasetMixConfig, DatasetSourceConfig
    from veyra.config.model_config import ModelConfig
    from veyra.config.training_config import TrainingConfig
    from veyra.data.cleaner import TextCleaner
    from veyra.data.kaggle import KaggleDatasetAdapter
    from veyra.data.pipeline import DatasetPipeline, ShardedTokenDataset
    from veyra.model.model import VeyraLM as ZonarethLM
    from veyra.tokenizer.bpe import VeyraTokenizer as ZonarethTokenizer
    from veyra.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="ZONARETH-LM Kaggle GPU Training Pipeline")
    parser.add_argument("--model", choices=["tiny", "125m", "350m"], default="125m", help="Target model architecture")
    parser.add_argument("--steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--save-interval", type=int, default=None, help="Steps between periodic checkpoint saves")
    parser.add_argument("--max-to-keep", type=int, default=2, help="Number of rolling step checkpoints to retain (prevents Kaggle disk overflow)")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size for tokenizer (default: 8192)")
    parser.add_argument(
        "--recipe",
        choices=["zonareth_mix", "veyra_mix", "fineweb_edu", "cosmopedia", "finemath", "the_stack", "auto"],
        default="auto",
        help="Pretraining dataset recipe to stream (FineWeb-Edu, Cosmopedia, etc.)",
    )
    parser.add_argument("--num-docs", type=int, default=10000, help="Number of documents to stream")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--run-sft", action="store_true", help="Automatically run SFT instruction tuning immediately after pretraining")
    parser.add_argument("--sft-steps", type=int, default=1500, help="Number of SFT steps if --run-sft is enabled (default: 1500)")
    parser.add_argument("--hf-repo", type=str, default=None, help="Optional Hugging Face repo ID (e.g. username/zonareth-125m) to push model")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face user access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ZONARETH-LM: KAGGLE GPU TRAINING PIPELINE")
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
    candidates = [
        Path("/kaggle/working/Zonareth"),
        Path("/kaggle/working/zonareth"),
        Path("/kaggle/working/Veyra"),
        Path("/kaggle/working/veyra"),
    ]
    root_dir = next((c for c in candidates if (c / "configs").exists()), Path.cwd())
    output_base = Path("/kaggle/working") if is_kaggle else Path("checkpoints")
    output_dir = output_base / f"zonareth_{args.model}"
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
            except Exception as e:
                print(f"      Could not auto-stage {name}: {e}")
    else:
        print(f"  Using local raw datasets in {root_dir / 'data' / 'raw'}")

    total_raw_bytes = sum(f.stat().st_size for f in raw_files if f.exists())
    recipe = args.recipe
    if recipe == "auto":
        recipe = "zonareth_mix" if total_raw_bytes < 100_000 else "none"

    if recipe != "none":
        print(f"\n  Activating Dataset Recipe: [{recipe}] (FineWeb-Edu, Cosmopedia, FineMath, The Stack)...")
        try:
            try:
                from zonareth.data.hf_streamer import PretrainingDatasetIngester
            except ImportError:
                from veyra.data.hf_streamer import PretrainingDatasetIngester
            ingester = PretrainingDatasetIngester(raw_dir=root_dir / "data" / "raw")
            if recipe in ("zonareth_mix", "veyra_mix"):
                paths = ingester.build_veyra_curated_recipe(total_docs=args.num_docs)
                raw_files.extend(paths)
            else:
                p = ingester.stream_dataset(recipe, num_documents=args.num_docs)
                raw_files.append(p)
        except Exception as e:
            print(f"  [INFO] Streaming unavailable ({e}). Falling back to HTTP download...")
            try:
                from scripts.download_pretraining_data import download_file, DATASET_URLS
                dl_path = root_dir / "data" / "raw" / "tinystories_pretrain.txt"
                download_file(DATASET_URLS["tinystories_sample"], dl_path, max_bytes=20 * 1024 * 1024)
                if dl_path.exists():
                    raw_files.append(dl_path)
            except Exception as e2:
                print(f"  [WARN] Fallback download failed: {e2}. Generating synthetic expansion...")
                boot_path = root_dir / "data" / "raw" / "bootstrap_corpus.txt"
                aug_path = root_dir / "data" / "raw" / "augmented_corpus.txt"
                base_content = boot_path.read_text(encoding="utf-8") if boot_path.exists() else "ZONARETH.\n"
                aug_path.write_text((base_content + "\n\n") * 60, encoding="utf-8")
                raw_files.append(aug_path)

    # Deduplicate raw_files
    raw_files = list({f.resolve(): f for f in raw_files if f.exists()}.values())

    # 4. Train or load Tokenizer
    tok_dir = root_dir / "data" / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/5] Building ZonarethTokenizer...")
    target_vocab = args.vocab_size if args.model != "tiny" else 1024

    if (tok_dir / "vocab.json").exists() and (tok_dir / "merges.json").exists():
        print(f"  Existing ZonarethTokenizer found in {tok_dir}. Loading...")
        tokenizer = ZonarethTokenizer.load(tok_dir)
        print(f"  ZonarethTokenizer loaded successfully! Vocab size: {tokenizer.vocab_size}")
    else:
        all_texts = []
        for rf in raw_files:
            try:
                with open(rf, "r", encoding="utf-8", errors="replace") as f:
                    if rf.suffix == ".jsonl":
                        for line_idx, line in enumerate(f):
                            if line_idx >= 400:
                                break
                            rec = json.loads(line)
                            if "text" in rec:
                                all_texts.append(rec["text"][:1500])
                    else:
                        sample_text = f.read(1 * 1024 * 1024)
                        if sample_text:
                            all_texts.append(sample_text)
            except Exception:
                pass

        tokenizer = ZonarethTokenizer(vocab_size=target_vocab)
        print(f"  Training tokenizer from {len(all_texts)} text blocks (target vocab: {target_vocab})...")
        tokenizer.train_from_texts(all_texts, min_frequency=2, show_progress=True)
        tokenizer.save(tok_dir)
        print(f"  ZonarethTokenizer trained successfully! Final vocab size: {tokenizer.vocab_size}")

    # 5. Preprocess & Shard
    print(f"\n[3/5] Preprocessing and Sharding Datasets...")
    shards_dir = root_dir / "data" / "shards"
    sources = []
    for rf in raw_files:
        fmt = "jsonl" if rf.suffix == ".jsonl" else ("csv" if rf.suffix == ".csv" else "txt")
        sources.append(
            DatasetSourceConfig(
                name=rf.stem,
                source_type="local",
                path_or_identifier=str(rf),
                file_format=fmt,
                text_column="text",
            )
        )
    mix_cfg = DatasetMixConfig(
        tokenizer_path=str(tok_dir),
        max_sequence_length=128 if args.model == "tiny" else 512,
        sources=sources,
    )
    pipeline = DatasetPipeline(tokenizer)
    stats = pipeline.process_and_shard(mix_cfg, shards_dir)
    print(f"  Sharding complete: {stats['train_sequences']} train sequences, {stats['val_sequences']} val sequences.")

    # 6. Load Training Config & Model
    print(f"\n[4/5] Initializing ZONARETH-LM Architecture ({args.model})...")
    model_cfg_candidates = [
        root_dir / "configs" / "models" / f"zonareth_{args.model}.yaml",
        root_dir / "configs" / "models" / f"veyra_{args.model}.yaml",
    ]
    model_cfg_path = next(p for p in model_cfg_candidates if p.exists())
    model_cfg = ModelConfig.from_yaml(model_cfg_path)
    model_cfg.vocab_size = tokenizer.vocab_size
    model_cfg.context_length = mix_cfg.max_sequence_length

    train_cfg_candidates = [
        root_dir / "configs" / "training" / f"pretrain_zonareth_{args.model}_kaggle.yaml",
        root_dir / "configs" / "training" / f"pretrain_zonareth_{args.model}.yaml",
        root_dir / "configs" / "training" / f"pretrain_{args.model}_kaggle.yaml",
        root_dir / "configs" / "training" / f"pretrain_{args.model}.yaml",
    ]
    train_cfg_path = next(p for p in train_cfg_candidates if p.exists())
    train_cfg = TrainingConfig.from_yaml(train_cfg_path)
    train_cfg.output_dir = str(output_dir)
    train_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.steps:
        train_cfg.max_steps = args.steps
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.save_interval:
        train_cfg.save_interval = args.save_interval

    model = ZonarethLM(model_cfg)
    print(f"  Model Parameters: {model.get_num_params():,}")

    train_shards = list((shards_dir / "train").glob("*.pt"))
    val_shards = list((shards_dir / "val").glob("*.pt"))
    train_ds = ShardedTokenDataset(train_shards, seq_length=model_cfg.context_length)
    val_ds = ShardedTokenDataset(val_shards, seq_length=model_cfg.context_length) if val_shards else None

    # 7. Execute Training (Progress saved every save_interval steps; rolling checkpoints prevent disk overflow)
    print(f"\n[5/5] Launching Training Loop on {train_cfg.device.upper()}...")
    print(f"  Step Progress Saved Every: {train_cfg.save_interval} steps")
    print(f"  Rolling Checkpoints Kept : {args.max_to_keep} (plus best.pt and latest.pt)")
    trainer = Trainer(
        model=model,
        training_config=train_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    train_results = trainer.train()

    # 8. Package Artifacts: Separate Lightweight Model (~250MB) from Full Resume Checkpoint
    print("\n" + "=" * 70)
    print("  PACKAGING ARTIFACTS FOR RELIABLE DOWNLOAD")
    print("=" * 70)

    # 8A. Lightweight Inference Release Package (NO optimizer state, ~250MB for 125M, ~700MB for 350M)
    release_dir = output_base / f"zonareth_{args.model}_release"
    release_dir.mkdir(parents=True, exist_ok=True)
    release_tok_dir = release_dir / "tokenizer"
    release_tok_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer
    for tf in tok_dir.glob("*"):
        shutil.copy2(tf, release_tok_dir / tf.name)

    # Copy clean model weights
    clean_model_pt = output_dir / "model.pt"
    if clean_model_pt.exists():
        shutil.copy2(clean_model_pt, release_dir / "model.pt")
    elif (output_dir / "latest.pt").exists():
        # Fallback: export clean weights on the fly
        try:
            from zonareth.checkpoints.manager import CheckpointManager
        except ImportError:
            from veyra.checkpoints.manager import CheckpointManager
        mgr = CheckpointManager(output_dir)
        mgr.export_inference_model(model, model_cfg, release_dir / "model.pt", precision=train_cfg.precision)

    # Write model config
    model_cfg.to_yaml(release_dir / "model_config.yaml")

    # Add standalone inference test runner inside release folder
    quickstart_code = """# Quickstart Inference for ZONARETH-LM
import torch
from pathlib import Path
import yaml
import sys

from zonareth.model.model import ZonarethLM
from zonareth.config.model_config import ModelConfig
from zonareth.tokenizer.bpe import ZonarethTokenizer
from zonareth.inference.engine import ZonarethInferenceEngine

ckpt_path = Path(__file__).parent / "model.pt"
tok_path = Path(__file__).parent / "tokenizer"
cfg_path = Path(__file__).parent / "model_config.yaml"

if not ckpt_path.exists():
    print(f"Error: {ckpt_path} not found.")
    sys.exit(1)

print(f"Loading ZONARETH-LM from {ckpt_path}...")
engine = ZonarethInferenceEngine.from_checkpoint(str(ckpt_path), str(tok_path))
prompt = "Once upon a time in a digital realm"
print(f"Prompt: {prompt}")
output = engine.generate(prompt, max_new_tokens=50, temperature=0.8)
print(f"Generated: {output}")
"""
    (release_dir / "quickstart_inference.py").write_text(quickstart_code, encoding="utf-8")

    # Create lightweight ZIP
    model_zip_path = output_base / f"zonareth_{args.model}_model.zip"
    with zipfile.ZipFile(model_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in release_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(release_dir))

    model_zip_size_mb = model_zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  [1/2] Lightweight Model Archive Created (Recommended for Download):")
    print(f"        File : {model_zip_path.resolve()}")
    print(f"        Size : {model_zip_size_mb:.2f} MB (Small, fast & ready for app.py)")

    # 8B. Full Resume Checkpoint Package (Contains single latest.pt for continuing training)
    resume_zip_path = output_base / f"zonareth_{args.model}_resume_checkpoint.zip"
    with zipfile.ZipFile(resume_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        latest_pt = output_dir / "latest.pt"
        if latest_pt.exists():
            zf.write(latest_pt, arcname="checkpoints/latest.pt")
        best_pt = output_dir / "best.pt"
        if best_pt.exists():
            zf.write(best_pt, arcname="checkpoints/best.pt")
        for tf in tok_dir.glob("*"):
            zf.write(tf, arcname=f"tokenizer/{tf.name}")
        zf.write(release_dir / "model_config.yaml", arcname="model_config.yaml")

    resume_zip_size_mb = resume_zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  [2/2] Full Training Resume Archive Created:")
    print(f"        File : {resume_zip_path.resolve()}")
    print(f"        Size : {resume_zip_size_mb:.2f} MB (Includes optimizer state for resumption)")

    # 8C. Optional: Hugging Face Hub Auto-Upload
    hf_repo = args.hf_repo or os.environ.get("HF_REPO")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_repo:
        print(f"\n  [CLOUD BACKUP] Pushing to Hugging Face Hub: {hf_repo}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, exist_ok=True, private=True)
            api.upload_folder(
                folder_path=str(release_dir),
                repo_id=hf_repo,
                commit_message=f"Auto-upload ZONARETH-{args.model} trained on Kaggle",
            )
            print(f"  [SUCCESS] Model pushed to: https://huggingface.co/{hf_repo}")
        except Exception as e:
            print(f"  [WARN] Hugging Face push failed: {e}")

    print("\n" + "=" * 70)
    print("  BASE PRETRAINING ARTIFACTS SAFELY SAVED!")
    print(f"  1. Lightweight Model : {model_zip_path.name} ({model_zip_size_mb:.2f} MB)")
    print(f"  2. Full Checkpoint   : {resume_zip_path.name} ({resume_zip_size_mb:.2f} MB)")
    print("  Files are available in Kaggle Working Directory / Output tab.")
    print("=" * 70 + "\n")

    # 9. Optional: Automated SFT Instruction Tuning (All-Mix)
    if args.run_sft:
        print("\n" + "=" * 70)
        print("  LAUNCHING AUTOMATED SFT INSTRUCTION TUNING (ALL-MIX)")
        print("  (Fine-tuning on OpenHermes + CodeAlpaca + GSM8K)")
        print("=" * 70)
        from scripts.run_sft_pipeline import main as run_sft_main
        orig_argv = sys.argv
        sys.argv = [
            "run_sft_pipeline.py",
            "--base-checkpoint", str(output_dir / "latest.pt"),
            "--tokenizer", str(tok_dir),
            "--dataset", "all_mix",
            "--steps", str(args.sft_steps),
            "--batch-size", str(args.batch_size or 8),
            "--output-dir", str(output_base / f"veyra_instruct_{args.model}"),
        ]
        try:
            run_sft_main()
        finally:
            sys.argv = orig_argv


if __name__ == "__main__":
    main()
