from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import sys
import psutil
import torch

from ..checkpoints.manager import CheckpointManager
from ..config.dataset_config import DatasetMixConfig
from ..config.model_config import ModelConfig
from ..config.training_config import TrainingConfig
from ..data.pipeline import DatasetPipeline, ShardedTokenDataset
from ..evaluation.evaluator import ModelEvaluator
from ..inference.engine import VeyraInferenceEngine
from ..model.generation import GenerationEngine
from ..model.model import VeyraLM
from ..tokenizer.bpe import VeyraTokenizer
from ..training.trainer import Trainer


def cmd_system(_args: argparse.Namespace) -> None:
    """Displays hardware details and model scale recommendations."""
    print("\n" + "=" * 60)
    print("  VEYRA SYSTEM & HARDWARE DIAGNOSTIC")
    print("=" * 60)
    print(f"  OS               : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python           : {platform.python_version()}")
    print(f"  PyTorch          : {torch.__version__}")

    # CPU & RAM
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  CPU Logical Cores: {cpu_count}")
    print(f"  RAM Total / Free : {ram_gb:.1f} GB / {ram_avail_gb:.1f} GB")

    # Disk Space
    disk = shutil.disk_usage(".")
    print(f"  Disk Free        : {disk.free / (1024 ** 3):.1f} GB")

    # GPU
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available   : {cuda_avail}")
    if cuda_avail:
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}           : {props.name} ({vram_gb:.1f} GB VRAM)")

    print("\n  Recommended Model Targets:")
    if cuda_avail and torch.cuda.get_device_properties(0).total_memory >= 16 * (1024**3):
        print("  [Optimal] VEYRA-1B (GPU Training & Inference)")
    elif cuda_avail and torch.cuda.get_device_properties(0).total_memory >= 8 * (1024**3):
        print("  [Optimal] VEYRA-350M / VEYRA-125M (GPU Training & Inference)")
    elif cuda_avail:
        print("  [Optimal] VEYRA-125M / VEYRA-TINY (GPU Training & Inference)")
    elif ram_avail_gb >= 16:
        print("  [Optimal] VEYRA-125M (CPU Training) / VEYRA-350M (CPU Inference)")
    else:
        print("  [Optimal] VEYRA-TINY (Development & Fast CPU Testing)")
    print("=" * 60 + "\n")


def cmd_model_info(args: argparse.Namespace) -> None:
    """Displays architecture specifications, parameter breakdown, and memory estimates."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    config = ModelConfig.from_yaml(config_path)
    params = config.calculate_parameter_count()
    memory = config.estimate_memory_mb()

    print("\n" + "=" * 60)
    print(f"  VEYRA-LM MODEL ARCHITECTURE: {config.model_name}")
    print("=" * 60)
    print(f"  Version               : {config.version}")
    print(f"  Vocabulary Size       : {config.vocab_size:,}")
    print(f"  Context Length        : {config.context_length:,} tokens")
    print(f"  Hidden Dimension      : {config.hidden_size}")
    print(f"  Number of Layers      : {config.num_layers}")
    print(f"  Attention Heads (Q)   : {config.num_heads}")
    print(f"  KV Heads (GQA)        : {config.num_kv_heads} (Head Dim: {config.head_dim})")
    print(f"  FFN Intermediate Size : {config.ffn_size}")
    print(f"  Activation Function   : {config.activation}")
    print(f"  Normalization         : {config.normalization} (eps: {config.norm_eps})")
    print(f"  RoPE Base Frequency   : {config.rope_theta}")
    print(f"  Weight Tying          : {config.tie_embeddings}")
    print("-" * 60)
    print("  PARAMETER BREAKDOWN:")
    print(f"    - Embeddings        : {params['embedding']:,}")
    print(f"    - Transformer Layers: {params['layers']:,}")
    print(f"    - Final Norm Layer  : {params['final_norm']:,}")
    print(f"    - LM Head (Untied)  : {params['lm_head']:,}")
    print(f"    - TOTAL PARAMETERS  : {params['total']:,}")
    print("-" * 60)
    print("  MEMORY ESTIMATES:")
    print(f"    - Model Weights (FP32)     : {memory['weights_mb']} MB")
    print(f"    - Training Optimizer State : {memory['training_optimizer_mb']} MB")
    print("=" * 60 + "\n")


def cmd_tokenizer(args: argparse.Namespace) -> None:
    """Manages VEYRA tokenizer operations."""
    subaction = args.tokenizer_action

    if subaction == "train":
        corpus_path = Path(args.corpus)
        out_dir = Path(args.output)
        if not corpus_path.exists():
            print(f"Error: Corpus file not found at '{corpus_path}'")
            sys.exit(1)

        print(f"Reading training corpus from '{corpus_path}'...")
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            texts = [f.read()]

        tokenizer = VeyraTokenizer(vocab_size=args.vocab_size)
        print(f"Training VeyraTokenizer to target vocab size {args.vocab_size}...")
        tokenizer.train_from_texts(texts, min_frequency=2)
        tokenizer.save(out_dir)
        print(f"VeyraTokenizer trained successfully! Final vocab size: {tokenizer.vocab_size}")
        print(f"Saved artifacts to '{out_dir.resolve()}'")

    elif subaction == "inspect":
        tok_dir = Path(args.path)
        tokenizer = VeyraTokenizer.load(tok_dir)
        print(f"Tokenizer at '{tok_dir}':")
        print(f"  Vocab Size     : {tokenizer.vocab_size}")
        print(f"  Merges Learned : {len(tokenizer.merges)}")
        print(f"  Special Tokens : {tokenizer.special_tokens}")

    elif subaction == "encode":
        tok_dir = Path(args.path)
        tokenizer = VeyraTokenizer.load(tok_dir)
        ids = tokenizer.encode(args.text, add_bos=True, add_eos=True)
        print(f"Tokens ({len(ids)}): {ids}")

    elif subaction == "decode":
        tok_dir = Path(args.path)
        tokenizer = VeyraTokenizer.load(tok_dir)
        ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
        text = tokenizer.decode(ids)
        print(f"Decoded: {text}")


def cmd_dataset_process(args: argparse.Namespace) -> None:
    """Cleans, tokenizes, and shards raw datasets."""
    cfg_path = Path(args.config)
    out_dir = Path(args.output)

    mix_config = DatasetMixConfig.from_yaml(cfg_path)
    tok_dir = Path(mix_config.tokenizer_path)
    if not tok_dir.exists():
        print(f"Error: Tokenizer directory not found at '{tok_dir}'. Train tokenizer first.")
        sys.exit(1)

    tokenizer = VeyraTokenizer.load(tok_dir)
    pipeline = DatasetPipeline(tokenizer)
    print(f"Processing dataset with mix configuration '{cfg_path}'...")
    stats = pipeline.process_and_shard(mix_config, out_dir)
    print("Dataset processed and sharded successfully!")
    print(f"  Total Documents : {stats['total_documents']}")
    print(f"  Total Tokens    : {stats['total_tokens']:,}")
    print(f"  Train Sequences : {stats['train_sequences']:,}")
    print(f"  Val Sequences   : {stats['val_sequences']:,}")
    print(f"  Saved to        : {out_dir.resolve()}")


def cmd_train(args: argparse.Namespace) -> None:
    """Runs VEYRA-LM training loop."""
    train_cfg_path = Path(args.config)
    if not train_cfg_path.exists():
        print(f"Error: Training config not found at '{train_cfg_path}'")
        sys.exit(1)

    train_cfg = TrainingConfig.from_yaml(train_cfg_path)
    model_cfg = ModelConfig.from_yaml(train_cfg.model_config_path)

    # Shard paths
    shards_dir = Path(train_cfg.dataset_dir)
    train_shards = list((shards_dir / "train").glob("*.pt"))
    val_shards = list((shards_dir / "val").glob("*.pt"))

    if not train_shards:
        print(f"Error: No training shards found in '{shards_dir}/train'. Run dataset processing first.")
        sys.exit(1)

    train_dataset = ShardedTokenDataset(train_shards, seq_length=model_cfg.context_length)
    val_dataset = ShardedTokenDataset(val_shards, seq_length=model_cfg.context_length) if val_shards else None

    model = VeyraLM(model_cfg)
    trainer = Trainer(
        model=model,
        training_config=train_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


def cmd_checkpoint(args: argparse.Namespace) -> None:
    """Manages model checkpoints."""
    if args.checkpoint_action == "list":
        ckpt_dir = Path(args.dir)
        mgr = CheckpointManager(ckpt_dir)
        ckpts = mgr.list_checkpoints()
        print(f"\nCheckpoints in '{ckpt_dir.resolve()}':")
        if not ckpts:
            print("  No checkpoints found.")
        for c in ckpts:
            step = c.get("step", "N/A")
            loss = c.get("loss", "N/A")
            ppl = c.get("perplexity", "N/A")
            print(f"  - {c.get('checkpoint_file', 'unknown')}: Step={step}, Loss={loss}, PPL={ppl}")
        print()
    elif args.checkpoint_action == "inspect":
        info = CheckpointManager.inspect_checkpoint(args.path)
        print("\n" + "=" * 60)
        print(f"  CHECKPOINT INSPECTION: {info['path']}")
        print("=" * 60)
        print(f"  Step       : {info['step']}")
        print(f"  Loss       : {info['loss']}")
        print(f"  Perplexity : {info['perplexity']}")
        print(f"  Saved At   : {info['saved_at']}")
        print(f"  Model      : {info['model_config'].get('model_name')}")
        print("=" * 60 + "\n")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generates autoregressive text from a trained checkpoint."""
    ckpt_path = Path(args.checkpoint)
    tok_path = Path(args.tokenizer)

    model, _ = CheckpointManager.load_checkpoint(ckpt_path)
    tokenizer = VeyraTokenizer.load(tok_path)
    engine = GenerationEngine(model)

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    print(f"\nPrompt: {args.prompt}")
    print("Generating completion...\n" + "-" * 50)

    def print_stream(token_id: int) -> None:
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        sys.stdout.write(piece)
        sys.stdout.flush()

    out_ids = engine.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=[tokenizer.eos_token_id],
        callback=print_stream,
    )
    print("\n" + "-" * 50 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veyra",
        description="VEYRA: Serious Personal Artificial Intelligence System",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # system
    subparsers.add_parser("system", help="Inspect hardware, GPU, and system resources")

    # model info
    p_model = subparsers.add_parser("model", help="Inspect model architecture configuration")
    p_model.add_argument("info", nargs="?", default="info")
    p_model.add_argument("--config", default="configs/models/veyra_tiny.yaml")

    # tokenizer
    p_tok = subparsers.add_parser("tokenizer", help="Train, inspect, or test tokenizer")
    p_tok.add_argument("tokenizer_action", choices=["train", "inspect", "encode", "decode", "stats"])
    p_tok.add_argument("--corpus", default="data/raw/bootstrap_corpus.txt")
    p_tok.add_argument("--vocab-size", type=int, default=4096)
    p_tok.add_argument("--output", default="data/tokenizer")
    p_tok.add_argument("--path", default="data/tokenizer")
    p_tok.add_argument("--text", default="Hello VEYRA")
    p_tok.add_argument("--ids", default="2, 4, 10, 3")

    # dataset
    p_data = subparsers.add_parser("dataset", help="Process and shard datasets")
    p_data.add_argument("dataset_action", choices=["process"])
    p_data.add_argument("--config", default="configs/datasets/starter_corpus.yaml")
    p_data.add_argument("--output", default="data/shards")

    # train
    p_train = subparsers.add_parser("train", help="Train VEYRA-LM")
    p_train.add_argument("--config", default="configs/training/pretrain_tiny.yaml")

    # checkpoint
    p_ckpt = subparsers.add_parser("checkpoint", help="List or inspect checkpoints")
    p_ckpt.add_argument("checkpoint_action", choices=["list", "inspect"])
    p_ckpt.add_argument("--dir", default="checkpoints/veyra_tiny")
    p_ckpt.add_argument("--path", default="checkpoints/veyra_tiny/latest.pt")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate text from a checkpoint")
    p_gen.add_argument("--checkpoint", default="checkpoints/veyra_tiny/latest.pt")
    p_gen.add_argument("--tokenizer", default="data/tokenizer")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-p", type=float, default=0.9)
    p_gen.add_argument("--max-tokens", type=int, default=100)

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        # If run without subcommands, run the interactive assistant!
        from ..app import main as run_assistant
        run_assistant()
        return

    args = parser.parse_args()
    if args.subcommand == "system":
        cmd_system(args)
    elif args.subcommand == "model":
        cmd_model_info(args)
    elif args.subcommand == "tokenizer":
        cmd_tokenizer(args)
    elif args.subcommand == "dataset":
        cmd_dataset_process(args)
    elif args.subcommand == "train":
        cmd_train(args)
    elif args.subcommand == "checkpoint":
        cmd_checkpoint(args)
    elif args.subcommand == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
