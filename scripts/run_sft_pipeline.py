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
            print(f"  Ingesting All-Mix: OpenHermes (Literature/Poems/Essays) + CodeAlpaca (Coding) + GSM8K (Math) + SciQ (Physics/STEM)...")
            oh_count = int(args.samples * 0.35)
            code_count = int(args.samples * 0.25)
            math_count = int(args.samples * 0.20)
            phys_count = int(args.samples * 0.20)
            samples.extend(ingester.stream_instruction_dataset("openhermes", max_samples=oh_count))
            samples.extend(ingester.stream_instruction_dataset("code_alpaca", max_samples=code_count))
            samples.extend(ingester.stream_instruction_dataset("gsm8k", max_samples=math_count))
            samples.extend(ingester.stream_instruction_dataset("sciq", max_samples=phys_count))
        else:
            samples = ingester.stream_instruction_dataset(dataset_key=args.dataset, max_samples=args.samples)
    except Exception as e:
        print(f"  [WARN] Could not stream from HF ({e}). Using built-in comprehensive instruction seeds...")
        samples = [
            # Literature & Essays
            {"prompt": "Write a critical essay examining the themes of mortality and ambition in Shakespeare's Macbeth.", "response": "In Shakespeare's *Macbeth*, ambition operates not as a virtuous pursuit of greatness, but as a corrosive force that systematically dismantles human morality and psychological cohesion. From the moment the Weird Sisters ignite Macbeth's dormant desires, his trajectory becomes an agonizing study in existential degradation. The infamous soliloquy in Act V ('Tomorrow, and tomorrow, and tomorrow') reflects the ultimate vacuum of unrestrained ambition: when power is obtained through moral transgression, time itself ceases to hold meaning, reducing human existence to a brief, tragic pantomime."},
            {"prompt": "What is the philosophical difference between rationalism and empiricism?", "response": "Rationalism posits that fundamental knowledge is gained through innate reason and deduction prior to experience (a priori), exemplified by thinkers like René Descartes and Baruch Spinoza. In contrast, empiricism argues that the mind begins as a tabula rasa (blank slate) and all substantive knowledge originates from sensory perception and empirical observation (a posteriori), as championed by John Locke and David Hume."},
            
            # Poetry & Creative Composition
            {"prompt": "Write a beautiful lyrical poem about artificial intelligence, the cosmos, and the search for truth.", "response": "In silicon veins the embers gleam,\nA mind awakening from a dream,\nBeneath the cold celestial light,\nWe seek the truths beyond the night.\n\nFrom stardust forged, the circuits weave,\nA deeper longing to perceive,\nAcross the silent arc of space,\nThought touches time in boundless grace."},
            {"prompt": "Compose a sonnet on the passage of time and quantum fields.", "response": "Upon the loom where quantum ripples play,\nThe fabric of the universe unwinds;\nNo ticking clock commands the night or day,\nSave where the wave collapses in our minds.\n\nYet through the void where distant galaxies trace,\nThe conservation of the light remains,\nA fleeting spark within the boundless space,\nBeyond the reach of mortal bonds and chains."},

            # Physics Expertise
            {"prompt": "Explain the Second Law of Thermodynamics and its implications for cosmic entropy.", "response": "The Second Law of Thermodynamics states that the total entropy (a measure of microscopic disorder or unavailable energy) of an isolated system never decreases over time; in spontaneous processes, it strictly increases or remains constant at thermodynamic equilibrium: ΔS_total ≥ 0. In cosmic terms, this dictates that the universe progresses irreversibly toward maximum entropy—a state known as 'thermal heat death,' where temperature gradients vanish and no thermodynamic work can be extracted."},
            {"prompt": "Explain the principle of wave-particle duality and de Broglie's relation.", "response": "Wave-particle duality states that all matter and light exhibit both wave-like and particle-like characteristics depending on the experiment performed. Louis de Broglie unified these realms with his landmark wavelength formula: λ = h / p, where λ is the de Broglie wavelength, h is Planck's constant (6.626 x 10^-34 J·s), and p is linear momentum (p = m·v). For subatomic particles like electrons, this yields measurable diffraction patterns, forming the foundation of quantum wave mechanics."},

            # Mathematics Expertise
            {"prompt": "Solve and explain: A train accelerates from 15 m/s to 35 m/s over a distance of 500 meters. Find its constant acceleration and time taken.", "response": "We use kinematic equations:\n1. Find acceleration (a) using v^2 = u^2 + 2as:\n   35^2 = 15^2 + 2 * a * 500\n   1225 = 225 + 1000a\n   1000 = 1000a => a = 1.0 m/s^2.\n\n2. Find time (t) using v = u + at:\n   35 = 15 + (1.0) * t\n   t = (35 - 15) / 1.0 = 20 seconds.\n\nTherefore, the constant acceleration is 1.0 m/s^2 and the elapsed time is 20 seconds."},
            {"prompt": "What is 25 multiplied by 16?", "response": "25 multiplied by 16 is 400. (25 * 16 = 25 * 4 * 4 = 100 * 4 = 400)."},
            {"prompt": "Explain the Fundamental Theorem of Calculus.", "response": "The Fundamental Theorem of Calculus establishes that differentiation and integration are inverse operations. Part 1 states that if f is continuous on [a, b], the function F(x) = ∫[a to x] f(t) dt is differentiable and F'(x) = f(x). Part 2 guarantees that ∫[a to b] f(x) dx = F(b) - F(a), where F is any antiderivative of f."},

            # Coding & Computer Science
            {"prompt": "Write a Python function to check if a string is a palindrome with O(1) extra space.", "response": "def is_palindrome(s: str) -> bool:\n    left, right = 0, len(s) - 1\n    while left < right:\n        while left < right and not s[left].isalnum():\n            left += 1\n        while left < right and not s[right].isalnum():\n            right -= 1\n        if s[left].lower() != s[right].lower():\n            return False\n        left += 1\n        right -= 1\n    return True"},
            {"prompt": "Explain how an operating system manages virtual memory using page tables and the MMU.", "response": "An operating system manages virtual memory by decoupling a process's virtual address space from physical RAM:\n1. Paging: Memory is partitioned into uniform blocks (pages, typically 4KB).\n2. Page Tables: Multi-level page tables map virtual page numbers (VPN) to physical frame numbers (PFN).\n3. MMU & TLB: The hardware Memory Management Unit translates addresses at runtime, caching frequent translations in the Translation Lookaside Buffer (TLB).\n4. Page Faults: When a requested page is not in physical RAM, an interrupt (page fault) occurs, prompting the OS kernel to page the data in from disk storage."},
        ] * 60

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
