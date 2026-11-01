import json
from pathlib import Path

nb = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 VEYRA-INSTRUCT: Automated GPU SFT Fine-Tuning\n",
                "### Transforming Pretrained Veyra-125M into an Intelligent, Conversational Assistant\n",
                "---\n",
                "This automated notebook:\n",
                "1. Downloads `best.pt` checkpoint from `shreyashmane649/kings` at Google Cloud datacenter speeds (~5s).\n",
                "2. Clones the latest Veyra codebase and dependencies.\n",
                "3. Runs **Supervised Fine-Tuning (SFT)** on **All-Mix** (OpenHermes 2.5 + CodeAlpaca + GSM8K).\n",
                "4. Executes the automated capability verification suite (Math reasoning, Poems, Python Code).\n",
                "5. Packages `veyra_instruct_125m_model.zip` ready for one-click local download."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Verify GPU & Download base checkpoint from kings\n",
                "import os, sys, torch\n",
                "from pathlib import Path\n",
                "\n",
                "print(f'PyTorch Version : {torch.__version__}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'✅ GPU Detected  : {torch.cuda.get_device_name(0)}')\n",
                "else:\n",
                "    print('⚠️ Running on CPU')\n",
                "\n",
                "# Pull previous checkpoint at high speed\n",
                "%cd /kaggle/working\n",
                "!mkdir -p base_input\n",
                "!kaggle kernels output shreyashmane649/kings -p /kaggle/working/base_input\n",
                "print('\\nFiles in base_input:')\n",
                "!ls -la /kaggle/working/base_input\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 2: Fresh Environment Setup & Install Latest Codebase\n",
                "%cd /kaggle/working\n",
                "!rm -rf Veyra\n",
                "!git clone https://github.com/shreyashmane-dev/Veyra.git\n",
                "%cd /kaggle/working/Veyra\n",
                "!pip install -e . -q\n",
                "!pip install datasets tqdm pyyaml huggingface_hub -q\n",
                "print('\\n✅ Veyra codebase installed!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 3: Run SFT Instruction Tuning (All-Mix: OpenHermes + CodeAlpaca + GSM8K)\n",
                "from pathlib import Path\n",
                "\n",
                "base_dir = Path('/kaggle/working/base_input')\n",
                "candidates = [\n",
                "    base_dir / 'zonareth_125m' / 'best.pt',\n",
                "    base_dir / 'zonareth_125m' / 'step_0006000.pt',\n",
                "    base_dir / 'best.pt',\n",
                "]\n",
                "ckpt_file = next((p for p in candidates if p.exists()), None)\n",
                "tok_file = base_dir / 'Veyra' / 'data' / 'tokenizer'\n",
                "if not tok_file.exists():\n",
                "    tok_file = Path('/kaggle/working/Veyra/data/tokenizer')\n",
                "\n",
                "print(f'Base Checkpoint : {ckpt_file}')\n",
                "print(f'Tokenizer Path  : {tok_file}')\n",
                "\n",
                "if ckpt_file:\n",
                "    !python scripts/run_sft_pipeline.py \\\n",
                "        --base-checkpoint {ckpt_file} \\\n",
                "        --tokenizer {tok_file} \\\n",
                "        --dataset all_mix \\\n",
                "        --steps 1500 \\\n",
                "        --batch-size 8 \\\n",
                "        --output-dir /kaggle/working/veyra_instruct_125m\n",
                "else:\n",
                "    print('❌ Checkpoint not found in base_input!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 4: Verification Suite (Math, Poems, Python Code, Dialogue)\n",
                "from pathlib import Path\n",
                "from veyra.inference.engine import VeyraInferenceEngine\n",
                "\n",
                "release_ckpt = Path('/kaggle/working/veyra_instruct_125m/release/model.pt')\n",
                "tok_dir = Path('/kaggle/working/veyra_instruct_125m/release/tokenizer')\n",
                "if not release_ckpt.exists():\n",
                "    release_ckpt = Path('/kaggle/working/veyra_instruct_125m/best.pt')\n",
                "\n",
                "if release_ckpt.exists() and tok_dir.exists():\n",
                "    print(f'Loading fine-tuned model from {release_ckpt}...')\n",
                "    engine = VeyraInferenceEngine.from_checkpoint(str(release_ckpt), str(tok_dir), max_new_tokens=100)\n",
                "    engine.model.config.is_instruct = True\n",
                "\n",
                "    test_suite = [\n",
                "        ('🎭 Poetry Generation', 'Write a short, beautiful poem about artificial intelligence gazing at the stars.'),\n",
                "        ('🧮 Math Reasoning', 'What is 45 multiplied by 16? Explain the calculation step-by-step.'),\n",
                "        ('💻 Python Code', 'Write a clean Python function to check if a number is prime.'),\n",
                "        ('🧠 Technical Explanation', 'Explain how an operating system manages virtual memory in simple terms.')\n",
                "    ]\n",
                "\n",
                "    print('\\n' + '=' * 65)\n",
                "    print('   VEYRA-INSTRUCT: CAPABILITY VERIFICATION RESULTS')\n",
                "    print('=' * 65)\n",
                "    for tag, prompt in test_suite:\n",
                "        print(f'\\n[{tag}]')\n",
                "        print(f'User : {prompt}')\n",
                "        resp = engine.generate(prompt, context=[])\n",
                "        print(f'VEYRA: {resp}')\n",
                "        print('-' * 50)\n",
                "else:\n",
                "    print('Model release files not found. Check Step 3 output.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 5: Packaging & Download Verification\n",
                "from pathlib import Path\n",
                "from IPython.display import display, FileLink, HTML\n",
                "\n",
                "zip_path = Path('/kaggle/working/veyra_instruct_125m_model.zip')\n",
                "if zip_path.exists():\n",
                "    size_mb = zip_path.stat().st_size / (1024 * 1024)\n",
                "    print(f'✅ FINAL MODEL READY FOR DOWNLOAD: {zip_path.name} ({size_mb:.1f} MB)')\n",
                "    display(FileLink(str(zip_path.name)))\n",
                "    html_btn = f'''\n",
                "    <div style=\"margin-top: 15px;\">\n",
                "        <a href=\"/files/{zip_path.name}\" download=\"{zip_path.name}\"\n",
                "           style=\"background-color: #10b981; color: white; padding: 12px 24px; font-weight: bold; border-radius: 8px; text-decoration: none; display: inline-block; font-size: 16px;\">\n",
                "           ⬇️ Download {zip_path.name} ({size_mb:.1f} MB)\n",
                "        </a>\n",
                "    </div>\n",
                "    '''\n",
                "    display(HTML(html_btn))\n",
                "else:\n",
                "    print('Model zip not found.')\n"
            ]
        }
    ]
}

Path("kaggle_sft_runner/kings.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
meta = {
    "id": "shreyashmane649/veyra-sft-instruct",
    "title": "Veyra SFT Instruct",
    "code_file": "kings.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "false",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}
Path("kaggle_sft_runner/kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("SFT Runner updated and metadata saved!")
