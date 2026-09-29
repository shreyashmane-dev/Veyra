# Training VEYRA-LM on Kaggle GPU

This guide provides instructions for training **VEYRA-LM** on Kaggle using free NVIDIA GPUs (Tesla T4 or P100) with mixed precision and gradient accumulation.

---

## Prerequisites

1. A free Kaggle account ([kaggle.com](https://www.kaggle.com)).
2. Phone verification on Kaggle (required to access free GPU accelerators and internet connectivity).

---

## Step 1: Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the right-hand sidebar under **Notebook Options**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle to **On**.
   - **Language**: Python.

---

## Step 2: Clone & Set Up the Repository

In the first cell of your Kaggle notebook, run:

```bash
!git clone https://github.com/shreyashmane-dev/Veyra.git
%cd Veyra
!pip install -e . -q
```

---

## Step 3: Verify GPU Detection

In the next cell, run the VEYRA system diagnostic to verify your Kaggle GPU:

```bash
!python -m veyra.cli.main system
```

You should see:
```text
CUDA Available   : True
GPU 0            : Tesla T4 (or Tesla P100) (15.8 GB VRAM)
```

---

## Step 4: Launch Training Pipeline

### Option A: Full-Scale Overnight Training (Pretraining 8,000 steps + SFT Instruct 1,500 steps)
*Recommended for full conversational intelligence, poetry, coding, and mathematical reasoning:*

```bash
!python scripts/run_kaggle_pipeline.py \
    --model 125m \
    --steps 8000 \
    --batch-size 8 \
    --num-docs 25000 \
    --save-interval 500 \
    --max-to-keep 2 \
    --run-sft \
    --sft-steps 1500
```

This automated runner:
1. Streams 25,000 multi-domain documents (*FineWeb-Edu*, *Cosmopedia*, *FineMath*, *The Stack*, *SlimPajama*).
2. Builds the optimal 8,192 BPE tokenizer in ~10 seconds.
3. Pretrains the 85.5M parameter Transformer on GPU with mixed precision and AdamW.
4. Keeps only the 2 latest step checkpoints + `best.pt` so Kaggle disk never runs out of space.
5. Immediately chains into **SFT instruction tuning** on *OpenHermes 2.5* (poems & conversation), *CodeAlpaca* (coding), and *GSM8K* (math).
6. Automatically packages the final model into `/kaggle/working/veyra_instruct_125m_model.zip`.

### Option B: Pretraining Only (No SFT)

```bash
!python scripts/run_kaggle_pipeline.py --model 125m --steps 5000 --batch-size 8 --num-docs 15000
```

### Option B: Quick Verification Run (VEYRA-TINY)

If you want a 1-minute test to verify the GPU loop:

```bash
!python scripts/run_kaggle_pipeline.py --model tiny --steps 100
```

---

## Step 5: (Optional) Adding External Kaggle Datasets

To train on large public corpora:
1. In your Kaggle notebook, click **"+ Add Data"** in the top right.
2. Search and attach any dataset (e.g. `tinystories`, `wikitext`, `python-code-dataset`).
3. VEYRA's `KaggleDatasetAdapter` will automatically discover datasets in `/kaggle/input/`, stage them, clean them, and incorporate them into the training shards!

---

## Step 6: Download Checkpoint to Your Local Machine

Once training completes:
1. Look at the right sidebar under **Output** (`/kaggle/working/`).
2. You will see `veyra_125m_artifacts.zip`.
3. Click the three dots next to the file and select **Download**.
4. On your local machine, extract the zip file:
   - Put checkpoints into: `d:\Projects\veyra\checkpoints\veyra_125m\`
   - Put tokenizer into: `d:\Projects\veyra\data\tokenizer\`

---

## Step 7: Launch VEYRA Locally with the Trained Model

Once downloaded, start VEYRA locally:

```powershell
python -m veyra
```

VEYRA will detect your newly trained `VEYRA-125M` weights and launch the cognitive terminal assistant powered by your model!
