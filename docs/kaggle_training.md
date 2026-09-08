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

### Option A: Serious Language Model Training (VEYRA-125M)

```bash
!python scripts/run_kaggle_pipeline.py --model 125m --steps 2500 --batch-size 8
```

This automated runner:
1. Ingests all text and datasets (including any Kaggle datasets attached in `/kaggle/input/`).
2. Trains the native `VeyraTokenizer` to target vocabulary size (32,000).
3. Preprocesses, cleans, deduplicates, and creates binary shards.
4. Initializes the `VEYRA-125M` Transformer architecture.
5. Runs the GPU training loop using `float16` mixed precision and AdamW.
6. Automatically packages the resulting checkpoints (`best.pt`, `latest.pt`) and tokenizer files into a single downloadable archive: `/kaggle/working/veyra_125m_artifacts.zip`.

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
