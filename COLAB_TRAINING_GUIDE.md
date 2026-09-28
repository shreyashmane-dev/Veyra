# 🚀 Google Colab Training Manual for ZONARETH-LM

This guide walks you through training **ZONARETH-LM** on **Google Colab** using free cloud GPUs (NVIDIA T4).

---

## ⚡ Quick Start (5 Steps)

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Click **New Notebook** (or click **Upload** and upload [scripts/colab_notebook.ipynb](file:///d:/Projects/veyra/scripts/colab_notebook.ipynb)).

---

### Step 2: Enable GPU Acceleration
Google Colab defaults to CPU. You must switch it to GPU before training:
1. In the top navigation bar, click **Runtime** ➔ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (free tier).
3. Click **Save**.
4. *(Optional verification)* Run this in a cell to check your GPU:
   ```python
   !nvidia-smi
   ```

---

### Step 3: Clone Repository & Install Dependencies
In the first code cell, paste and run:
```python
# Clone project & install dependencies
!git clone https://github.com/shreyashmane-dev/Veyra.git
%cd Veyra
!pip install -e . -q
!pip install datasets tqdm pyyaml -q
print("✅ ZONARETH environment ready!")
```

---

### Step 4: Launch Training Pipeline
In the next code cell, run the automated training pipeline:

#### Recommended: Standard ZONARETH-125M LLM
```python
!python scripts/run_kaggle_pipeline.py --model 125m --steps 2500 --batch-size 8
```

#### Quick Test: ZONARETH-Tiny (~2 minutes)
```python
!python scripts/run_kaggle_pipeline.py --model tiny --steps 300 --batch-size 4
```

**What the pipeline does automatically:**
1. Downloads / streams high-quality pretraining data (TinyStories & FineWeb-Edu).
2. Trains the Byte-Pair Encoding (`ZonarethTokenizer`).
3. Shards tokens into binary `.pt` files.
4. Trains the `ZonarethLM` Transformer with FP16 precision, Rotary Position Embeddings (RoPE), SwiGLU, and Cosine learning rate scheduling.
5. Saves checkpoints and packages everything into a `.zip` archive.

---

### Step 5: Test the Trained Model in Colab
Test your newly trained model with a prompt right inside Colab:
```python
from pathlib import Path
from zonareth.inference.engine import ZonarethInferenceEngine

ckpt = "checkpoints/zonareth_125m/latest.pt"
tok = "data/tokenizer"

engine = ZonarethInferenceEngine.from_checkpoint(ckpt, tok)

prompt = "Once upon a time, in a futuristic city,"
print("ZONARETH >", engine.generate(prompt, max_new_tokens=80, temperature=0.8))
```

---

### Step 6: Download Your Trained Weights
Download your trained model archive (`.zip`) directly to your computer:
```python
from google.colab import files
import glob

zip_files = glob.glob("*.zip") + glob.glob("checkpoints/*.zip")
if zip_files:
    print(f"Downloading {zip_files[0]}...")
    files.download(zip_files[0])
```
*(Or click the folder icon on the left sidebar of Colab, locate `zonareth_125m_artifacts.zip`, and click **Download**).*

---

## 💾 Optional: Save Checkpoints to Google Drive

To avoid losing progress if Colab disconnects:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints to Google Drive
!cp -r checkpoints /content/drive/MyDrive/ZONARETH_Checkpoints/
```

---

## 📁 Using Custom Training Data on Colab

If you want to train ZONARETH on your own text:
1. Click the **Folder icon** on the left sidebar in Colab.
2. Navigate to `Veyra/data/raw/`.
3. Drag and drop your `.txt` or `.jsonl` files into `data/raw/`.
4. Run Step 4—the pipeline will automatically detect and ingest your files!
