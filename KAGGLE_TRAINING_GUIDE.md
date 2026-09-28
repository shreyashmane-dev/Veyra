# 🚀 How to Run ZONARETH-LM Training on Kaggle (Sleep-Proof Guide)

Everything is configured for **ZONARETH-LM**. You can train overnight on Kaggle's free GPUs (T4 x2 or P100) without losing progress or overflowing disk space.

---

## 💤 How to Train While Sleeping (Zero Data Loss)

> [!IMPORTANT]
> If you run cells manually in an interactive browser tab and close your laptop to sleep, Kaggle's interactive session will disconnect after 60 minutes of inactivity and wipe ephemeral disk!
>
> **The Official Sleep-Proof Method:**
> 1. In your Kaggle notebook, set **Accelerator** to **`GPU T4 x2`** or **`GPU P100`** and turn **Internet** to **`On`**.
> 2. In the top right corner, click **`Save Version`**.
> 3. Under Version Type, choose **`Save & Run All (Commit)`**.
> 4. Click **`Save`**.
> 5. **You can now shut down your PC, close your browser, and go to sleep!** Kaggle will spin up a dedicated 12-hour background VM, run the entire training pipeline to completion, and permanently save all model outputs in the **Output** tab!

---

## 📦 The 6 GB Zip Problem & How It's Solved

- **Why was the zip 6 GB?**
  Raw training checkpoints store the model weights PLUS full AdamW optimizer states (momentum + variance for every single parameter in FP32) across multiple saved steps. This made zip archives 6 GB+, caused browser downloads to fail, and risked overflowing Kaggle's 20 GB disk limit.
- **The Solution:**
  1. **Rolling Step Checkpointing (`--max-to-keep 2`)**: Progress is saved every 250 steps, keeping only the 2 latest step checkpoints + `best.pt` + `latest.pt`. Disk space stays under 4 GB.
  2. **Lightweight Release Archive (`zonareth_125m_model.zip`)**: Strips optimizer states and exports clean FP16 model weights + tokenizer + config + quickstart test script.
     - **Size**: **~250 MB for 125M** (~700 MB for 350M).
     - **Download**: Takes ~15 seconds, never times out, and works directly in `app.py`!
  3. **Resume Checkpoint (`zonareth_125m_resume_checkpoint.zip`)**: A separate archive containing only the latest full checkpoint with optimizer states in case you want to continue training later.

---

## ⚡ Quick Start on Kaggle (3 Steps)

### Step 1: Open Kaggle & Create a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com/) and click **`+ Create`** -> **`New Notebook`**.
2. In the right panel under **Notebook options**:
   - Set **Accelerator** to **`GPU T4 x2`** or **`GPU P100`**.
   - Ensure **Internet** is toggled to **`On`**.

### Step 2: Import the Notebook
Choose either option:
- **Option A (Upload Notebook File - Recommended)**: Click **File** -> **Upload Notebook** and select [scripts/kaggle_notebook.ipynb](file:///d:/Projects/veyra/scripts/kaggle_notebook.ipynb).
- **Option B (Direct One-Liner)**: In a code cell, paste and run:
  ```bash
  !git clone https://github.com/shreyashmane-dev/Veyra.git
  %cd Veyra
  !pip install -e . -q
  !python scripts/run_kaggle_pipeline.py --model 125m --steps 2500 --batch-size 8 --save-interval 250 --max-to-keep 2
  ```

### Step 3: Downloading Your Model
- **Interactive Mode**: Step 6 of the notebook automatically triggers a browser download for `zonareth_125m_model.zip`.
- **Background Mode (Save & Run All)**: Open the notebook, click the **Output** tab on the right sidebar, and click the download button next to `zonareth_125m_model.zip`.
- **Optional Cloud Sync**: Set `HF_TOKEN` and `HF_REPO` (e.g., `username/zonareth-125m`) in Step 7 to push the model straight to Hugging Face Hub.

---

## 📊 Dataset Ingestion on Kaggle

The pipeline supports 3 dataset ingestion options out of the box:
1. **Automatic Streaming / Download (Recommended - Zero Setup)**:
   - Automatically downloads curated **TinyStories** or streams **FineWeb-Edu** pretraining data.
2. **Kaggle Attached Datasets**:
   - Click **`+ Add Input`** in Kaggle and search for any text dataset.
   - The script automatically detects datasets in `/kaggle/input/` and stages them into `data/raw/`.
3. **Custom Files**:
   - Place any `.txt` or `.jsonl` file into `data/raw/`.

---

## 💻 Using Your Downloaded Model Locally

1. Extract `zonareth_125m_model.zip` into your project directory.
2. Run the interactive assistant:
   ```powershell
   python app.py
   ```
3. Or test generation directly:
   ```powershell
   python zonareth_125m_release/quickstart_inference.py
   ```
