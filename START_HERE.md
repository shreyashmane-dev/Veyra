# Start VEYRA

## Linux/macOS

```bash
cd veyra
chmod +x install.sh
./install.sh
.venv/bin/veyra
```

To make `veyra` available everywhere, add the `.venv/bin` directory to your PATH.

## Windows PowerShell

```powershell
cd veyra
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
veyra
```

## First commands

```text
/status
teach CN is Computer Networks
/memory
```

## Training direction

The project will evolve toward a self-trained VEYRA-LM. Do not put arbitrary Kaggle data directly into training; check its license/terms, clean it, deduplicate it, and create train/validation/test splits first.
