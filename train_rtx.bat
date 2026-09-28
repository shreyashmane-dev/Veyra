@echo off
echo ======================================================================
echo   ZONARETH-LM: NVIDIA RTX 3050 / 4050 GPU TRAINING LAUNCHER
echo ======================================================================
echo.

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT FOUND!'" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] CUDA is not active in your Python environment!
    echo Please install PyTorch with CUDA using:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU Detected!
python -c "import torch; print('  Device:', torch.cuda.get_device_name(0)); print('  VRAM  :', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1), 'GB')"
echo.

echo Installing local project in editable mode...
pip install -e . -q
pip install datasets tqdm pyyaml -q

echo.
echo Launching ZONARETH-125M Training (Optimal for 4GB/6GB VRAM)...
python scripts/run_kaggle_pipeline.py --model 125m --steps 2500 --batch-size 4 --save-interval 250 --max-to-keep 2

echo.
echo ======================================================================
echo   TRAINING COMPLETE!
echo   Your model archive is ready: zonareth_125m_model.zip (~250 MB)
echo ======================================================================
pause
