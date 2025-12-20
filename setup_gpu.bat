@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo    Stock Filter Optimizer - GPU Setup Script
echo ============================================================
echo.

:: Check for Python
echo [1/6] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)
echo [OK] Python found

:: Check for NVIDIA GPU
echo.
echo [2/6] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA GPU not detected or drivers not installed
    echo GPU acceleration will not be available
    echo Install NVIDIA drivers from: https://www.nvidia.com/drivers
    pause
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
echo [OK] NVIDIA GPU detected

:: Check/Create virtual environment
echo.
echo [3/6] Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [OK] Virtual environment ready

:: Install base requirements
echo.
echo [4/6] Installing base requirements...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
echo [OK] Base packages installed

:: Install PyTorch with CUDA
echo.
echo [5/6] Installing PyTorch with CUDA support...
echo Detecting CUDA version...

:: Try to detect CUDA version
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do set DRIVER_VERSION=%%i

echo.
echo Select your CUDA version:
echo   1. CUDA 11.8 (recommended for most systems)
echo   2. CUDA 12.1 (newer GPUs)
echo   3. CPU only (no GPU)
echo.
set /p CUDA_CHOICE="Enter choice (1-3): "

if "%CUDA_CHOICE%"=="1" (
    echo Installing PyTorch for CUDA 11.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%CUDA_CHOICE%"=="2" (
    echo Installing PyTorch for CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing PyTorch CPU version...
    pip install torch torchvision torchaudio
)

:: Install additional GPU packages
echo.
echo [6/6] Installing additional GPU packages...
pip install pynvml

:: Verify installation
echo.
echo ============================================================
echo    Verifying GPU Setup
echo ============================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================================
echo    GPU Setup Complete!
echo ============================================================
echo.
echo To train with GPU acceleration:
echo   run.bat train-gpu
echo.
echo Or with custom parameters:
echo   run.bat train-gpu --xgb-trees 2000 --lgb-trees 2000 --lstm-epochs 100
echo.
pause
