#!/bin/bash

echo "============================================================"
echo "   Stock Filter Optimizer - GPU Setup Script"
echo "============================================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for Python
echo "[1/6] Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Python found${NC}"

# Check for NVIDIA GPU
echo ""
echo "[2/6] Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}[WARNING] NVIDIA GPU not detected or drivers not installed${NC}"
    echo "GPU acceleration will not be available"
    echo "Install NVIDIA drivers from: https://www.nvidia.com/drivers"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}[OK] NVIDIA GPU detected${NC}"
fi

# Setup virtual environment
echo ""
echo "[3/6] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}[OK] Virtual environment ready${NC}"

# Install base requirements
echo ""
echo "[4/6] Installing base requirements..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo -e "${GREEN}[OK] Base packages installed${NC}"

# Install PyTorch with CUDA
echo ""
echo "[5/6] Installing PyTorch with CUDA support..."
echo ""
echo "Select your CUDA version:"
echo "  1. CUDA 11.8 (recommended for most systems)"
echo "  2. CUDA 12.1 (newer GPUs)"
echo "  3. CPU only (no GPU)"
echo ""
read -p "Enter choice (1-3): " CUDA_CHOICE

case $CUDA_CHOICE in
    1)
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    *)
        echo "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio
        ;;
esac

# Install additional GPU packages
echo ""
echo "[6/6] Installing additional GPU packages..."
pip install pynvml

# Verify installation
echo ""
echo "============================================================"
echo "   Verifying GPU Setup"
echo "============================================================"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "============================================================"
echo -e "${GREEN}   GPU Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "To train with GPU acceleration:"
echo "  ./run.sh train-gpu"
echo ""
echo "Or with custom parameters:"
echo "  ./run.sh train-gpu --xgb-trees 2000 --lgb-trees 2000 --lstm-epochs 100"
echo ""
