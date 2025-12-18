#!/bin/bash

echo "============================================================"
echo "   Stock Filter Optimizer - Setup Script"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Python
echo "[1/5] Checking for Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python3 is not installed${NC}"
    echo "Please install Python 3.9+ using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  macOS: brew install python3"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}[OK] Found Python $PYTHON_VERSION${NC}"

# Check Python version
MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo -e "${RED}[ERROR] Python 3.9+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

# Check for pip
echo ""
echo "[2/5] Checking for pip..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}[WARNING] pip3 not found, attempting to install...${NC}"
    python3 -m ensurepip --upgrade 2>/dev/null || {
        echo -e "${RED}[ERROR] Could not install pip${NC}"
        echo "Please install pip manually"
        exit 1
    }
fi
echo -e "${GREEN}[OK] pip is available${NC}"

# Create virtual environment
echo ""
echo "[3/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}[INFO] Virtual environment already exists${NC}"
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to create virtual environment${NC}"
        echo "Try: sudo apt install python3-venv (on Ubuntu/Debian)"
        exit 1
    fi
    echo -e "${GREEN}[OK] Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "[4/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
echo "This may take a few minutes..."
echo ""

pip install --upgrade pip > /dev/null 2>&1

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}[WARNING] Some packages may have failed to install${NC}"
    echo "Attempting to install core packages individually..."

    pip install pandas numpy
    pip install yfinance
    pip install scikit-learn
    pip install xgboost lightgbm
    pip install fastapi uvicorn
    pip install ta pandas-ta
    pip install matplotlib plotly seaborn
    pip install tqdm joblib
fi

echo ""
echo "============================================================"
echo -e "${GREEN}   Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "To run the application:"
echo "  1. Run './run.sh' to start the web server"
echo "  2. Open http://localhost:8000 in your browser"
echo ""
echo "Or use command line options:"
echo "  ./run.sh              - Start web server"
echo "  ./run.sh collect      - Collect stock data"
echo "  ./run.sh train        - Train ML models"
echo "  ./run.sh optimize     - Optimize filters"
echo "  ./run.sh all          - Run complete pipeline"
echo ""
