#!/bin/bash

echo "============================================================"
echo "   Stock Filter Optimizer"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}[ERROR] Virtual environment not found!${NC}"
    echo "Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to activate virtual environment${NC}"
    exit 1
fi

# Get command argument
COMMAND=${1:-serve}

case $COMMAND in
    serve)
        echo "Starting web server..."
        echo ""
        echo "============================================================"
        echo -e "${CYAN}   Web Dashboard Starting${NC}"
        echo "============================================================"
        echo ""
        echo -e "   Open your browser to: ${GREEN}http://localhost:8000${NC}"
        echo ""
        echo "   Press Ctrl+C to stop the server"
        echo "============================================================"
        echo ""
        python3 main.py serve
        ;;

    collect)
        echo "Collecting S&P 500 data..."
        echo "This may take 10-20 minutes depending on your internet connection."
        echo ""
        python3 main.py collect
        ;;

    train)
        echo "Training ML models (CPU)..."
        echo "This may take 5-15 minutes depending on your hardware."
        echo ""
        python3 main.py train
        ;;

    train-gpu)
        echo "Training ML models with GPU/CUDA acceleration..."
        echo "Using XGBoost GPU, LightGBM GPU, and PyTorch LSTM"
        echo ""
        python3 main.py train-gpu "${@:2}"
        ;;

    optimize)
        echo "Optimizing filter criteria..."
        echo ""
        python3 main.py optimize
        ;;

    backtest)
        echo "Running backtest..."
        echo ""
        python3 main.py backtest
        ;;

    all)
        echo "Running complete pipeline..."
        echo "This will take 30-60 minutes."
        echo ""
        python3 main.py all
        ;;

    help|--help|-h)
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  serve      Start web dashboard (default)"
        echo "  collect    Download S&P 500 historical data"
        echo "  train      Train ML models (CPU)"
        echo "  train-gpu  Train ML models with GPU/CUDA acceleration"
        echo "  optimize   Find optimal filter criteria"
        echo "  backtest   Validate filters on historical data"
        echo "  all        Run complete pipeline"
        echo "  help       Show this help message"
        echo ""
        echo "GPU Options (for train-gpu):"
        echo "  --xgb-trees N     XGBoost trees (default: 1000)"
        echo "  --lgb-trees N     LightGBM trees (default: 1000)"
        echo "  --lstm-epochs N   LSTM epochs (default: 50)"
        echo "  --no-lstm         Disable LSTM model"
        echo ""
        ;;

    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo "Run './run.sh help' for available commands"
        exit 1
        ;;
esac
