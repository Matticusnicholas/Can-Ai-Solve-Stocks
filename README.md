# Stock Filter Optimizer

An AI-powered tool that analyzes S&P 500 historical data to discover the optimal stock screening criteria for predicting 3-month breakouts.

## Overview

Instead of filtering stocks directly, this tool:
1. **Collects** extensive historical data for all S&P 500 companies
2. **Engineers** 100+ technical and fundamental features
3. **Identifies** which stocks had breakouts (20%+ gains in 3 months)
4. **Trains** ensemble ML models (XGBoost, LightGBM, Random Forest)
5. **Extracts** the most predictive features and optimal thresholds
6. **Generates** human-readable filter criteria you can use in any stock screener

## Features

- **Comprehensive Data Collection**: 10+ years of daily OHLCV data for S&P 500
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, Volume patterns, etc.
- **Ensemble ML Pipeline**: XGBoost, LightGBM, Random Forest with hyperparameter optimization
- **Feature Importance Analysis**: Identify which metrics matter most for breakouts
- **Optimal Threshold Discovery**: Find the exact values for each filter criterion
- **Web Dashboard**: Visualize results and explore the optimal filter criteria

## Installation

```bash
# Clone the repository
git clone https://github.com/Matticusnicholas/Can-Ai-Solve-Stocks.git
cd Can-Ai-Solve-Stocks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Collect Data
```bash
python -m src.data.collector
```

### 2. Train Models & Find Optimal Filters
```bash
python -m src.ml.train
```

### 3. Launch Web Dashboard
```bash
python -m src.web.app
```

Then open http://localhost:8000 in your browser.

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── collector.py      # S&P 500 data collection
│   │   └── sp500_list.py     # S&P 500 constituent list
│   ├── features/
│   │   ├── technical.py      # Technical indicators
│   │   ├── fundamental.py    # Fundamental metrics
│   │   └── engineer.py       # Feature engineering pipeline
│   ├── ml/
│   │   ├── labeler.py        # Breakout labeling
│   │   ├── models.py         # ML model definitions
│   │   ├── train.py          # Training pipeline
│   │   └── optimizer.py      # Filter criteria optimizer
│   ├── backtest/
│   │   └── engine.py         # Backtesting engine
│   └── web/
│       ├── app.py            # FastAPI backend
│       └── frontend/         # React frontend
├── data/                     # Downloaded data (gitignored)
├── models/                   # Trained models (gitignored)
└── results/                  # Analysis results
```

## How It Works

### Breakout Definition
A "breakout" is defined as a stock that gains **20% or more** within the following **3 months (63 trading days)**. This threshold is configurable.

### Machine Learning Approach
We use an ensemble of gradient boosting models because:
- They excel at ranking feature importance
- They handle mixed feature types well
- They're robust to outliers common in financial data
- Feature importances translate directly to filter criteria

### Filter Optimization
The system doesn't just find important features - it finds the **optimal threshold values**:
- Uses decision tree splits and quantile analysis
- Validates thresholds via backtesting
- Outputs criteria like: "RSI < 35 AND Volume_Ratio > 1.5 AND BB_Width < 0.1"

## Sources & Research

This tool is built on research from:
- [MDPI Stock Prediction Review](https://www.mdpi.com/2673-9909/5/3/76)
- [Technical Indicators for ML](https://arxiv.org/html/2412.15448v1/)
- [XGBoost vs LSTM Comparison](https://netanel.io/posts/xgb_vs_lstm/)
- [Hybrid Gradient Boosting + LSTM](https://arxiv.org/html/2505.23084v1)

## License

MIT
