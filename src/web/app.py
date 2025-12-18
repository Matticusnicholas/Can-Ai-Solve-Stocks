"""
FastAPI Backend for Stock Filter Optimizer
"""
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio

from src.data.collector import DataCollector
from src.features.engineer import FeatureEngineer
from src.ml.labeler import BreakoutLabeler
from src.ml.models import EnsembleModel
from src.ml.train import ModelTrainer
from src.ml.optimizer import FilterOptimizer
from src.backtest.engine import BacktestEngine


app = FastAPI(
    title="Stock Filter Optimizer",
    description="AI-powered tool to find optimal stock screening criteria for 3-month breakouts",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    def __init__(self):
        self.data_loaded = False
        self.model_trained = False
        self.collector = DataCollector()
        self.trainer = None
        self.optimizer = None
        self.ohlcv_df = None
        self.features_df = None
        self.X = None
        self.y = None
        self.feature_importance = None
        self.backtest_results = None
        self.job_status = {}

state = AppState()


# Pydantic models
class TrainingConfig(BaseModel):
    breakout_threshold: float = 0.20
    lookforward_days: int = 63
    test_size: float = 0.2
    min_date: Optional[str] = None


class OptimizationConfig(BaseModel):
    top_n_features: int = 30
    min_precision: float = 0.4
    target_recall: float = 0.3


class BacktestConfig(BaseModel):
    period_months: int = 3


class FilterCriteria(BaseModel):
    feature: str
    operator: str
    threshold: float


# API Routes

@app.get("/")
async def root():
    """Serve the main dashboard"""
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")


@app.get("/api/status")
async def get_status():
    """Get current application status"""
    return {
        "data_loaded": state.data_loaded,
        "model_trained": state.model_trained,
        "num_tickers": state.ohlcv_df['ticker'].nunique() if state.ohlcv_df is not None else 0,
        "num_samples": len(state.X) if state.X is not None else 0,
        "num_features": len(state.X.columns) if state.X is not None else 0,
        "job_status": state.job_status
    }


@app.post("/api/collect-data")
async def collect_data(background_tasks: BackgroundTasks):
    """Start data collection for S&P 500"""
    job_id = f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state.job_status[job_id] = {"status": "running", "progress": 0, "message": "Starting data collection..."}

    async def run_collection():
        try:
            state.job_status[job_id]["message"] = "Collecting S&P 500 data..."
            state.ohlcv_df = state.collector.collect_all_sp500(period="10y")
            state.data_loaded = True
            state.job_status[job_id] = {"status": "completed", "progress": 100, "message": "Data collection complete"}
        except Exception as e:
            state.job_status[job_id] = {"status": "failed", "error": str(e)}

    background_tasks.add_task(asyncio.to_thread, lambda: asyncio.run(run_collection()))
    return {"job_id": job_id, "message": "Data collection started"}


@app.post("/api/load-data")
async def load_data():
    """Load previously collected data"""
    try:
        state.ohlcv_df, fundamentals_df = state.collector.load_data()
        if state.ohlcv_df is None:
            raise HTTPException(status_code=404, detail="No data found. Run data collection first.")
        state.data_loaded = True
        return {
            "message": "Data loaded successfully",
            "rows": len(state.ohlcv_df),
            "tickers": state.ohlcv_df['ticker'].nunique()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Train the ML model"""
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Data not loaded. Load data first.")

    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state.job_status[job_id] = {"status": "running", "progress": 0, "message": "Starting training..."}

    def run_training():
        try:
            state.trainer = ModelTrainer()

            # Prepare data
            state.job_status[job_id]["message"] = "Preparing data..."
            state.job_status[job_id]["progress"] = 10

            X_train, X_test, y_train, y_test = state.trainer.prepare_data(
                breakout_threshold=config.breakout_threshold,
                lookforward_days=config.lookforward_days,
                test_size=config.test_size,
                min_date=config.min_date
            )

            state.X = pd.concat([X_train, X_test])
            state.y = pd.concat([y_train, y_test])

            # Train
            state.job_status[job_id]["message"] = "Training models..."
            state.job_status[job_id]["progress"] = 30

            state.trainer.train(X_train, y_train)

            # Evaluate
            state.job_status[job_id]["message"] = "Evaluating..."
            state.job_status[job_id]["progress"] = 70

            metrics = state.trainer.evaluate(X_test, y_test)
            state.feature_importance = state.trainer.ensemble.get_feature_importance()

            # Save
            state.trainer.save_model()
            state.model_trained = True

            state.job_status[job_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Training complete",
                "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                          for k, v in metrics.items() if k != 'confusion_matrix' and k != 'classification_report'}
            }

        except Exception as e:
            state.job_status[job_id] = {"status": "failed", "error": str(e)}
            import traceback
            traceback.print_exc()

    background_tasks.add_task(run_training)
    return {"job_id": job_id, "message": "Training started"}


@app.post("/api/optimize")
async def optimize_filters(config: OptimizationConfig):
    """Optimize filter criteria"""
    if not state.model_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Train model first.")

    try:
        state.optimizer = FilterOptimizer()
        state.optimizer.optimize(
            state.X,
            state.y,
            state.feature_importance,
            top_n_features=config.top_n_features,
            min_precision=config.min_precision,
            target_recall=config.target_recall
        )

        # Evaluate
        performance = state.optimizer.evaluate_filter_performance(state.X, state.y)
        screener_config = state.optimizer.get_screener_config()

        # Save
        results_dir = Path(__file__).parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        state.optimizer.save(str(results_dir / "optimal_filters.json"))

        return {
            "message": "Optimization complete",
            "num_criteria": len(state.optimizer.criteria),
            "performance": performance,
            "screener_config": screener_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/filters")
async def get_filters():
    """Get current optimal filter criteria"""
    if state.optimizer is None:
        # Try to load from file
        results_path = Path(__file__).parent.parent.parent / "results" / "optimal_filters.json"
        if results_path.exists():
            state.optimizer = FilterOptimizer.load(str(results_path))
        else:
            raise HTTPException(status_code=404, detail="No filters optimized yet")

    return {
        "criteria": [c.to_dict() for c in state.optimizer.criteria],
        "rules": state.optimizer.get_filter_rules(),
        "screener_config": state.optimizer.get_screener_config()
    }


@app.get("/api/feature-importance")
async def get_feature_importance(top_n: int = 50):
    """Get feature importance rankings"""
    if state.feature_importance is None:
        raise HTTPException(status_code=404, detail="No feature importance available. Train model first.")

    return {
        "features": state.feature_importance.head(top_n).to_dict(orient='records')
    }


@app.post("/api/backtest")
async def run_backtest(config: BacktestConfig):
    """Run backtest on historical data"""
    if state.optimizer is None or state.X is None:
        raise HTTPException(status_code=400, detail="Optimize filters first")

    try:
        # Load processed data
        data_path = Path(__file__).parent.parent.parent / "data" / "sp500_processed.parquet"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Processed data not found")

        df = pd.read_parquet(data_path)

        # Apply filters
        filter_mask = state.optimizer.apply_filters(state.X)

        # Align mask with dataframe
        aligned_mask = pd.Series(False, index=df.index)
        common_idx = df.index.intersection(filter_mask.index)
        aligned_mask.loc[common_idx] = filter_mask.loc[common_idx]

        # Run backtest
        engine = BacktestEngine()
        results = engine.run_backtest(df, aligned_mask, period_months=config.period_months)

        state.backtest_results = results
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest-results")
async def get_backtest_results():
    """Get backtest results"""
    if state.backtest_results is None:
        raise HTTPException(status_code=404, detail="No backtest results. Run backtest first.")
    return state.backtest_results


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a background job"""
    if job_id not in state.job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.job_status[job_id]


@app.get("/api/data-stats")
async def get_data_stats():
    """Get statistics about loaded data"""
    if state.ohlcv_df is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    stats = state.collector.get_data_stats(state.ohlcv_df)
    return stats


@app.post("/api/predict")
async def predict_breakout(features: Dict[str, float]):
    """Predict breakout probability for given features"""
    if not state.model_trained or state.trainer is None:
        raise HTTPException(status_code=400, detail="Model not trained")

    try:
        # Create DataFrame from features
        X = pd.DataFrame([features])

        # Get prediction
        proba = state.trainer.ensemble.predict_proba(X)[0]

        # Check filters
        passes_filter = False
        if state.optimizer:
            filter_mask = state.optimizer.apply_filters(X)
            passes_filter = bool(filter_mask.iloc[0])

        return {
            "breakout_probability": float(proba),
            "passes_filter": passes_filter
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files (frontend)
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


def main():
    """Run the web server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
