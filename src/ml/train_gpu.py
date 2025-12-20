"""
GPU-Accelerated Training Pipeline for Stock Breakout Prediction
Utilizes CUDA for XGBoost, LightGBM, and PyTorch LSTM
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json
import time
from sklearn.model_selection import train_test_split

from .models_gpu import EnsembleGPU, check_gpu_status, GPU_AVAILABLE
from .labeler import BreakoutLabeler
from .optimizer import FilterOptimizer
from ..features.engineer import FeatureEngineer
from ..data.collector import DataCollector


class GPUModelTrainer:
    """
    GPU-Accelerated training pipeline for breakout prediction
    Maximizes GPU utilization for fast training
    """

    def __init__(
        self,
        data_dir: str = None,
        models_dir: str = None,
        results_dir: str = None
    ):
        base_path = Path(__file__).parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else base_path / "data"
        self.models_dir = Path(models_dir) if models_dir else base_path / "models"
        self.results_dir = Path(results_dir) if results_dir else base_path / "results"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.ensemble = None
        self.feature_columns = []

        # Check GPU status on init
        check_gpu_status()

    def prepare_data(
        self,
        breakout_threshold: float = 0.20,
        lookforward_days: int = 63,
        test_size: float = 0.2,
        min_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare data for training
        """
        collector = DataCollector(str(self.data_dir))
        ohlcv_df, fundamentals_df = collector.load_data()

        if ohlcv_df is None:
            raise ValueError("No OHLCV data found. Run data collection first.")

        print(f"Loaded {len(ohlcv_df):,} rows of OHLCV data")

        if min_date:
            ohlcv_df = ohlcv_df[ohlcv_df['date'] >= min_date]
            print(f"Filtered to {len(ohlcv_df):,} rows after {min_date}")

        # Engineer features
        print("\nEngineering features...")
        featured_df = self.feature_engineer.engineer_features(ohlcv_df, fundamentals_df)

        # Label breakouts
        print("\nLabeling breakouts...")
        labeler = BreakoutLabeler(
            breakout_threshold=breakout_threshold,
            lookforward_days=lookforward_days
        )
        labeled_df = labeler.label_breakouts(featured_df)

        stats = labeler.get_breakout_stats(labeled_df)
        print(f"\nBreakout Statistics:")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Breakout count: {stats['breakout_count']:,}")
        print(f"  Breakout rate: {stats['breakout_rate']:.2%}")

        # Save processed data
        labeled_df.to_parquet(self.data_dir / "sp500_processed.parquet")

        # Prepare training data
        X, y, feature_cols = self.feature_engineer.prepare_training_data(
            labeled_df,
            target_col='breakout'
        )
        self.feature_columns = feature_cols

        print(f"\nPrepared {len(X):,} samples with {len(feature_cols)} features")

        # Time-based split
        dates = labeled_df.loc[X.index, 'date']
        split_idx = int(len(X) * (1 - test_size))
        sorted_indices = dates.sort_values().index

        train_idx = sorted_indices[:split_idx]
        test_idx = sorted_indices[split_idx:]

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        print(f"\nTrain set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_lstm: bool = True,
        xgb_estimators: int = 1000,
        lgb_estimators: int = 1000,
        lstm_epochs: int = 50
    ) -> EnsembleGPU:
        """
        Train GPU-accelerated ensemble model
        """
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        print(f"\n{'='*60}")
        print("GPU-ACCELERATED TRAINING")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {len(X_train.columns)}")
        print(f"GPU Available: {GPU_AVAILABLE}")

        # Configure models for maximum GPU utilization
        xgb_params = {
            'n_estimators': xgb_estimators,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        lgb_params = {
            'n_estimators': lgb_estimators,
            'num_leaves': 127,
            'max_depth': -1,
            'learning_rate': 0.05,
        }

        lstm_params = {
            'hidden_size': 256,
            'num_layers': 3,
            'dropout': 0.3,
            'batch_size': 512,  # Larger batch for GPU
            'epochs': lstm_epochs,
            'learning_rate': 0.001,
        }

        # Initialize and train ensemble
        self.ensemble = EnsembleGPU(
            use_lstm=use_lstm and GPU_AVAILABLE,
            xgb_params=xgb_params,
            lgb_params=lgb_params,
            lstm_params=lstm_params
        )

        start_time = time.time()
        self.ensemble.fit(X_train, y_train, X_val, y_val, verbose=True)
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Total training time: {total_time:.1f}s")
        print(f"{'='*60}")

        return self.ensemble

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate model performance"""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")

        metrics = self.ensemble.evaluate(X_test, y_test, threshold=threshold)

        print(f"\n{'='*60}")
        print("GPU ENSEMBLE PERFORMANCE")
        print(f"{'='*60}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

        return metrics

    def optimize_filters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n_features: int = 30,
        min_precision: float = 0.4,
        target_recall: float = 0.3
    ) -> FilterOptimizer:
        """Run filter optimization"""
        print(f"\n{'='*60}")
        print("OPTIMIZING FILTER CRITERIA")
        print(f"{'='*60}")

        feature_importance = self.ensemble.get_feature_importance()

        optimizer = FilterOptimizer()
        optimizer.optimize(
            X, y, feature_importance,
            top_n_features=top_n_features,
            min_precision=min_precision,
            target_recall=target_recall
        )

        # Evaluate
        performance = optimizer.evaluate_filter_performance(X, y)

        print(f"\nOptimal Filter Criteria:")
        for rule in optimizer.get_filter_rules():
            print(f"  {rule}")

        print(f"\nFilter Performance:")
        print(f"  Base breakout rate:     {performance['base_breakout_rate']:.2%}")
        print(f"  Filtered breakout rate: {performance['filtered_breakout_rate']:.2%}")
        print(f"  Precision lift:         {performance['precision_lift']:.1%}")
        print(f"  Recall:                 {performance['recall']:.2%}")

        # Save
        optimizer.save(str(self.results_dir / "optimal_filters_gpu.json"))

        return optimizer

    def save_model(self, filename: str = None):
        """Save trained model"""
        if self.ensemble is None:
            raise ValueError("No model to save.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_gpu_{timestamp}.joblib"

        model_path = self.models_dir / filename
        self.ensemble.save(str(model_path))
        print(f"Model saved to {model_path}")

        # Save feature columns
        feature_path = self.models_dir / filename.replace('.joblib', '_features.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f)

        return model_path

    def save_results(
        self,
        metrics: Dict[str, Any],
        feature_importance: pd.DataFrame,
        filename: str = None
    ):
        """Save training results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_gpu_{timestamp}"

        # Save metrics
        metrics_path = self.results_dir / f"{filename}_metrics.json"
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                clean_metrics[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                clean_metrics[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                clean_metrics[k] = int(v)
            else:
                clean_metrics[k] = v

        with open(metrics_path, 'w') as f:
            json.dump(clean_metrics, f, indent=2)

        # Save feature importance
        importance_path = self.results_dir / f"{filename}_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)

        print(f"Results saved to {self.results_dir}")


def run_gpu_pipeline(
    breakout_threshold: float = 0.20,
    lookforward_days: int = 63,
    use_lstm: bool = True,
    xgb_estimators: int = 1000,
    lgb_estimators: int = 1000,
    lstm_epochs: int = 50
):
    """
    Run the complete GPU-accelerated training pipeline
    """
    print("\n" + "="*60)
    print("  STOCK FILTER OPTIMIZER - GPU ACCELERATED")
    print("="*60 + "\n")

    trainer = GPUModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        breakout_threshold=breakout_threshold,
        lookforward_days=lookforward_days
    )

    # Train with GPU
    trainer.train(
        X_train, y_train,
        use_lstm=use_lstm,
        xgb_estimators=xgb_estimators,
        lgb_estimators=lgb_estimators,
        lstm_epochs=lstm_epochs
    )

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    # Get feature importance
    importance = trainer.ensemble.get_feature_importance()

    # Optimize filters
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    trainer.optimize_filters(X_all, y_all)

    # Save
    trainer.save_model()
    trainer.save_results(metrics, importance)

    print("\n" + "="*60)
    print("  GPU TRAINING COMPLETE!")
    print("="*60 + "\n")

    return trainer, metrics, importance


if __name__ == "__main__":
    run_gpu_pipeline()
