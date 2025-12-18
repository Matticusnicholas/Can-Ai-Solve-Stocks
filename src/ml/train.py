"""
Training pipeline for stock breakout prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tqdm import tqdm

from .models import EnsembleModel
from .labeler import BreakoutLabeler
from ..features.engineer import FeatureEngineer
from ..data.collector import DataCollector


class ModelTrainer:
    """
    End-to-end training pipeline for breakout prediction
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

    def prepare_data(
        self,
        breakout_threshold: float = 0.20,
        lookforward_days: int = 63,
        test_size: float = 0.2,
        min_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training

        Args:
            breakout_threshold: Threshold for breakout labeling
            lookforward_days: Days to look forward for breakout
            test_size: Fraction of data for testing
            min_date: Minimum date to include (for faster testing)

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Load data
        collector = DataCollector(str(self.data_dir))
        ohlcv_df, fundamentals_df = collector.load_data()

        if ohlcv_df is None:
            raise ValueError("No OHLCV data found. Run data collection first.")

        print(f"Loaded {len(ohlcv_df):,} rows of OHLCV data")

        # Filter by date if specified
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

        # Get breakout stats
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

        # Time-based split (to avoid data leakage)
        # Sort by date and split
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
        print(f"Train breakout rate: {y_train.mean():.2%}")
        print(f"Test breakout rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_weights: Optional[Dict[str, float]] = None
    ) -> EnsembleModel:
        """
        Train the ensemble model
        """
        # Create validation split if not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )

        print(f"\nTraining on {len(X_train):,} samples")
        print(f"Validating on {len(X_val):,} samples")

        # Initialize and train ensemble
        self.ensemble = EnsembleModel(weights=model_weights)
        self.ensemble.fit(X_train, y_train, X_val, y_val, verbose=True)

        return self.ensemble

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model performance
        """
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")

        metrics = self.ensemble.evaluate(X_test, y_test, threshold=threshold)

        print(f"\nEnsemble Performance (threshold={threshold}):")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

        return metrics

    def find_optimal_threshold(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        min_precision: float = 0.5
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find the optimal probability threshold for predictions

        Args:
            X_test: Test features
            y_test: Test labels
            min_precision: Minimum precision required

        Returns:
            Optimal threshold and metrics at that threshold
        """
        y_proba = self.ensemble.predict_proba(X_test)

        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}

        for threshold in np.arange(0.3, 0.8, 0.05):
            y_pred = (y_proba >= threshold).astype(int)

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            if precision >= min_precision and f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': threshold
                }

        print(f"\nOptimal threshold: {best_threshold}")
        print(f"  Precision: {best_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {best_metrics.get('recall', 0):.4f}")
        print(f"  F1: {best_metrics.get('f1', 0):.4f}")

        return best_threshold, best_metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc_roc': []
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{n_splits}")

            X_train_cv = X.iloc[train_idx]
            X_test_cv = X.iloc[test_idx]
            y_train_cv = y.iloc[train_idx]
            y_test_cv = y.iloc[test_idx]

            # Train
            model = EnsembleModel()
            model.fit(X_train_cv, y_train_cv, verbose=False)

            # Evaluate
            metrics = model.evaluate(X_test_cv, y_test_cv)

            for key in cv_results:
                cv_results[key].append(metrics[key])

        # Print summary
        print("\nCross-Validation Results:")
        for metric, values in cv_results.items():
            print(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

        return cv_results

    def save_model(self, filename: str = None):
        """Save trained model"""
        if self.ensemble is None:
            raise ValueError("No model to save. Train first.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_model_{timestamp}.joblib"

        model_path = self.models_dir / filename
        self.ensemble.save(str(model_path))
        print(f"Model saved to {model_path}")

        # Save feature columns
        feature_path = self.models_dir / filename.replace('.joblib', '_features.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f)

        return model_path

    def load_model(self, filename: str) -> EnsembleModel:
        """Load a trained model"""
        model_path = self.models_dir / filename
        self.ensemble = EnsembleModel.load(str(model_path))

        # Load feature columns
        feature_path = self.models_dir / filename.replace('.joblib', '_features.json')
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_columns = json.load(f)

        return self.ensemble

    def save_results(
        self,
        metrics: Dict[str, Any],
        feature_importance: pd.DataFrame,
        filename: str = None
    ):
        """Save training results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}"

        # Save metrics
        metrics_path = self.results_dir / f"{filename}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, np.ndarray):
                    clean_metrics[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    clean_metrics[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    clean_metrics[k] = int(v)
                else:
                    clean_metrics[k] = v
            json.dump(clean_metrics, f, indent=2)

        # Save feature importance
        importance_path = self.results_dir / f"{filename}_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)

        print(f"Results saved to {self.results_dir}")


def run_full_pipeline(
    breakout_threshold: float = 0.20,
    lookforward_days: int = 63
):
    """
    Run the complete training pipeline
    """
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        breakout_threshold=breakout_threshold,
        lookforward_days=lookforward_days
    )

    # Train
    trainer.train(X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    # Find optimal threshold
    opt_threshold, _ = trainer.find_optimal_threshold(X_test, y_test)

    # Get feature importance
    importance = trainer.ensemble.get_feature_importance()

    # Save everything
    trainer.save_model()
    trainer.save_results(metrics, importance)

    return trainer, metrics, importance


if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score

    trainer, metrics, importance = run_full_pipeline()

    print("\n" + "="*60)
    print("TOP 20 MOST IMPORTANT FEATURES FOR BREAKOUT PREDICTION")
    print("="*60)
    print(importance.head(20).to_string(index=False))
