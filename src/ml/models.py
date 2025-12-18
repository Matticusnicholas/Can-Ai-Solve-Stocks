"""
Machine Learning Models for Stock Breakout Prediction
Ensemble of XGBoost, LightGBM, and Random Forest
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path


class BaseModel:
    """Base class for ML models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def get_feature_importance(self) -> pd.DataFrame:
        raise NotImplementedError


class XGBoostModel(BaseModel):
    """XGBoost classifier for breakout prediction"""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        scale_pos_weight: float = None,
        random_state: int = 42
    ):
        super().__init__("XGBoost")
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'n_jobs': -1,
            'eval_metric': 'auc',
            'use_label_encoder': False
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False
    ):
        self.feature_names = list(X.columns)

        # Handle class imbalance if not specified
        if self.params['scale_pos_weight'] is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            self.params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1

        self.model = xgb.XGBClassifier(**self.params)

        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = verbose

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class LightGBMModel(BaseModel):
    """LightGBM classifier for breakout prediction"""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        super().__init__("LightGBM")
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_samples': min_child_samples,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple]] = None,
        **kwargs
    ):
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = [lgb.log_evaluation(period=0)]  # Suppress output

        fit_params = {'callbacks': callbacks}
        if eval_set:
            fit_params['eval_set'] = eval_set

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class RandomForestModel(BaseModel):
    """Random Forest classifier for breakout prediction"""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 15,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        super().__init__("RandomForest")
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state,
            'n_jobs': -1
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        self.feature_names = list(X.columns)
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class EnsembleModel:
    """
    Ensemble model combining XGBoost, LightGBM, and Random Forest
    Uses weighted averaging of predictions
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_xgb: bool = True,
        use_lgb: bool = True,
        use_rf: bool = True
    ):
        """
        Args:
            weights: Dictionary of model weights for ensemble
            use_xgb: Include XGBoost in ensemble
            use_lgb: Include LightGBM in ensemble
            use_rf: Include Random Forest in ensemble
        """
        self.models = {}
        self.weights = weights or {'XGBoost': 0.4, 'LightGBM': 0.4, 'RandomForest': 0.2}
        self.feature_names = []
        self.metrics = {}

        if use_xgb:
            self.models['XGBoost'] = XGBoostModel()
        if use_lgb:
            self.models['LightGBM'] = LightGBMModel()
        if use_rf:
            self.models['RandomForest'] = RandomForestModel()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ):
        """
        Train all models in the ensemble
        """
        self.feature_names = list(X_train.columns)

        for name, model in self.models.items():
            if verbose:
                print(f"Training {name}...")

            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(X_train, y_train, eval_set=eval_set)

            # Calculate validation metrics
            if X_val is not None:
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)
                self.metrics[name] = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                    'auc_roc': roc_auc_score(y_val, y_proba)
                }

                if verbose:
                    print(f"  AUC-ROC: {self.metrics[name]['auc_roc']:.4f}")
                    print(f"  Precision: {self.metrics[name]['precision']:.4f}")
                    print(f"  Recall: {self.metrics[name]['recall']:.4f}")

        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict using ensemble voting"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from ensemble"""
        weighted_proba = np.zeros(len(X))
        total_weight = 0

        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            weighted_proba += model.predict_proba(X) * weight
            total_weight += weight

        return weighted_proba / total_weight

    def get_feature_importance(self, method: str = 'average') -> pd.DataFrame:
        """
        Get combined feature importance from all models

        Args:
            method: 'average' for weighted average, or model name for specific model
        """
        if method in self.models:
            return self.models[method].get_feature_importance()

        # Weighted average importance
        importance_dfs = []
        for name, model in self.models.items():
            imp_df = model.get_feature_importance()
            imp_df['weight'] = self.weights.get(name, 1.0)
            imp_df['weighted_importance'] = imp_df['importance'] * imp_df['weight']
            importance_dfs.append(imp_df[['feature', 'weighted_importance']])

        combined = pd.concat(importance_dfs)
        result = combined.groupby('feature')['weighted_importance'].sum().reset_index()
        result.columns = ['feature', 'importance']
        result['importance'] = result['importance'] / sum(self.weights.values())

        return result.sort_values('importance', ascending=False)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        """
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    def save(self, path: str):
        """Save ensemble model to disk"""
        save_dict = {
            'models': {name: model.model for name, model in self.models.items()},
            'weights': self.weights,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(save_dict, path)

    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load ensemble model from disk"""
        save_dict = joblib.load(path)

        ensemble = cls(weights=save_dict['weights'])
        ensemble.feature_names = save_dict['feature_names']
        ensemble.metrics = save_dict.get('metrics', {})

        for name, model in save_dict['models'].items():
            if name in ensemble.models:
                ensemble.models[name].model = model
                ensemble.models[name].feature_names = save_dict['feature_names']

        return ensemble

    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features"""
        importance_df = self.get_feature_importance()
        return importance_df.head(n)['feature'].tolist()
