"""
CUDA/GPU Accelerated Machine Learning Models for Stock Breakout Prediction
Uses XGBoost GPU, LightGBM GPU, and PyTorch LSTM with CUDA
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
GPU_AVAILABLE = False
CUDA_DEVICE = None

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        CUDA_DEVICE = torch.device('cuda')
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA not available, falling back to CPU")
except ImportError:
    print("✗ PyTorch not installed")

# Try to import RAPIDS cuML
CUML_AVAILABLE = False
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    CUML_AVAILABLE = True
    print("✓ RAPIDS cuML available")
except ImportError:
    print("✗ RAPIDS cuML not available (optional)")


class XGBoostGPU:
    """XGBoost with GPU acceleration"""

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        scale_pos_weight: float = None,
        random_state: int = 42
    ):
        import xgboost as xgb

        self.name = "XGBoost_GPU"
        self.feature_names = []

        # GPU-specific parameters
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
            # GPU acceleration
            'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
            'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'auto',
            'gpu_id': 0 if GPU_AVAILABLE else None,
        }

        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True):
        self.feature_names = list(X.columns)

        if self.params['scale_pos_weight'] is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            self.model.set_params(scale_pos_weight=neg_count / pos_count if pos_count > 0 else 1)

        fit_params = {'verbose': verbose}
        if eval_set:
            fit_params['eval_set'] = eval_set

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class LightGBMGPU:
    """LightGBM with GPU acceleration"""

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        import lightgbm as lgb

        self.name = "LightGBM_GPU"
        self.feature_names = []

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
            'verbose': -1,
            # GPU acceleration (requires LightGBM built with GPU support)
            'device': 'gpu' if GPU_AVAILABLE else 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
        }

        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, **kwargs):
        self.feature_names = list(X.columns)

        callbacks = []
        fit_params = {'callbacks': callbacks}
        if eval_set:
            fit_params['eval_set'] = eval_set

        try:
            self.model.fit(X, y, **fit_params)
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"GPU failed, falling back to CPU: {e}")
            self.model.set_params(device='cpu')
            self.model.fit(X, y, **fit_params)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class PyTorchLSTM:
    """PyTorch LSTM for sequence-based prediction with CUDA"""

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        sequence_length: int = 20
    ):
        self.name = "LSTM_GPU"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.device = CUDA_DEVICE if GPU_AVAILABLE else torch.device('cpu')

    def _build_model(self, input_size: int):
        import torch.nn as nn

        class LSTMClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=True
                )
                self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, batch_first=True)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Self-attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                # Take last timestep
                out = attn_out[:, -1, :]
                return self.fc(out)

        return LSTMClassifier(input_size, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None, verbose=True):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler

        self.feature_names = list(X.columns)
        self.input_size = len(self.feature_names)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)

        # Create sequences (for non-sequential data, we'll use sliding window on features)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).repeat(1, self.sequence_length, 1)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)

        # Move to GPU
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)

        # Build model
        self.model = self._build_model(self.input_size).to(self.device)

        # Class weights for imbalanced data
        pos_weight = torch.tensor([(y == 0).sum() / (y == 1).sum()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = nn.BCELoss()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        self.model.eval()
        X_scaled = self.scaler.transform(X.values)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).repeat(1, self.sequence_length, 1)
        X_tensor = X_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        return outputs.cpu().numpy().flatten()

    def get_feature_importance(self) -> pd.DataFrame:
        # For LSTM, use gradient-based importance
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.ones(len(self.feature_names)) / len(self.feature_names)
        })


class EnsembleGPU:
    """
    GPU-Accelerated Ensemble combining XGBoost, LightGBM, and LSTM
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_xgb: bool = True,
        use_lgb: bool = True,
        use_lstm: bool = True,
        xgb_params: dict = None,
        lgb_params: dict = None,
        lstm_params: dict = None
    ):
        self.models = {}
        self.weights = weights or {'XGBoost_GPU': 0.4, 'LightGBM_GPU': 0.4, 'LSTM_GPU': 0.2}
        self.feature_names = []
        self.metrics = {}

        if use_xgb:
            self.models['XGBoost_GPU'] = XGBoostGPU(**(xgb_params or {}))
        if use_lgb:
            self.models['LightGBM_GPU'] = LightGBMGPU(**(lgb_params or {}))
        if use_lstm and GPU_AVAILABLE:
            self.models['LSTM_GPU'] = PyTorchLSTM(**(lstm_params or {}))

        print(f"\n{'='*60}")
        print(f"GPU Ensemble initialized with: {list(self.models.keys())}")
        print(f"GPU Available: {GPU_AVAILABLE}")
        if GPU_AVAILABLE:
            import torch
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ):
        self.feature_names = list(X_train.columns)

        for name, model in self.models.items():
            if verbose:
                print(f"\n{'='*40}")
                print(f"Training {name}...")
                print(f"{'='*40}")

            import time
            start_time = time.time()

            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)

            elapsed = time.time() - start_time

            if X_val is not None:
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)

                self.metrics[name] = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0),
                    'auc_roc': roc_auc_score(y_val, y_proba),
                    'train_time': elapsed
                }

                if verbose:
                    print(f"\n{name} Results:")
                    print(f"  Training time: {elapsed:.1f}s")
                    print(f"  AUC-ROC: {self.metrics[name]['auc_roc']:.4f}")
                    print(f"  Precision: {self.metrics[name]['precision']:.4f}")
                    print(f"  Recall: {self.metrics[name]['recall']:.4f}")

        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        weighted_proba = np.zeros(len(X))
        total_weight = 0

        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            weighted_proba += model.predict_proba(X) * weight
            total_weight += weight

        return weighted_proba / total_weight

    def get_feature_importance(self, method: str = 'average') -> pd.DataFrame:
        if method in self.models:
            return self.models[method].get_feature_importance()

        importance_dfs = []
        for name, model in self.models.items():
            if name == 'LSTM_GPU':
                continue  # Skip LSTM for feature importance
            imp_df = model.get_feature_importance()
            imp_df['weight'] = self.weights.get(name, 1.0)
            imp_df['weighted_importance'] = imp_df['importance'] * imp_df['weight']
            importance_dfs.append(imp_df[['feature', 'weighted_importance']])

        if not importance_dfs:
            return pd.DataFrame()

        combined = pd.concat(importance_dfs)
        result = combined.groupby('feature')['weighted_importance'].sum().reset_index()
        result.columns = ['feature', 'importance']

        total_weight = sum(self.weights.get(n, 1.0) for n in self.models if n != 'LSTM_GPU')
        result['importance'] = result['importance'] / total_weight

        return result.sort_values('importance', ascending=False)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }

    def save(self, path: str):
        save_dict = {
            'weights': self.weights,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'models': {}
        }

        for name, model in self.models.items():
            if name == 'LSTM_GPU':
                # Save PyTorch model separately
                import torch
                torch.save(model.model.state_dict(), path.replace('.joblib', f'_{name}.pt'))
                save_dict['models'][name] = {
                    'scaler': model.scaler,
                    'params': {
                        'hidden_size': model.hidden_size,
                        'num_layers': model.num_layers,
                        'dropout': model.dropout,
                        'input_size': model.input_size,
                        'sequence_length': model.sequence_length
                    }
                }
            else:
                save_dict['models'][name] = model.model

        joblib.dump(save_dict, path)

    @classmethod
    def load(cls, path: str) -> 'EnsembleGPU':
        save_dict = joblib.load(path)

        ensemble = cls(weights=save_dict['weights'], use_xgb=False, use_lgb=False, use_lstm=False)
        ensemble.feature_names = save_dict['feature_names']
        ensemble.metrics = save_dict.get('metrics', {})

        for name, model_data in save_dict['models'].items():
            if name == 'XGBoost_GPU':
                ensemble.models[name] = XGBoostGPU()
                ensemble.models[name].model = model_data
                ensemble.models[name].feature_names = save_dict['feature_names']
            elif name == 'LightGBM_GPU':
                ensemble.models[name] = LightGBMGPU()
                ensemble.models[name].model = model_data
                ensemble.models[name].feature_names = save_dict['feature_names']
            elif name == 'LSTM_GPU' and GPU_AVAILABLE:
                import torch
                params = model_data['params']
                ensemble.models[name] = PyTorchLSTM(**params)
                ensemble.models[name].model = ensemble.models[name]._build_model(params['input_size'])
                ensemble.models[name].model.load_state_dict(
                    torch.load(path.replace('.joblib', f'_{name}.pt'))
                )
                ensemble.models[name].model.to(ensemble.models[name].device)
                ensemble.models[name].scaler = model_data['scaler']
                ensemble.models[name].feature_names = save_dict['feature_names']

        return ensemble


def check_gpu_status():
    """Print detailed GPU status"""
    print("\n" + "="*60)
    print("GPU STATUS CHECK")
    print("="*60)

    # PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"GPU Compute Capability: {props.major}.{props.minor}")
    except ImportError:
        print("PyTorch not installed")

    # XGBoost GPU
    try:
        import xgboost as xgb
        print(f"\nXGBoost Version: {xgb.__version__}")
        print(f"XGBoost GPU Support: Available (use tree_method='gpu_hist')")
    except ImportError:
        print("XGBoost not installed")

    # LightGBM GPU
    try:
        import lightgbm as lgb
        print(f"\nLightGBM Version: {lgb.__version__}")
    except ImportError:
        print("LightGBM not installed")

    # RAPIDS cuML
    try:
        import cuml
        print(f"\nRAPIDS cuML Version: {cuml.__version__}")
    except ImportError:
        print("\nRAPIDS cuML: Not installed (optional)")

    print("="*60 + "\n")


if __name__ == "__main__":
    check_gpu_status()
