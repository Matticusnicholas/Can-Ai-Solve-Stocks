"""
Filter Criteria Optimizer
Converts ML feature importances into actionable stock screener criteria
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
import json


@dataclass
class FilterCriterion:
    """Single filter criterion with threshold"""
    feature: str
    operator: str  # '<', '>', '<=', '>=', 'between'
    threshold: float
    threshold_upper: Optional[float] = None  # For 'between' operator
    importance: float = 0.0
    precision_lift: float = 0.0  # How much this filter improves precision
    recall_impact: float = 0.0  # Impact on recall

    def to_dict(self) -> dict:
        return {
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'threshold_upper': self.threshold_upper,
            'importance': self.importance,
            'precision_lift': self.precision_lift,
            'recall_impact': self.recall_impact
        }

    def to_string(self) -> str:
        if self.operator == 'between':
            return f"{self.threshold:.4f} < {self.feature} < {self.threshold_upper:.4f}"
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


class FilterOptimizer:
    """
    Optimizes filter criteria based on ML model insights

    Uses multiple strategies:
    1. Feature importance ranking from ensemble
    2. Decision tree splits for optimal thresholds
    3. Quantile analysis for breakout vs non-breakout
    4. Greedy search for best filter combinations
    """

    def __init__(self):
        self.criteria: List[FilterCriterion] = []
        self.feature_stats: Dict[str, Dict] = {}
        self.optimization_results: Dict[str, Any] = {}

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importance: pd.DataFrame,
        top_n_features: int = 30,
        min_precision: float = 0.4,
        target_recall: float = 0.3
    ) -> List[FilterCriterion]:
        """
        Find optimal filter criteria

        Args:
            X: Feature matrix
            y: Target labels (breakout/no breakout)
            feature_importance: Feature importance DataFrame
            top_n_features: Number of top features to consider
            min_precision: Minimum precision for filter criteria
            target_recall: Target recall to maintain

        Returns:
            List of FilterCriterion objects
        """
        print("Optimizing filter criteria...")

        # Get top features
        top_features = feature_importance.head(top_n_features)['feature'].tolist()

        # Analyze each feature
        for feature in top_features:
            if feature not in X.columns:
                continue

            self._analyze_feature(X[feature], y, feature)

        # Find optimal thresholds using decision trees
        self._find_tree_thresholds(X[top_features], y)

        # Find optimal thresholds using quantile analysis
        self._find_quantile_thresholds(X[top_features], y)

        # Evaluate and rank criteria
        self._evaluate_criteria(X, y)

        # Select best criteria meeting precision/recall targets
        self._select_best_criteria(X, y, min_precision, target_recall)

        # Add importance scores
        for criterion in self.criteria:
            imp_row = feature_importance[feature_importance['feature'] == criterion.feature]
            if not imp_row.empty:
                criterion.importance = imp_row['importance'].values[0]

        # Sort by importance
        self.criteria.sort(key=lambda x: x.importance, reverse=True)

        return self.criteria

    def _analyze_feature(
        self,
        feature_values: pd.Series,
        labels: pd.Series,
        feature_name: str
    ):
        """Analyze a single feature's distribution for breakouts vs non-breakouts"""
        breakout_values = feature_values[labels == 1]
        non_breakout_values = feature_values[labels == 0]

        self.feature_stats[feature_name] = {
            'breakout_mean': breakout_values.mean(),
            'breakout_median': breakout_values.median(),
            'breakout_std': breakout_values.std(),
            'breakout_q25': breakout_values.quantile(0.25),
            'breakout_q75': breakout_values.quantile(0.75),
            'non_breakout_mean': non_breakout_values.mean(),
            'non_breakout_median': non_breakout_values.median(),
            'non_breakout_std': non_breakout_values.std(),
            'non_breakout_q25': non_breakout_values.quantile(0.25),
            'non_breakout_q75': non_breakout_values.quantile(0.75),
            'mean_diff': breakout_values.mean() - non_breakout_values.mean(),
            'median_diff': breakout_values.median() - non_breakout_values.median()
        }

    def _find_tree_thresholds(self, X: pd.DataFrame, y: pd.Series):
        """Use decision tree to find optimal split points"""

        # Train shallow decision tree
        tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100)
        tree.fit(X.fillna(0), y)

        # Extract split thresholds
        feature_names = X.columns.tolist()
        tree_features = tree.tree_.feature
        tree_thresholds = tree.tree_.threshold
        tree_impurity = tree.tree_.impurity

        for i, (feat_idx, threshold) in enumerate(zip(tree_features, tree_thresholds)):
            if feat_idx >= 0:  # Valid feature
                feature_name = feature_names[feat_idx]

                # Determine operator based on tree structure
                # Check which direction leads to higher breakout probability
                left_mask = X[feature_name] <= threshold
                right_mask = X[feature_name] > threshold

                left_rate = y[left_mask].mean() if left_mask.sum() > 0 else 0
                right_rate = y[right_mask].mean() if right_mask.sum() > 0 else 0

                if right_rate > left_rate:
                    operator = '>'
                else:
                    operator = '<='

                criterion = FilterCriterion(
                    feature=feature_name,
                    operator=operator,
                    threshold=float(threshold)
                )
                self.criteria.append(criterion)

    def _find_quantile_thresholds(self, X: pd.DataFrame, y: pd.Series):
        """Find thresholds using quantile analysis"""

        for feature in X.columns:
            stats = self.feature_stats.get(feature, {})

            # Skip if no stats
            if not stats:
                continue

            # If breakouts have higher values, use lower quantile as threshold
            if stats.get('mean_diff', 0) > 0:
                threshold = X[feature][y == 1].quantile(0.25)
                operator = '>'
            else:
                threshold = X[feature][y == 1].quantile(0.75)
                operator = '<'

            if pd.notna(threshold):
                criterion = FilterCriterion(
                    feature=feature,
                    operator=operator,
                    threshold=float(threshold)
                )
                # Check if similar criterion already exists
                exists = any(
                    c.feature == feature and c.operator == operator
                    for c in self.criteria
                )
                if not exists:
                    self.criteria.append(criterion)

    def _evaluate_criteria(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate precision lift and recall impact for each criterion"""

        base_rate = y.mean()

        for criterion in self.criteria:
            feature = criterion.feature
            if feature not in X.columns:
                continue

            # Apply filter
            if criterion.operator == '>':
                mask = X[feature] > criterion.threshold
            elif criterion.operator == '>=':
                mask = X[feature] >= criterion.threshold
            elif criterion.operator == '<':
                mask = X[feature] < criterion.threshold
            elif criterion.operator == '<=':
                mask = X[feature] <= criterion.threshold
            elif criterion.operator == 'between':
                mask = (X[feature] > criterion.threshold) & (X[feature] < criterion.threshold_upper)
            else:
                continue

            # Handle NaN
            mask = mask.fillna(False)

            if mask.sum() > 0:
                filtered_rate = y[mask].mean()
                criterion.precision_lift = (filtered_rate - base_rate) / base_rate if base_rate > 0 else 0
                criterion.recall_impact = mask.sum() / len(mask)  # Coverage

    def _select_best_criteria(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        min_precision: float,
        target_recall: float
    ):
        """Select the best combination of criteria"""

        # Filter criteria with positive precision lift
        positive_criteria = [c for c in self.criteria if c.precision_lift > 0]

        # Sort by precision lift
        positive_criteria.sort(key=lambda x: x.precision_lift, reverse=True)

        # Greedy selection
        selected = []
        current_mask = pd.Series([True] * len(X), index=X.index)
        base_rate = y.mean()

        for criterion in positive_criteria:
            feature = criterion.feature
            if feature not in X.columns:
                continue

            # Apply this criterion
            if criterion.operator == '>':
                new_mask = current_mask & (X[feature] > criterion.threshold)
            elif criterion.operator == '>=':
                new_mask = current_mask & (X[feature] >= criterion.threshold)
            elif criterion.operator == '<':
                new_mask = current_mask & (X[feature] < criterion.threshold)
            elif criterion.operator == '<=':
                new_mask = current_mask & (X[feature] <= criterion.threshold)
            else:
                continue

            new_mask = new_mask.fillna(False)

            if new_mask.sum() < 100:  # Minimum sample size
                continue

            # Check if this improves precision while maintaining recall
            new_precision = y[new_mask].mean() if new_mask.sum() > 0 else 0
            new_recall = (y[new_mask].sum() / y.sum()) if y.sum() > 0 else 0

            if new_precision >= min_precision and new_recall >= target_recall * 0.5:
                selected.append(criterion)
                current_mask = new_mask

                if len(selected) >= 10:  # Limit number of criteria
                    break

        self.criteria = selected

    def get_filter_rules(self) -> List[str]:
        """Get human-readable filter rules"""
        return [c.to_string() for c in self.criteria]

    def apply_filters(
        self,
        X: pd.DataFrame,
        criteria: Optional[List[FilterCriterion]] = None
    ) -> pd.Series:
        """Apply filter criteria to data and return mask"""

        if criteria is None:
            criteria = self.criteria

        mask = pd.Series([True] * len(X), index=X.index)

        for criterion in criteria:
            feature = criterion.feature
            if feature not in X.columns:
                continue

            if criterion.operator == '>':
                mask = mask & (X[feature] > criterion.threshold)
            elif criterion.operator == '>=':
                mask = mask & (X[feature] >= criterion.threshold)
            elif criterion.operator == '<':
                mask = mask & (X[feature] < criterion.threshold)
            elif criterion.operator == '<=':
                mask = mask & (X[feature] <= criterion.threshold)
            elif criterion.operator == 'between':
                mask = mask & (X[feature] > criterion.threshold) & (X[feature] < criterion.threshold_upper)

        return mask.fillna(False)

    def evaluate_filter_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate the combined filter performance"""

        mask = self.apply_filters(X)

        total_samples = len(X)
        filtered_samples = mask.sum()
        base_breakout_rate = y.mean()
        filtered_breakout_rate = y[mask].mean() if filtered_samples > 0 else 0

        total_breakouts = y.sum()
        captured_breakouts = y[mask].sum()

        return {
            'total_samples': int(total_samples),
            'filtered_samples': int(filtered_samples),
            'filter_pass_rate': filtered_samples / total_samples if total_samples > 0 else 0,
            'base_breakout_rate': float(base_breakout_rate),
            'filtered_breakout_rate': float(filtered_breakout_rate),
            'precision_lift': (filtered_breakout_rate - base_breakout_rate) / base_breakout_rate if base_breakout_rate > 0 else 0,
            'total_breakouts': int(total_breakouts),
            'captured_breakouts': int(captured_breakouts),
            'recall': captured_breakouts / total_breakouts if total_breakouts > 0 else 0,
            'num_criteria': len(self.criteria)
        }

    def to_json(self) -> str:
        """Export criteria to JSON"""
        return json.dumps({
            'criteria': [c.to_dict() for c in self.criteria],
            'feature_stats': self.feature_stats,
            'optimization_results': self.optimization_results
        }, indent=2, default=str)

    def save(self, path: str):
        """Save criteria to file"""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'FilterOptimizer':
        """Load criteria from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        optimizer = cls()
        optimizer.criteria = [
            FilterCriterion(**c) for c in data.get('criteria', [])
        ]
        optimizer.feature_stats = data.get('feature_stats', {})
        optimizer.optimization_results = data.get('optimization_results', {})

        return optimizer

    def get_screener_config(self) -> Dict[str, Any]:
        """
        Generate a configuration that can be used with stock screeners

        Returns dict with:
        - readable_rules: Human-readable filter rules
        - technical_criteria: Technical indicator thresholds
        - fundamental_criteria: Fundamental metric thresholds
        """
        technical_keywords = [
            'rsi', 'macd', 'sma', 'ema', 'bb_', 'atr', 'adx', 'cci',
            'stoch', 'volume', 'obv', 'momentum', 'roc', 'williams',
            'price_vs', 'return_', 'volatility', 'breakout', 'squeeze'
        ]

        fundamental_keywords = [
            'fund_', 'pe_ratio', 'market_cap', 'profit_margin',
            'revenue', 'earnings', 'debt', 'cash_flow', 'beta'
        ]

        technical_criteria = []
        fundamental_criteria = []
        other_criteria = []

        for criterion in self.criteria:
            rule = criterion.to_string()

            is_technical = any(kw in criterion.feature.lower() for kw in technical_keywords)
            is_fundamental = any(kw in criterion.feature.lower() for kw in fundamental_keywords)

            if is_technical:
                technical_criteria.append({
                    'rule': rule,
                    'feature': criterion.feature,
                    'operator': criterion.operator,
                    'value': criterion.threshold,
                    'importance': criterion.importance
                })
            elif is_fundamental:
                fundamental_criteria.append({
                    'rule': rule,
                    'feature': criterion.feature,
                    'operator': criterion.operator,
                    'value': criterion.threshold,
                    'importance': criterion.importance
                })
            else:
                other_criteria.append({
                    'rule': rule,
                    'feature': criterion.feature,
                    'operator': criterion.operator,
                    'value': criterion.threshold,
                    'importance': criterion.importance
                })

        return {
            'readable_rules': self.get_filter_rules(),
            'technical_criteria': technical_criteria,
            'fundamental_criteria': fundamental_criteria,
            'other_criteria': other_criteria,
            'total_criteria': len(self.criteria)
        }


def optimize_filters(
    X: pd.DataFrame,
    y: pd.Series,
    feature_importance: pd.DataFrame,
    save_path: str = None
) -> FilterOptimizer:
    """
    Convenience function to run filter optimization
    """
    optimizer = FilterOptimizer()
    optimizer.optimize(X, y, feature_importance)

    if save_path:
        optimizer.save(save_path)

    return optimizer
