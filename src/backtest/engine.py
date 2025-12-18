"""
Backtesting Engine
Validates filter criteria performance on historical data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from tqdm import tqdm


@dataclass
class BacktestResult:
    """Results from a single backtest period"""
    start_date: str
    end_date: str
    signals_generated: int
    breakouts_captured: int
    precision: float
    recall: float
    avg_return: float
    max_return: float
    min_return: float
    win_rate: float
    avg_days_to_peak: float


class BacktestEngine:
    """
    Backtests filter criteria on historical stock data

    Simulates applying the filter criteria at each point in time
    and tracks how well it would have predicted breakouts.
    """

    def __init__(
        self,
        breakout_threshold: float = 0.20,
        lookforward_days: int = 63,
        min_samples_per_period: int = 50
    ):
        """
        Args:
            breakout_threshold: Minimum gain to consider a breakout
            lookforward_days: Days to look forward for breakout
            min_samples_per_period: Minimum signals to evaluate a period
        """
        self.breakout_threshold = breakout_threshold
        self.lookforward_days = lookforward_days
        self.min_samples_per_period = min_samples_per_period
        self.results: List[BacktestResult] = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        filter_mask: pd.Series,
        date_col: str = 'date',
        period_months: int = 3,
        progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data

        Args:
            df: DataFrame with features and 'breakout', 'max_gain_forward' columns
            filter_mask: Boolean mask from filter criteria
            date_col: Date column name
            period_months: Length of each backtest period in months

        Returns:
            Dictionary with backtest summary statistics
        """
        df = df.copy()
        df['filter_signal'] = filter_mask

        # Ensure date is datetime
        df[date_col] = pd.to_datetime(df[date_col])

        # Get date range
        min_date = df[date_col].min()
        max_date = df[date_col].max()

        # Create backtest periods
        periods = []
        current_start = min_date
        while current_start < max_date - timedelta(days=self.lookforward_days):
            period_end = current_start + timedelta(days=period_months * 30)
            if period_end > max_date - timedelta(days=self.lookforward_days):
                period_end = max_date - timedelta(days=self.lookforward_days)
            periods.append((current_start, period_end))
            current_start = period_end

        self.results = []

        iterator = tqdm(periods, desc="Backtesting") if progress else periods

        for start_date, end_date in iterator:
            result = self._backtest_period(df, start_date, end_date, date_col)
            if result is not None:
                self.results.append(result)

        return self._summarize_results()

    def _backtest_period(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        date_col: str
    ) -> Optional[BacktestResult]:
        """Run backtest for a single period"""

        # Filter to period
        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
        period_df = df[mask]

        # Get signals in this period
        signals = period_df[period_df['filter_signal'] == True]

        if len(signals) < self.min_samples_per_period:
            return None

        # Calculate metrics
        signals_generated = len(signals)

        # Check if breakout column exists
        if 'breakout' in signals.columns:
            breakouts_captured = signals['breakout'].sum()
            precision = breakouts_captured / signals_generated if signals_generated > 0 else 0

            # Calculate recall (of breakouts in this period)
            total_breakouts = period_df['breakout'].sum()
            recall = breakouts_captured / total_breakouts if total_breakouts > 0 else 0
        else:
            breakouts_captured = 0
            precision = 0
            recall = 0

        # Return analysis
        if 'max_gain_forward' in signals.columns:
            returns = signals['max_gain_forward'].dropna()
            avg_return = returns.mean() if len(returns) > 0 else 0
            max_return = returns.max() if len(returns) > 0 else 0
            min_return = returns.min() if len(returns) > 0 else 0
            win_rate = (returns >= self.breakout_threshold).mean() if len(returns) > 0 else 0
        else:
            avg_return = max_return = min_return = win_rate = 0

        # Days to peak
        if 'days_to_breakout' in signals.columns:
            breakout_signals = signals[signals['breakout'] == 1]
            avg_days = breakout_signals['days_to_breakout'].mean() if len(breakout_signals) > 0 else 0
        else:
            avg_days = 0

        return BacktestResult(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            signals_generated=int(signals_generated),
            breakouts_captured=int(breakouts_captured),
            precision=float(precision),
            recall=float(recall),
            avg_return=float(avg_return),
            max_return=float(max_return),
            min_return=float(min_return),
            win_rate=float(win_rate),
            avg_days_to_peak=float(avg_days)
        )

    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize backtest results across all periods"""

        if not self.results:
            return {'error': 'No valid backtest periods'}

        precisions = [r.precision for r in self.results]
        recalls = [r.recall for r in self.results]
        returns = [r.avg_return for r in self.results]
        win_rates = [r.win_rate for r in self.results]
        signals = [r.signals_generated for r in self.results]

        return {
            'num_periods': len(self.results),
            'date_range': {
                'start': self.results[0].start_date,
                'end': self.results[-1].end_date
            },
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions)),
                'min': float(np.min(precisions)),
                'max': float(np.max(precisions))
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls)),
                'min': float(np.min(recalls)),
                'max': float(np.max(recalls))
            },
            'returns': {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns))
            },
            'win_rate': {
                'mean': float(np.mean(win_rates)),
                'std': float(np.std(win_rates)),
                'min': float(np.min(win_rates)),
                'max': float(np.max(win_rates))
            },
            'signals_per_period': {
                'mean': float(np.mean(signals)),
                'total': int(sum(signals))
            },
            'total_breakouts_captured': int(sum(r.breakouts_captured for r in self.results)),
            'period_results': [
                {
                    'start': r.start_date,
                    'end': r.end_date,
                    'precision': r.precision,
                    'recall': r.recall,
                    'avg_return': r.avg_return,
                    'signals': r.signals_generated,
                    'breakouts': r.breakouts_captured
                }
                for r in self.results
            ]
        }

    def run_walk_forward_test(
        self,
        df: pd.DataFrame,
        filter_optimizer,
        feature_importance: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str = 'date',
        train_months: int = 24,
        test_months: int = 3,
        progress: bool = True
    ) -> Dict[str, Any]:
        """
        Walk-forward backtesting with reoptimization

        Simulates real-world usage where filters are optimized on past data
        and applied to future data.

        Args:
            df: Full dataset with features
            filter_optimizer: FilterOptimizer class (not instance)
            feature_importance: Feature importance for initialization
            X: Feature matrix aligned with df
            y: Labels aligned with df
            date_col: Date column name
            train_months: Months of data for training
            test_months: Months of data for testing

        Returns:
            Walk-forward test results
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        min_date = df[date_col].min()
        max_date = df[date_col].max()

        results = []

        # Start after initial training period
        current_date = min_date + timedelta(days=train_months * 30)

        iterator_dates = []
        while current_date < max_date - timedelta(days=self.lookforward_days):
            iterator_dates.append(current_date)
            current_date += timedelta(days=test_months * 30)

        iterator = tqdm(iterator_dates, desc="Walk-forward test") if progress else iterator_dates

        for test_start in iterator:
            # Training period
            train_end = test_start
            train_start = train_end - timedelta(days=train_months * 30)

            # Test period
            test_end = test_start + timedelta(days=test_months * 30)

            # Get training data
            train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
            train_indices = df[train_mask].index

            # Get test data
            test_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)
            test_indices = df[test_mask].index

            if len(train_indices) < 1000 or len(test_indices) < 100:
                continue

            # Optimize filters on training data
            X_train = X.loc[X.index.isin(train_indices)]
            y_train = y.loc[y.index.isin(train_indices)]

            optimizer = filter_optimizer()
            try:
                optimizer.optimize(X_train, y_train, feature_importance)
            except Exception as e:
                print(f"Optimization failed for period ending {test_start}: {e}")
                continue

            # Apply to test data
            X_test = X.loc[X.index.isin(test_indices)]
            y_test = y.loc[y.index.isin(test_indices)]

            if len(X_test) == 0:
                continue

            filter_mask = optimizer.apply_filters(X_test)

            # Evaluate
            filtered_count = filter_mask.sum()
            if filtered_count == 0:
                continue

            filtered_breakouts = y_test[filter_mask].sum()
            precision = filtered_breakouts / filtered_count if filtered_count > 0 else 0

            total_breakouts = y_test.sum()
            recall = filtered_breakouts / total_breakouts if total_breakouts > 0 else 0

            results.append({
                'period_start': test_start.strftime('%Y-%m-%d'),
                'period_end': test_end.strftime('%Y-%m-%d'),
                'signals': int(filtered_count),
                'breakouts_captured': int(filtered_breakouts),
                'precision': float(precision),
                'recall': float(recall),
                'num_criteria': len(optimizer.criteria)
            })

        if not results:
            return {'error': 'No valid walk-forward periods'}

        # Summarize
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]

        return {
            'num_periods': len(results),
            'avg_precision': float(np.mean(precisions)),
            'std_precision': float(np.std(precisions)),
            'avg_recall': float(np.mean(recalls)),
            'std_recall': float(np.std(recalls)),
            'total_signals': int(sum(r['signals'] for r in results)),
            'total_breakouts_captured': int(sum(r['breakouts_captured'] for r in results)),
            'period_results': results
        }

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, pd.Series],
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Compare multiple filter strategies

        Args:
            df: DataFrame with data
            strategies: Dict of strategy name -> filter mask
            date_col: Date column

        Returns:
            Comparison DataFrame
        """
        comparison_results = []

        for name, mask in strategies.items():
            print(f"\nBacktesting strategy: {name}")
            results = self.run_backtest(df, mask, date_col, progress=False)

            if 'error' not in results:
                comparison_results.append({
                    'strategy': name,
                    'avg_precision': results['precision']['mean'],
                    'std_precision': results['precision']['std'],
                    'avg_recall': results['recall']['mean'],
                    'avg_return': results['returns']['mean'],
                    'avg_win_rate': results['win_rate']['mean'],
                    'total_signals': results['signals_per_period']['total'],
                    'breakouts_captured': results['total_breakouts_captured']
                })

        return pd.DataFrame(comparison_results)


def run_backtest(
    df: pd.DataFrame,
    filter_mask: pd.Series,
    breakout_threshold: float = 0.20,
    lookforward_days: int = 63
) -> Dict[str, Any]:
    """
    Convenience function to run a backtest
    """
    engine = BacktestEngine(
        breakout_threshold=breakout_threshold,
        lookforward_days=lookforward_days
    )
    return engine.run_backtest(df, filter_mask)
