"""
Feature engineering pipeline
Combines technical indicators with fundamental data
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from tqdm import tqdm
from pathlib import Path

from .technical import TechnicalIndicators


class FeatureEngineer:
    """
    Complete feature engineering pipeline for stock data
    """

    def __init__(self):
        self.feature_columns = []
        self.categorical_columns = ['sector', 'industry']

    def engineer_features(
        self,
        ohlcv_df: pd.DataFrame,
        fundamentals_df: Optional[pd.DataFrame] = None,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Engineer features for all stocks in the dataset

        Args:
            ohlcv_df: OHLCV data with 'ticker' column
            fundamentals_df: Fundamental data (optional)
            progress: Show progress bar

        Returns:
            DataFrame with all engineered features
        """
        tickers = ohlcv_df['ticker'].unique()
        all_features = []

        iterator = tqdm(tickers, desc="Engineering features") if progress else tickers

        for ticker in iterator:
            ticker_df = ohlcv_df[ohlcv_df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

            # Skip if not enough data
            if len(ticker_df) < 252:  # Need at least 1 year
                continue

            try:
                # Add technical indicators
                ticker_df = TechnicalIndicators.add_all_indicators(ticker_df)

                # Add fundamental features if available
                if fundamentals_df is not None and ticker in fundamentals_df['ticker'].values:
                    fund_row = fundamentals_df[fundamentals_df['ticker'] == ticker].iloc[0]
                    ticker_df = self._add_fundamental_features(ticker_df, fund_row)

                all_features.append(ticker_df)

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        if not all_features:
            return pd.DataFrame()

        # Combine all data
        result_df = pd.concat(all_features, ignore_index=True)

        # Store feature columns (excluding identifiers and target)
        exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                        'dividends', 'stock_splits', 'expected_return', 'sector', 'industry']
        self.feature_columns = [c for c in result_df.columns if c not in exclude_cols]

        return result_df

    def _add_fundamental_features(
        self,
        df: pd.DataFrame,
        fundamentals: pd.Series
    ) -> pd.DataFrame:
        """Add fundamental data as features"""

        # Static fundamentals (broadcast to all rows)
        fundamental_features = [
            'market_cap', 'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
            'price_to_sales', 'ev_to_revenue', 'ev_to_ebitda', 'profit_margin',
            'operating_margin', 'return_on_assets', 'return_on_equity',
            'revenue_growth', 'earnings_growth', 'current_ratio', 'debt_to_equity',
            'beta', 'short_ratio', 'short_percent_of_float', 'dividend_yield',
            'payout_ratio'
        ]

        for feat in fundamental_features:
            if feat in fundamentals.index:
                df[f'fund_{feat}'] = fundamentals[feat]

        # Add sector and industry as categorical
        if 'sector' in fundamentals.index:
            df['sector'] = fundamentals['sector']
        if 'industry' in fundamentals.index:
            df['industry'] = fundamentals['industry']

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'breakout',
        min_samples: int = 100,
        drop_na_threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for ML training

        Args:
            df: DataFrame with engineered features
            target_col: Target column name
            min_samples: Minimum samples required
            drop_na_threshold: Drop columns with more than this fraction of NaN

        Returns:
            Tuple of (features DataFrame, target Series, feature column names)
        """
        # Remove rows with NaN target
        df = df[df[target_col].notna()].copy()

        if len(df) < min_samples:
            raise ValueError(f"Not enough samples: {len(df)} < {min_samples}")

        # Get feature columns
        exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                        'dividends', 'stock_splits', 'expected_return',
                        target_col, 'sector', 'industry']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Drop columns with too many NaN
        nan_fractions = df[feature_cols].isna().mean()
        valid_cols = nan_fractions[nan_fractions < drop_na_threshold].index.tolist()

        # Prepare X and y
        X = df[valid_cols].copy()
        y = df[target_col].copy()

        # Fill remaining NaN with median
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        return X, y, valid_cols

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names"""
        return self.feature_columns

    @staticmethod
    def create_time_splits(
        df: pd.DataFrame,
        date_col: str = 'date',
        n_splits: int = 5,
        test_size: int = 63  # ~3 months
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Create time-series cross-validation splits

        Args:
            df: DataFrame with date column
            date_col: Name of date column
            n_splits: Number of CV splits
            test_size: Size of test set in trading days

        Returns:
            List of (train_idx, test_idx) tuples
        """
        dates = df[date_col].sort_values().unique()
        n_dates = len(dates)

        splits = []
        test_start_step = (n_dates - test_size) // n_splits

        for i in range(n_splits):
            test_start_idx = test_start_step * (i + 1)
            test_end_idx = test_start_idx + test_size

            if test_end_idx > n_dates:
                break

            train_dates = dates[:test_start_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            train_mask = df[date_col].isin(train_dates)
            test_mask = df[date_col].isin(test_dates)

            train_idx = df[train_mask].index
            test_idx = df[test_mask].index

            splits.append((train_idx, test_idx))

        return splits


def engineer_and_save(
    ohlcv_path: str,
    fundamentals_path: Optional[str] = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Convenience function to engineer features and save to file
    """
    ohlcv_df = pd.read_parquet(ohlcv_path)

    fundamentals_df = None
    if fundamentals_path and Path(fundamentals_path).exists():
        fundamentals_df = pd.read_parquet(fundamentals_path)

    engineer = FeatureEngineer()
    result = engineer.engineer_features(ohlcv_df, fundamentals_df)

    if output_path:
        result.to_parquet(output_path)
        print(f"Saved engineered features to {output_path}")

    return result
