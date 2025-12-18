"""
Breakout labeling system
Identifies stocks that had significant gains within a specified time window
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class BreakoutLabeler:
    """
    Labels historical data with breakout events
    A breakout is defined as achieving X% gain within Y trading days
    """

    def __init__(
        self,
        breakout_threshold: float = 0.20,  # 20% gain
        lookforward_days: int = 63,  # ~3 months
        max_drawdown_limit: Optional[float] = None  # Optional: limit drawdown during period
    ):
        """
        Args:
            breakout_threshold: Minimum gain to qualify as breakout (0.20 = 20%)
            lookforward_days: Number of trading days to look forward
            max_drawdown_limit: Maximum drawdown allowed during the period (optional)
        """
        self.breakout_threshold = breakout_threshold
        self.lookforward_days = lookforward_days
        self.max_drawdown_limit = max_drawdown_limit

    def label_breakouts(
        self,
        df: pd.DataFrame,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Add breakout labels to the dataset

        Args:
            df: DataFrame with 'ticker', 'date', 'close', 'high', 'low' columns
            progress: Show progress bar

        Returns:
            DataFrame with 'breakout' and related columns added
        """
        tickers = df['ticker'].unique()
        all_labeled = []

        iterator = tqdm(tickers, desc="Labeling breakouts") if progress else tickers

        for ticker in iterator:
            ticker_df = df[df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

            ticker_df = self._label_single_stock(ticker_df)
            all_labeled.append(ticker_df)

        return pd.concat(all_labeled, ignore_index=True)

    def _label_single_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label breakouts for a single stock"""

        n = len(df)
        breakout = np.zeros(n, dtype=int)
        max_gain = np.full(n, np.nan)
        max_loss = np.full(n, np.nan)
        days_to_breakout = np.full(n, np.nan)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        for i in range(n - self.lookforward_days):
            entry_price = close[i]

            # Get future prices
            future_highs = high[i+1:i+1+self.lookforward_days]
            future_lows = low[i+1:i+1+self.lookforward_days]
            future_closes = close[i+1:i+1+self.lookforward_days]

            # Calculate max gain and max loss
            max_high = np.max(future_highs)
            min_low = np.min(future_lows)

            gain = (max_high - entry_price) / entry_price
            loss = (entry_price - min_low) / entry_price

            max_gain[i] = gain
            max_loss[i] = loss

            # Check if breakout occurred
            is_breakout = gain >= self.breakout_threshold

            # Optional: Check drawdown constraint
            if is_breakout and self.max_drawdown_limit is not None:
                # Find when breakout occurred
                gains = (future_highs - entry_price) / entry_price
                breakout_idx = np.where(gains >= self.breakout_threshold)[0]

                if len(breakout_idx) > 0:
                    first_breakout = breakout_idx[0]
                    # Check max drawdown before breakout
                    if first_breakout > 0:
                        pre_breakout_lows = future_lows[:first_breakout]
                        max_dd = (entry_price - np.min(pre_breakout_lows)) / entry_price
                        if max_dd > self.max_drawdown_limit:
                            is_breakout = False

            if is_breakout:
                breakout[i] = 1
                # Find days to first breakout
                gains = (future_highs - entry_price) / entry_price
                breakout_idx = np.where(gains >= self.breakout_threshold)[0]
                if len(breakout_idx) > 0:
                    days_to_breakout[i] = breakout_idx[0] + 1

        df['breakout'] = breakout
        df['max_gain_forward'] = max_gain
        df['max_loss_forward'] = max_loss
        df['days_to_breakout'] = days_to_breakout

        # Additional labels for different thresholds
        df['gain_10pct'] = (df['max_gain_forward'] >= 0.10).astype(int)
        df['gain_15pct'] = (df['max_gain_forward'] >= 0.15).astype(int)
        df['gain_25pct'] = (df['max_gain_forward'] >= 0.25).astype(int)
        df['gain_30pct'] = (df['max_gain_forward'] >= 0.30).astype(int)

        return df

    def get_breakout_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about breakouts in the dataset
        """
        if 'breakout' not in df.columns:
            return {}

        total = len(df[df['breakout'].notna()])
        breakouts = df['breakout'].sum()

        stats = {
            'total_samples': total,
            'breakout_count': int(breakouts),
            'breakout_rate': breakouts / total if total > 0 else 0,
            'threshold': self.breakout_threshold,
            'lookforward_days': self.lookforward_days,
        }

        if 'days_to_breakout' in df.columns:
            breakout_df = df[df['breakout'] == 1]
            if len(breakout_df) > 0:
                stats['avg_days_to_breakout'] = breakout_df['days_to_breakout'].mean()
                stats['median_days_to_breakout'] = breakout_df['days_to_breakout'].median()

        if 'max_gain_forward' in df.columns:
            stats['avg_max_gain'] = df['max_gain_forward'].mean()
            stats['avg_max_loss'] = df['max_loss_forward'].mean()

        return stats

    def analyze_breakouts_by_period(
        self,
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Analyze breakout rates by time period
        """
        if 'breakout' not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df['year'] = pd.to_datetime(df[date_col]).dt.year
        df['month'] = pd.to_datetime(df[date_col]).dt.month

        yearly_stats = df.groupby('year').agg({
            'breakout': ['sum', 'count', 'mean']
        }).round(4)

        yearly_stats.columns = ['breakout_count', 'total_samples', 'breakout_rate']

        return yearly_stats


def create_labels(
    df: pd.DataFrame,
    threshold: float = 0.20,
    days: int = 63
) -> pd.DataFrame:
    """
    Convenience function to add breakout labels
    """
    labeler = BreakoutLabeler(
        breakout_threshold=threshold,
        lookforward_days=days
    )
    return labeler.label_breakouts(df)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))

    from pathlib import Path

    data_path = Path(__file__).parent.parent.parent / "data" / "sp500_features.parquet"

    if data_path.exists():
        df = pd.read_parquet(data_path)
        labeler = BreakoutLabeler(breakout_threshold=0.20, lookforward_days=63)
        df = labeler.label_breakouts(df)

        stats = labeler.get_breakout_stats(df)
        print("\nBreakout Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
