"""
Data collector for S&P 500 historical stock data
Uses yfinance for free historical data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sp500_list import get_sp500_tickers


class DataCollector:
    """
    Collects historical OHLCV and fundamental data for S&P 500 stocks
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def collect_single_stock(
        self,
        ticker: str,
        period: str = "max",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Collect historical data for a single stock

        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                return None

            # Add ticker column
            df['Ticker'] = ticker

            # Reset index to make Date a column
            df = df.reset_index()

            # Standardize column names
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]

            return df

        except Exception as e:
            print(f"Error collecting data for {ticker}: {e}")
            return None

    def collect_fundamental_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Collect fundamental data for a stock
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract relevant fundamental metrics
            fundamentals = {
                'ticker': ticker,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'enterprise_value': info.get('enterpriseValue'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_equity': info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'current_ratio': info.get('currentRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'free_cash_flow': info.get('freeCashflow'),
                'operating_cash_flow': info.get('operatingCashflow'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_day_average': info.get('fiftyDayAverage'),
                'two_hundred_day_average': info.get('twoHundredDayAverage'),
                'avg_volume': info.get('averageVolume'),
                'avg_volume_10d': info.get('averageVolume10days'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'short_ratio': info.get('shortRatio'),
                'short_percent_of_float': info.get('shortPercentOfFloat'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
            }

            return fundamentals

        except Exception as e:
            print(f"Error collecting fundamentals for {ticker}: {e}")
            return None

    def collect_all_sp500(
        self,
        period: str = "10y",
        max_workers: int = 10,
        save_progress: bool = True
    ) -> pd.DataFrame:
        """
        Collect historical data for all S&P 500 stocks

        Args:
            period: Data period
            max_workers: Number of parallel download threads
            save_progress: Save data incrementally

        Returns:
            Combined DataFrame with all stock data
        """
        tickers = get_sp500_tickers()
        print(f"Collecting data for {len(tickers)} S&P 500 stocks...")

        all_data = []
        all_fundamentals = []
        failed_tickers = []

        # Use threading for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self._collect_stock_with_fundamentals, ticker, period): ticker
                for ticker in tickers
            }

            # Process results as they complete
            for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Downloading"):
                ticker = future_to_ticker[future]
                try:
                    ohlcv_df, fundamentals = future.result()

                    if ohlcv_df is not None:
                        all_data.append(ohlcv_df)
                    else:
                        failed_tickers.append(ticker)

                    if fundamentals is not None:
                        all_fundamentals.append(fundamentals)

                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    failed_tickers.append(ticker)

        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            if save_progress:
                # Save OHLCV data
                ohlcv_path = self.data_dir / "sp500_ohlcv.parquet"
                combined_df.to_parquet(ohlcv_path)
                print(f"Saved OHLCV data to {ohlcv_path}")

                # Save fundamentals
                if all_fundamentals:
                    fundamentals_df = pd.DataFrame(all_fundamentals)
                    fundamentals_path = self.data_dir / "sp500_fundamentals.parquet"
                    fundamentals_df.to_parquet(fundamentals_path)
                    print(f"Saved fundamentals to {fundamentals_path}")

                # Save failed tickers
                if failed_tickers:
                    failed_path = self.data_dir / "failed_tickers.json"
                    with open(failed_path, 'w') as f:
                        json.dump(failed_tickers, f)
                    print(f"Failed to collect {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

            return combined_df

        return pd.DataFrame()

    def _collect_stock_with_fundamentals(
        self,
        ticker: str,
        period: str
    ) -> tuple:
        """Helper to collect both OHLCV and fundamental data"""
        ohlcv = self.collect_single_stock(ticker, period=period)
        fundamentals = self.collect_fundamental_data(ticker)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

        return ohlcv, fundamentals

    def load_data(self) -> tuple:
        """
        Load previously collected data

        Returns:
            Tuple of (ohlcv_df, fundamentals_df)
        """
        ohlcv_path = self.data_dir / "sp500_ohlcv.parquet"
        fundamentals_path = self.data_dir / "sp500_fundamentals.parquet"

        ohlcv_df = None
        fundamentals_df = None

        if ohlcv_path.exists():
            ohlcv_df = pd.read_parquet(ohlcv_path)
            print(f"Loaded OHLCV data: {len(ohlcv_df)} rows, {ohlcv_df['ticker'].nunique()} tickers")

        if fundamentals_path.exists():
            fundamentals_df = pd.read_parquet(fundamentals_path)
            print(f"Loaded fundamentals: {len(fundamentals_df)} companies")

        return ohlcv_df, fundamentals_df

    def get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the collected data"""
        if df is None or df.empty:
            return {}

        return {
            'total_rows': len(df),
            'num_tickers': df['ticker'].nunique(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'tickers': df['ticker'].unique().tolist(),
            'columns': df.columns.tolist()
        }


if __name__ == "__main__":
    collector = DataCollector()

    # Collect all S&P 500 data
    print("Starting S&P 500 data collection...")
    df = collector.collect_all_sp500(period="10y")

    if not df.empty:
        stats = collector.get_data_stats(df)
        print(f"\nCollection complete!")
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Number of tickers: {stats['num_tickers']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
