"""
Technical indicators calculation
Computes 100+ technical indicators for stock analysis
"""
import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Calculates comprehensive technical indicators for stock data
    """

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with all technical indicators added
        """
        df = df.copy()

        # Ensure we have the required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Price-based features
        df = TechnicalIndicators._add_price_features(df)

        # Moving averages
        df = TechnicalIndicators._add_moving_averages(df)

        # Momentum indicators
        df = TechnicalIndicators._add_momentum_indicators(df)

        # Volatility indicators
        df = TechnicalIndicators._add_volatility_indicators(df)

        # Volume indicators
        df = TechnicalIndicators._add_volume_indicators(df)

        # Trend indicators
        df = TechnicalIndicators._add_trend_indicators(df)

        # Pattern recognition
        df = TechnicalIndicators._add_pattern_features(df)

        # Statistical features
        df = TechnicalIndicators._add_statistical_features(df)

        return df

    @staticmethod
    def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-derived features"""

        # Returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_21d'] = df['close'].pct_change(21)  # ~1 month
        df['return_63d'] = df['close'].pct_change(63)  # ~3 months

        # Log returns
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5d'] = np.log(df['close'] / df['close'].shift(5))

        # Price position
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_to_high'] = (df['high'] - df['close']) / df['close']
        df['close_to_low'] = (df['close'] - df['low']) / df['close']
        df['open_close_range'] = (df['close'] - df['open']) / df['open']

        # Gap analysis
        df['gap_up'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Price relative to recent highs/lows
        for period in [5, 10, 21, 63]:
            df[f'price_vs_high_{period}d'] = df['close'] / df['high'].rolling(period).max()
            df[f'price_vs_low_{period}d'] = df['close'] / df['low'].rolling(period).min()

        return df

    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Moving average indicators"""

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}']

        # Exponential Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_vs_ema_{period}'] = df['close'] / df[f'ema_{period}']

        # Moving Average Crossovers
        df['sma_5_20_cross'] = df['sma_5'] / df['sma_20']
        df['sma_20_50_cross'] = df['sma_20'] / df['sma_50']
        df['sma_50_200_cross'] = df['sma_50'] / df['sma_200']
        df['ema_5_20_cross'] = df['ema_5'] / df['ema_20']
        df['ema_20_50_cross'] = df['ema_20'] / df['ema_50']

        # Golden/Death Cross signals
        df['golden_cross'] = ((df['sma_50'] > df['sma_200']) &
                              (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
        df['death_cross'] = ((df['sma_50'] < df['sma_200']) &
                              (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)

        # MA slopes
        for period in [20, 50, 200]:
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(5)

        return df

    @staticmethod
    def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators"""

        # RSI - Relative Strength Index
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_slope'] = df['macd_histogram'].diff()

        # Rate of Change (ROC)
        for period in [5, 10, 21, 63]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                   df['close'].shift(period)) * 100

        # Momentum
        for period in [10, 21]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Williams %R
        for period in [14, 21]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)

        # CCI - Commodity Channel Index
        for period in [14, 20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            tp_sma = tp.rolling(window=period).mean()
            tp_mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f'cci_{period}'] = (tp - tp_sma) / (0.015 * tp_mad)

        # Ultimate Oscillator
        bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
        tr = np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        df['ultimate_oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

        # Awesome Oscillator
        df['awesome_oscillator'] = (((df['high'] + df['low']) / 2).rolling(5).mean() -
                                    ((df['high'] + df['low']) / 2).rolling(34).mean())

        return df

    @staticmethod
    def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators"""

        # Bollinger Bands
        for period in [20]:
            for std_mult in [2]:
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = sma + (std * std_mult)
                df[f'bb_lower_{period}'] = sma - (std * std_mult)
                df[f'bb_middle_{period}'] = sma
                df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
                df[f'bb_position_{period}'] = ((df['close'] - df[f'bb_lower_{period}']) /
                                               (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']))

                # Bollinger Band Squeeze (key breakout indicator!)
                df[f'bb_squeeze_{period}'] = df[f'bb_width_{period}'].rolling(20).apply(
                    lambda x: 1 if x.iloc[-1] == x.min() else 0
                )

        # ATR - Average True Range
        for period in [14, 21]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(window=period).mean()
            df[f'atr_percent_{period}'] = df[f'atr_{period}'] / df['close'] * 100

        # Volatility (standard deviation)
        for period in [10, 21, 63]:
            df[f'volatility_{period}'] = df['return_1d'].rolling(window=period).std() * np.sqrt(252)

        # Keltner Channels
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        atr_10 = df['atr_14']
        df['keltner_upper'] = ema_20 + (2 * atr_10)
        df['keltner_lower'] = ema_20 - (2 * atr_10)
        df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'])

        # Donchian Channels
        for period in [20]:
            df[f'donchian_high_{period}'] = df['high'].rolling(window=period).max()
            df[f'donchian_low_{period}'] = df['low'].rolling(window=period).min()
            df[f'donchian_mid_{period}'] = (df[f'donchian_high_{period}'] + df[f'donchian_low_{period}']) / 2
            df[f'donchian_position_{period}'] = ((df['close'] - df[f'donchian_low_{period}']) /
                                                  (df[f'donchian_high_{period}'] - df[f'donchian_low_{period}']))

        # Historical Volatility Ratio
        df['hv_ratio'] = df['volatility_10'] / df['volatility_63']

        return df

    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""

        # Volume Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
        df['obv_slope'] = df['obv'].diff(5) / 5

        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['ad_line'] = (clv * df['volume']).cumsum()

        # Chaikin Money Flow
        for period in [20, 21]:
            mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mf_mult = mf_mult.fillna(0)
            mf_volume = mf_mult * df['volume']
            df[f'cmf_{period}'] = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

        # Money Flow Index (MFI)
        for period in [14]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            mf = tp * df['volume']
            mf_pos = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
            mf_neg = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
            df[f'mfi_{period}'] = 100 - (100 / (1 + mf_pos / mf_neg))

        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()

        # Ease of Movement
        dm = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        br = (df['volume'] / 1e8) / (df['high'] - df['low'])
        df['eom'] = dm / br
        df['eom_sma_14'] = df['eom'].rolling(window=14).mean()

        # Force Index
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_13'] = df['force_index'].ewm(span=13, adjust=False).mean()

        # Volume-weighted price features
        df['vwap_ratio'] = df['close'] / ((df['close'] * df['volume']).rolling(20).sum() /
                                           df['volume'].rolling(20).sum())

        # Negative Volume Index
        df['nvi'] = np.where(df['volume'] < df['volume'].shift(1),
                             df['close'].pct_change(), 0).cumsum()

        # Positive Volume Index
        df['pvi'] = np.where(df['volume'] > df['volume'].shift(1),
                             df['close'].pct_change(), 0).cumsum()

        return df

    @staticmethod
    def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Trend indicators"""

        # ADX - Average Directional Index
        for period in [14]:
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()

            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            tr = pd.concat([
                df['high'] - df['low'],
                np.abs(df['high'] - df['close'].shift()),
                np.abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)

            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'adx_{period}'] = dx.rolling(window=period).mean()
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di
            df[f'di_diff_{period}'] = plus_di - minus_di

        # Aroon Indicator
        for period in [25]:
            df[f'aroon_up_{period}'] = 100 * df['high'].rolling(window=period + 1).apply(
                lambda x: x.argmax()) / period
            df[f'aroon_down_{period}'] = 100 * df['low'].rolling(window=period + 1).apply(
                lambda x: x.argmin()) / period
            df[f'aroon_oscillator_{period}'] = df[f'aroon_up_{period}'] - df[f'aroon_down_{period}']

        # TRIX
        ema1 = df['close'].ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        df['trix'] = ema3.pct_change() * 100

        # Vortex Indicator
        for period in [14]:
            tr = pd.concat([
                df['high'] - df['low'],
                np.abs(df['high'] - df['close'].shift()),
                np.abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)

            vm_plus = np.abs(df['high'] - df['low'].shift())
            vm_minus = np.abs(df['low'] - df['high'].shift())

            df[f'vortex_plus_{period}'] = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
            df[f'vortex_minus_{period}'] = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
            df[f'vortex_diff_{period}'] = df[f'vortex_plus_{period}'] - df[f'vortex_minus_{period}']

        # Mass Index
        ema_hl = (df['high'] - df['low']).ewm(span=9, adjust=False).mean()
        ema_ema_hl = ema_hl.ewm(span=9, adjust=False).mean()
        df['mass_index'] = (ema_hl / ema_ema_hl).rolling(window=25).sum()

        # Trend Strength
        df['trend_strength'] = np.abs(df['close'] - df['sma_50']) / df['atr_14']

        return df

    @staticmethod
    def _add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick and price pattern features"""

        # Candlestick body analysis
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['body_to_shadow'] = df['body_size'] / (df['upper_shadow'] + df['lower_shadow'] + 0.0001)

        # Bullish/Bearish candle
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['bullish_streak'] = df['is_bullish'].groupby(
            (df['is_bullish'] != df['is_bullish'].shift()).cumsum()
        ).cumsum() * df['is_bullish']

        # Doji detection
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)

        # Hammer pattern (simplified)
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) &
                           (df['upper_shadow'] < df['body_size'])).astype(int)

        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_high_streak'] = df['higher_high'].rolling(5).sum()
        df['lower_low_streak'] = df['lower_low'].rolling(5).sum()

        # Price compression (consolidation detection)
        df['price_range_5d'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
        df['price_range_10d'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']
        df['consolidation'] = df['price_range_10d'] / df['price_range_10d'].rolling(50).mean()

        # Breakout from consolidation
        df['breaking_high_20'] = (df['close'] > df['high'].shift(1).rolling(20).max()).astype(int)
        df['breaking_low_20'] = (df['close'] < df['low'].shift(1).rolling(20).min()).astype(int)

        return df

    @staticmethod
    def _add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Statistical and derived features"""

        # Skewness and Kurtosis of returns
        df['return_skew_20'] = df['return_1d'].rolling(window=20).skew()
        df['return_kurt_20'] = df['return_1d'].rolling(window=20).kurt()

        # Z-score of price
        for period in [20, 50]:
            mean = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'price_zscore_{period}'] = (df['close'] - mean) / std

        # Relative volume
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()

        # Price acceleration
        df['price_acceleration'] = df['return_1d'] - df['return_1d'].shift(1)

        # Momentum divergence (price making new highs but momentum not)
        price_high_20 = df['close'].rolling(20).max()
        rsi_high_20 = df['rsi_14'].rolling(20).max()
        df['bearish_divergence'] = ((df['close'] >= price_high_20 * 0.99) &
                                    (df['rsi_14'] < rsi_high_20 * 0.9)).astype(int)

        # Support/Resistance levels (simplified)
        df['near_52w_high'] = df['close'] / df['close'].rolling(252).max()
        df['near_52w_low'] = df['close'] / df['close'].rolling(252).min()

        # Days since 52-week high/low
        df['days_since_52w_high'] = df['close'].rolling(252).apply(
            lambda x: len(x) - 1 - np.argmax(x) if len(x) > 0 else np.nan
        )

        # Average return expectation based on momentum
        df['expected_return'] = df['return_21d'].shift(-21)  # Forward looking (for training only)

        return df


# Convenience function
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for a stock dataframe"""
    return TechnicalIndicators.add_all_indicators(df)
