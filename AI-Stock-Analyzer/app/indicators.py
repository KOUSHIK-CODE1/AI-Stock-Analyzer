"""Technical indicator calculations.

The MACD calculation intentionally mirrors the original repository:
12-period EMA minus 26-period EMA, followed by a 9-period EMA signal line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands


@dataclass(frozen=True)
class MACDConfig:
    """Classic MACD settings preserved from the source repository."""

    fast_span: int = 12
    slow_span: int = 26
    signal_span: int = 9


class TechnicalIndicatorCalculator:
    """Adds MACD and supporting technical indicators to OHLCV data."""

    def __init__(self, macd_config: MACDConfig | None = None) -> None:
        self.macd_config = macd_config or MACDConfig()

    def add_original_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preserve the original repository's EMA-based MACD logic."""
        frame = data.copy()
        close = pd.to_numeric(frame["Close"], errors="coerce")

        # Original repo logic:
        # exp1 = close.ewm(span=12, adjust=False).mean()
        # exp2 = close.ewm(span=26, adjust=False).mean()
        # macd = exp1 - exp2
        # signal = macd.ewm(span=9, adjust=False).mean()
        frame["EMA_12"] = close.ewm(
            span=self.macd_config.fast_span, adjust=False
        ).mean()
        frame["EMA_26"] = close.ewm(
            span=self.macd_config.slow_span, adjust=False
        ).mean()
        frame["MACD"] = frame["EMA_12"] - frame["EMA_26"]
        frame["Signal_Line"] = frame["MACD"].ewm(
            span=self.macd_config.signal_span, adjust=False
        ).mean()
        frame["MACD_Histogram"] = frame["MACD"] - frame["Signal_Line"]
        frame["EMA_Diff"] = frame["EMA_12"] - frame["EMA_26"]
        return frame

    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced indicators used by signals, ML, and backtesting."""
        frame = self.add_original_macd(data)

        close = pd.to_numeric(frame["Close"], errors="coerce")
        high = pd.to_numeric(frame["High"], errors="coerce")
        low = pd.to_numeric(frame["Low"], errors="coerce")
        volume = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)

        frame["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
        frame["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
        frame["EMA_12_TA"] = EMAIndicator(close=close, window=12).ema_indicator()
        frame["EMA_26_TA"] = EMAIndicator(close=close, window=26).ema_indicator()
        frame["RSI"] = RSIIndicator(close=close, window=14).rsi()

        bollinger = BollingerBands(close=close, window=20, window_dev=2)
        frame["BB_Upper"] = bollinger.bollinger_hband()
        frame["BB_Middle"] = bollinger.bollinger_mavg()
        frame["BB_Lower"] = bollinger.bollinger_lband()
        frame["BB_Width"] = (
            (frame["BB_Upper"] - frame["BB_Lower"]) / frame["BB_Middle"]
        ).replace([np.inf, -np.inf], np.nan)

        frame["ATR"] = AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        stoch_rsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        frame["StochRSI"] = stoch_rsi.stochrsi_k() * 100

        frame["Volume_SMA_20"] = volume.rolling(window=20, min_periods=1).mean()
        frame["Volume_Ratio"] = (volume / frame["Volume_SMA_20"]).replace(
            [np.inf, -np.inf], np.nan
        )
        frame["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        frame["ADI"] = AccDistIndexIndicator(
            high=high, low=low, close=close, volume=volume
        ).acc_dist_index()

        frame["Daily_Return"] = close.pct_change()
        frame["Momentum_5"] = close.pct_change(periods=5)
        frame["Momentum_10"] = close.pct_change(periods=10)
        frame["Price_Change"] = close.diff()
        frame["Volatility_20"] = frame["Daily_Return"].rolling(window=20).std()
        frame["Close_to_SMA20"] = (close / frame["SMA_20"] - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        frame["Close_to_SMA50"] = (close / frame["SMA_50"] - 1).replace(
            [np.inf, -np.inf], np.nan
        )

        return frame.replace([np.inf, -np.inf], np.nan)
