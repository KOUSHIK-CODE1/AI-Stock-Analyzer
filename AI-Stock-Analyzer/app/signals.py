"""Trading signal generation built around the original MACD philosophy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    """Thresholds for the smart signal engine."""

    macd_mode_days: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    volume_spike_ratio: float = 1.5


class SignalEngine:
    """Create original MACD decisions and richer multi-factor signals."""

    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()

    def add_original_macd_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preserve the upstream buy/sell decision logic."""
        frame = data.copy()
        macd = frame["MACD"]
        signal = frame["Signal_Line"]

        # Exact original decision philosophy: buy when MACD is above signal,
        # sell otherwise. This remains visible for auditability.
        frame["Original_MACD_Decision"] = np.where(macd > signal, "buy", "sell")

        bullish_cross = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        bearish_cross = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        frame["MACD_Crossover"] = np.select(
            [bullish_cross, bearish_cross],
            ["bullish", "bearish"],
            default="none",
        )
        frame["MACD_Signal"] = np.select(
            [bullish_cross, bearish_cross],
            ["Buy", "Sell"],
            default="Hold",
        )
        return frame

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Strong Buy, Buy, Hold, Sell, and Strong Sell labels."""
        frame = self.add_original_macd_signals(data)
        score = pd.Series(0, index=frame.index, dtype="float64")

        macd_above_signal = frame["MACD"] > frame["Signal_Line"]
        score += np.where(macd_above_signal, 1, -1)
        score += np.where(frame["MACD_Crossover"].eq("bullish"), 2, 0)
        score += np.where(frame["MACD_Crossover"].eq("bearish"), -2, 0)

        score += np.where(frame["RSI"] <= self.config.rsi_oversold, 2, 0)
        score += np.where(frame["RSI"] >= self.config.rsi_overbought, -2, 0)

        sma20 = frame["SMA_20"]
        sma50 = frame["SMA_50"]
        golden_cross = (sma20 > sma50) & (sma20.shift(1) <= sma50.shift(1))
        death_cross = (sma20 < sma50) & (sma20.shift(1) >= sma50.shift(1))
        score += np.where(golden_cross, 2, 0)
        score += np.where(death_cross, -2, 0)

        trend_up = (frame["Close"] > sma50) & (frame["EMA_Diff"] > 0)
        trend_down = (frame["Close"] < sma50) & (frame["EMA_Diff"] < 0)
        score += np.where(trend_up, 1, 0)
        score += np.where(trend_down, -1, 0)

        volume_spike = frame["Volume_Ratio"] >= self.config.volume_spike_ratio
        up_day = frame["Daily_Return"] > 0
        down_day = frame["Daily_Return"] < 0
        score += np.where(volume_spike & up_day, 1, 0)
        score += np.where(volume_spike & down_day, -1, 0)

        score += np.where(frame["Momentum_5"] > 0, 1, 0)
        score += np.where(frame["Momentum_5"] < 0, -1, 0)

        frame["Trend_Confirmed"] = np.where(
            trend_up, "Uptrend", np.where(trend_down, "Downtrend", "Mixed")
        )
        frame["Signal_Score"] = score.fillna(0)
        frame["Trade_Signal"] = frame["Signal_Score"].apply(self._score_to_signal)
        frame["Signal_Reason"] = frame.apply(self._build_reason, axis=1)
        frame["Buy_Marker"] = np.where(
            frame["Trade_Signal"].isin(["Strong Buy", "Buy"]), frame["Close"], np.nan
        )
        frame["Sell_Marker"] = np.where(
            frame["Trade_Signal"].isin(["Strong Sell", "Sell"]), frame["Close"], np.nan
        )
        return frame

    def original_macd_summary(self, data: pd.DataFrame) -> dict[str, Any]:
        """Return the same final decision idea used by the original repo."""
        if "Original_MACD_Decision" not in data.columns:
            data = self.add_original_macd_signals(data)

        lookback = max(1, self.config.macd_mode_days)
        recent = data["Original_MACD_Decision"].dropna().tail(lookback)
        if recent.empty:
            mode = "unknown"
            latest = "unknown"
        else:
            mode = str(recent.mode().iloc[0])
            latest = str(recent.iloc[-1])

        return {
            "lookback_days": lookback,
            "mode_decision": mode,
            "latest_decision": latest,
        }

    @staticmethod
    def _score_to_signal(score: float) -> str:
        if score >= 5:
            return "Strong Buy"
        if score >= 2:
            return "Buy"
        if score <= -5:
            return "Strong Sell"
        if score <= -2:
            return "Sell"
        return "Hold"

    @staticmethod
    def _build_reason(row: pd.Series) -> str:
        """Create a compact reason string for dashboards and CSV exports."""
        reasons: list[str] = []
        if row.get("MACD_Crossover") == "bullish":
            reasons.append("bullish MACD crossover")
        elif row.get("MACD_Crossover") == "bearish":
            reasons.append("bearish MACD crossover")
        elif row.get("MACD", 0) > row.get("Signal_Line", 0):
            reasons.append("MACD above signal")
        else:
            reasons.append("MACD below signal")

        rsi = row.get("RSI")
        if pd.notna(rsi):
            if rsi <= 30:
                reasons.append("RSI oversold")
            elif rsi >= 70:
                reasons.append("RSI overbought")

        if row.get("Trend_Confirmed") == "Uptrend":
            reasons.append("trend up")
        elif row.get("Trend_Confirmed") == "Downtrend":
            reasons.append("trend down")

        if row.get("Volume_Ratio", 0) >= 1.5:
            reasons.append("volume spike")

        return ", ".join(reasons)
