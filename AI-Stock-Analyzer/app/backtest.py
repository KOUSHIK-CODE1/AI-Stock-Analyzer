"""Long-only strategy backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings."""

    initial_capital: float = 10_000.0
    fee_rate: float = 0.001
    risk_free_rate: float = 0.0


@dataclass
class BacktestResult:
    """Backtest outputs used by the CLI and dashboard."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


class StrategyBacktester:
    """Simulate a simple long-only strategy from generated signals."""

    buy_signals = {"Buy", "Strong Buy"}
    sell_signals = {"Sell", "Strong Sell"}

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, signal_col: str = "Trade_Signal") -> BacktestResult:
        if data.empty:
            raise ValueError("Cannot backtest an empty DataFrame.")
        if signal_col not in data.columns:
            raise ValueError(f"Signal column '{signal_col}' not found.")

        cash = float(self.config.initial_capital)
        shares = 0.0
        entry_cost = 0.0
        trades: list[dict[str, Any]] = []
        curve: list[dict[str, Any]] = []

        for date, row in data.iterrows():
            price = float(row["Close"])
            signal = str(row[signal_col])

            if price <= 0 or np.isnan(price):
                continue

            if signal in self.buy_signals and shares == 0 and cash > 0:
                fee = cash * self.config.fee_rate
                investable_cash = max(cash - fee, 0)
                shares = investable_cash / price
                entry_cost = cash
                cash = 0.0
                trades.append(
                    {
                        "Date": date,
                        "Action": "BUY",
                        "Signal": signal,
                        "Price": price,
                        "Shares": shares,
                        "Fee": fee,
                        "Cash": cash,
                        "Portfolio_Value": shares * price,
                        "PnL": 0.0,
                    }
                )

            elif signal in self.sell_signals and shares > 0:
                gross_value = shares * price
                fee = gross_value * self.config.fee_rate
                cash = gross_value - fee
                pnl = cash - entry_cost
                trades.append(
                    {
                        "Date": date,
                        "Action": "SELL",
                        "Signal": signal,
                        "Price": price,
                        "Shares": shares,
                        "Fee": fee,
                        "Cash": cash,
                        "Portfolio_Value": cash,
                        "PnL": pnl,
                    }
                )
                shares = 0.0
                entry_cost = 0.0

            portfolio_value = cash + shares * price
            curve.append(
                {
                    "Date": date,
                    "Close": price,
                    "Signal": signal,
                    "Cash": cash,
                    "Shares": shares,
                    "Position_Value": shares * price,
                    "Portfolio_Value": portfolio_value,
                }
            )

        equity_curve = pd.DataFrame(curve).set_index("Date")
        trades_frame = pd.DataFrame(trades)
        metrics = self._calculate_metrics(data, equity_curve, trades_frame)
        return BacktestResult(equity_curve=equity_curve, trades=trades_frame, metrics=metrics)

    def _calculate_metrics(
        self,
        data: pd.DataFrame,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> dict[str, float]:
        final_value = (
            float(equity_curve["Portfolio_Value"].iloc[-1])
            if not equity_curve.empty
            else self.config.initial_capital
        )
        total_return = final_value / self.config.initial_capital - 1

        first_close = float(data["Close"].iloc[0])
        last_close = float(data["Close"].iloc[-1])
        benchmark_return = last_close / first_close - 1 if first_close else 0.0

        daily_returns = equity_curve["Portfolio_Value"].pct_change().dropna()
        if daily_returns.std(ddof=0) > 0:
            daily_rf = self.config.risk_free_rate / 252
            sharpe = (
                (daily_returns - daily_rf).mean()
                / daily_returns.std(ddof=0)
                * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        rolling_peak = equity_curve["Portfolio_Value"].cummax()
        drawdown = equity_curve["Portfolio_Value"] / rolling_peak - 1
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        sell_trades = trades[trades["Action"].eq("SELL")] if not trades.empty else trades
        wins = sell_trades[sell_trades["PnL"] > 0] if not sell_trades.empty else sell_trades
        win_rate = len(wins) / len(sell_trades) if len(sell_trades) else 0.0

        days = max((data.index[-1] - data.index[0]).days, 1)
        annualized_return = (1 + total_return) ** (365 / days) - 1

        return {
            "initial_capital": float(self.config.initial_capital),
            "final_value": final_value,
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "benchmark_return_pct": benchmark_return * 100,
            "sharpe_ratio": float(sharpe),
            "max_drawdown_pct": max_drawdown * 100,
            "trade_count": float(len(trades)),
            "winning_trade_count": float(len(wins)),
            "win_rate_pct": win_rate * 100,
        }
