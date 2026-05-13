"""High-level orchestration for stock analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from app.backtest import BacktestConfig, BacktestResult, StrategyBacktester
from app.indicators import TechnicalIndicatorCalculator
from app.ml_model import MLResult, StockMovementPredictor
from app.signals import SignalConfig, SignalEngine
from app.utils import (
    CHARTS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    download_stock_data,
    ensure_project_directories,
    export_dataframe,
    latest_price_snapshot,
    logger,
    normalize_ticker,
    safe_filename,
)


@dataclass
class StockAnalysisResult:
    """Complete result bundle returned by StockAnalyzer."""

    ticker: str
    data: pd.DataFrame
    macd_summary: dict[str, Any]
    backtest: BacktestResult
    ml: MLResult | None = None
    latest_snapshot: dict[str, float] = field(default_factory=dict)
    export_paths: dict[str, Path] = field(default_factory=dict)


class StockAnalyzer:
    """Download data, calculate indicators, generate signals, train ML, export."""

    def __init__(
        self,
        signal_config: SignalConfig | None = None,
        output_dir: Path = OUTPUTS_DIR,
        chart_dir: Path = CHARTS_DIR,
    ) -> None:
        ensure_project_directories()
        self.indicators = TechnicalIndicatorCalculator()
        self.signals = SignalEngine(signal_config)
        self.predictor = StockMovementPredictor()
        self.output_dir = output_dir
        self.chart_dir = chart_dir

    def analyze(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d",
        initial_capital: float = 10_000.0,
        run_ml: bool = True,
        export: bool = True,
    ) -> StockAnalysisResult:
        symbol = normalize_ticker(ticker)
        raw = download_stock_data(symbol, period=period, interval=interval)
        with_indicators = self.indicators.add_all_indicators(raw)
        analyzed = self.signals.generate_signals(with_indicators)
        macd_summary = self.signals.original_macd_summary(analyzed)

        backtester = StrategyBacktester(
            BacktestConfig(initial_capital=float(initial_capital))
        )
        backtest = backtester.run(analyzed)

        ml_result: MLResult | None = None
        if run_ml:
            try:
                ml_result = self.predictor.train(analyzed)
            except ValueError as exc:
                logger.warning("ML skipped for %s: %s", symbol, exc)

        result = StockAnalysisResult(
            ticker=symbol,
            data=analyzed,
            macd_summary=macd_summary,
            backtest=backtest,
            ml=ml_result,
            latest_snapshot=latest_price_snapshot(analyzed),
        )

        if export:
            result.export_paths = self.export_results(result)

        return result

    def compare_stocks(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Analyze several tickers and return a compact comparison table."""
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            try:
                result = self.analyze(
                    ticker,
                    period=period,
                    interval=interval,
                    run_ml=False,
                    export=False,
                )
                latest = result.data.iloc[-1]
                rows.append(
                    {
                        "Ticker": result.ticker,
                        "Close": latest["Close"],
                        "Daily_Return_%": latest["Daily_Return"] * 100,
                        "Trade_Signal": latest["Trade_Signal"],
                        "Signal_Score": latest["Signal_Score"],
                        "Original_MACD": result.macd_summary["latest_decision"],
                        "RSI": latest["RSI"],
                        "Trend": latest["Trend_Confirmed"],
                    }
                )
            except Exception as exc:
                rows.append({"Ticker": ticker.upper(), "Error": str(exc)})

        return pd.DataFrame(rows)

    def export_results(self, result: StockAnalysisResult) -> dict[str, Path]:
        """Write indicators, signals, predictions, backtest results, and charts."""
        symbol = safe_filename(result.ticker)
        paths: dict[str, Path] = {}

        paths["analysis_csv"] = export_dataframe(
            result.data, self.output_dir / f"{symbol}_indicators_signals.csv"
        )
        paths["backtest_equity_csv"] = export_dataframe(
            result.backtest.equity_curve,
            self.output_dir / f"{symbol}_backtest_equity.csv",
        )

        if not result.backtest.trades.empty:
            trade_path = self.output_dir / f"{symbol}_backtest_trades.csv"
            result.backtest.trades.to_csv(trade_path, index=False)
            paths["backtest_trades_csv"] = trade_path

        metrics_path = self.output_dir / f"{symbol}_backtest_metrics.csv"
        pd.DataFrame([result.backtest.metrics]).to_csv(metrics_path, index=False)
        paths["backtest_metrics_csv"] = metrics_path

        if result.ml is not None:
            paths["predictions_csv"] = export_dataframe(
                result.ml.prediction_frame,
                self.output_dir / f"{symbol}_ml_predictions.csv",
            )
            importance_path = self.output_dir / f"{symbol}_feature_importance.csv"
            result.ml.feature_importance.to_csv(importance_path, index=False)
            paths["feature_importance_csv"] = importance_path
            model_path = MODELS_DIR / f"{symbol}_{safe_filename(result.ml.model_name)}.joblib"
            self.predictor.save_model(result.ml, model_path)
            paths["model"] = model_path

        paths["macd_chart_png"] = self.save_original_macd_chart(result)
        return paths

    def save_original_macd_chart(self, result: StockAnalysisResult) -> Path:
        """Save a Matplotlib MACD chart inspired by the original project."""
        symbol = safe_filename(result.ticker)
        path = self.chart_dir / f"{symbol}_original_macd.png"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = result.data.dropna(subset=["Close", "MACD", "Signal_Line"])
        fig, (price_ax, macd_ax) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        price_ax.plot(data.index, data["Close"], label=f"{result.ticker} Close", color="black")
        price_ax.set_title(f"{result.ticker} Price and Original MACD System")
        price_ax.legend(loc="upper left")
        price_ax.grid(alpha=0.2)

        macd_ax.plot(data.index, data["MACD"], label="MACD", color="green", linewidth=1)
        macd_ax.plot(
            data.index,
            data["Signal_Line"],
            label="Signal Line",
            color="red",
            linewidth=1,
        )
        macd_ax.bar(
            data.index,
            data["MACD_Histogram"],
            label="Histogram",
            color=["#2ca02c" if value >= 0 else "#d62728" for value in data["MACD_Histogram"]],
            alpha=0.35,
        )
        macd_ax.legend(loc="upper left")
        macd_ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path
