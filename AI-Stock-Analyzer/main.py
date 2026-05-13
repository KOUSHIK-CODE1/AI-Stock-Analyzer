"""Command-line entry point for batch stock analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.analyzer import StockAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI-powered stock analysis.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "AMZN", "GOOG", "NFLX"],
        help="Ticker symbols to analyze.",
    )
    parser.add_argument("--period", default="5y", help="yfinance period, such as 1y or 5y.")
    parser.add_argument("--interval", default="1d", help="yfinance interval, such as 1d.")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_000.0,
        help="Starting cash for backtesting.",
    )
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = StockAnalyzer()

    print("AI Stock Analyzer")
    print("-----------------")

    for ticker in args.tickers:
        try:
            result = analyzer.analyze(
                ticker=ticker,
                period=args.period,
                interval=args.interval,
                initial_capital=args.initial_capital,
                run_ml=not args.skip_ml,
                export=True,
            )
            latest = result.data.iloc[-1]
            ml_text = (
                f"{result.ml.latest_prediction} ({result.ml.latest_probability * 100:.1f}%)"
                if result.ml
                else "skipped"
            )

            print(f"\n{result.ticker}")
            print(f"  Close: ${latest['Close']:.2f}")
            print(f"  Original MACD latest: {result.macd_summary['latest_decision']}")
            print(
                "  Original MACD mode "
                f"({result.macd_summary['lookback_days']}d): "
                f"{result.macd_summary['mode_decision']}"
            )
            print(f"  Smart signal: {latest['Trade_Signal']} ({latest['Signal_Score']:.0f})")
            print(f"  ML trend: {ml_text}")
            print(
                "  Backtest: "
                f"{result.backtest.metrics['total_return_pct']:.2f}% return, "
                f"{result.backtest.metrics['max_drawdown_pct']:.2f}% max drawdown"
            )
            print("  Exports:")
            for name, path in result.export_paths.items():
                print(f"    {name}: {Path(path)}")
        except Exception as exc:
            print(f"\n{ticker.upper()}: failed - {exc}")


if __name__ == "__main__":
    main()
