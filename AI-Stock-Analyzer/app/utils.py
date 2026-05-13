"""Shared utilities for data loading, logging, paths, and exports."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = PROJECT_ROOT / "charts"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a project logger once and return it."""
    logger = logging.getLogger("ai_stock_analyzer")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = setup_logging()


def ensure_project_directories() -> dict[str, Path]:
    """Create the folders used by the dashboard, CLI, and exports."""
    directories = {
        "data": DATA_DIR,
        "outputs": OUTPUTS_DIR,
        "charts": CHARTS_DIR,
        "models": MODELS_DIR,
        "notebooks": NOTEBOOKS_DIR,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def normalize_ticker(ticker: str) -> str:
    """Return an uppercase ticker suitable for yfinance."""
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker symbol cannot be empty.")
    return cleaned


def safe_filename(value: str) -> str:
    """Build a simple filename-safe token from user-provided text."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("._") or "stock"


def flatten_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Handle the MultiIndex format yfinance sometimes returns."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()

    frame = data.copy()
    levels = list(range(frame.columns.nlevels))
    ticker_upper = ticker.upper()

    for level in levels:
        values = [str(value).upper() for value in frame.columns.get_level_values(level)]
        if ticker_upper in values:
            return frame.xs(ticker, axis=1, level=level, drop_level=True)

    frame.columns = frame.columns.get_level_values(0)
    return frame


def standardize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names from yfinance."""
    frame = data.copy()
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "volume": "Volume",
    }
    normalized = {
        column: rename_map.get(str(column).strip().lower(), str(column).strip())
        for column in frame.columns
    }
    frame = frame.rename(columns=normalized)

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}")

    if "Volume" not in frame.columns:
        frame["Volume"] = 0

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame = frame.sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    return frame


def download_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    save_raw: bool = True,
) -> pd.DataFrame:
    """Download stock data from yfinance and optionally cache the raw CSV."""
    ensure_project_directories()
    symbol = normalize_ticker(ticker)
    logger.info("Downloading %s data: period=%s interval=%s", symbol, period, interval)

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for ticker '{symbol}'.")

    frame = flatten_yfinance_columns(data, symbol)
    frame = standardize_ohlcv_columns(frame)

    if save_raw:
        export_dataframe(frame, DATA_DIR / f"{safe_filename(symbol)}_raw.csv")

    return frame


def export_dataframe(data: pd.DataFrame, path: Path) -> Path:
    """Save a DataFrame to CSV with a stable date index label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=True, index_label=data.index.name or "Date")
    return path


def export_records(records: list[dict[str, Any]], path: Path) -> Path:
    """Save a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def dataframe_to_csv_bytes(data: pd.DataFrame) -> bytes:
    """Convert a DataFrame to bytes for Streamlit download buttons."""
    return data.to_csv(index=True, index_label=data.index.name or "Date").encode("utf-8")


def latest_price_snapshot(data: pd.DataFrame) -> dict[str, float]:
    """Return common latest-price fields used by the dashboard."""
    if data.empty:
        return {"close": 0.0, "change": 0.0, "change_pct": 0.0}

    close = float(data["Close"].iloc[-1])
    previous = float(data["Close"].iloc[-2]) if len(data) > 1 else close
    change = close - previous
    change_pct = (change / previous * 100) if previous else 0.0
    return {"close": close, "change": change, "change_pct": change_pct}
