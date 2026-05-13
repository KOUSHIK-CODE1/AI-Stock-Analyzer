# AI Stock Analyzer

An advanced stock analysis project built from the MACD crossover philosophy 
The original 12/26 EMA MACD calculation, 9-period signal line, `buy`/`sell`
decision logic, and recent-days mode summary are preserved, then extended with
technical indicators, machine learning, backtesting, CSV exports, and a
Streamlit dashboard.

> This project is for education and research only. It is not financial advice.

## Features

- Download OHLCV market data with `yfinance`
- Preserve the original MACD system
- Add RSI, SMA 20, SMA 50, EMA 12, EMA 26, Bollinger Bands, ATR, Stochastic RSI,
  OBV, ADI, volume ratios, momentum, returns, and volatility
- Generate `Strong Buy`, `Buy`, `Hold`, `Sell`, and `Strong Sell` signals
- Train ML models for next-day direction (Random Forest, Logistic Regression, XGBoost)
- Report model accuracy, confusion matrix, bullish probability, and latest trend
- Backtest a long-only trading strategy with comprehensive metrics
- Visualize price, MACD, RSI, predictions, and strategy performance with Plotly
- Export indicator values, signals, ML predictions, backtest trades, metrics, and charts
- Compare multiple tickers from the dashboard

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Use the sidebar to enter a ticker, period, interval, initial capital, and whether
to train ML models. Results are saved automatically.

## Run the CLI Pipeline

Analyze default stocks:

```bash
python main.py
```

Analyze selected tickers:

```bash
python main.py --tickers AAPL MSFT NVDA --period 5y --initial-capital 10000
```

Skip ML training for faster runs:

```bash
python main.py --tickers AAPL --period 2y --skip-ml
```

## Project Structure

```
AI-Stock-Analyzer/
├── app/
│   ├── analyzer.py      # Main analysis orchestrator
│   ├── indicators.py    # Technical indicators
│   ├── signals.py       # Signal generation
│   ├── ml_model.py      # ML model training
│   ├── backtest.py      # Backtesting engine
│   ├── dashboard.py     # Streamlit UI
│   └── utils.py         # Utility functions
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Output Files

After running, check these directories:
- `data/` - Raw OHLCV stock data
- `outputs/` - Indicators, signals, ML predictions, backtest metrics
- `charts/` - Generated Plotly/matplotlib visualizations
- `models/` - Saved ML model files
