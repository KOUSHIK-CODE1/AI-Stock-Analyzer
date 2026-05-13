# AI Stock Analyzer

An advanced stock analysis project built from the MACD crossover philosophy 
The original 12/26 EMA MACD calculation, 9-period signal line, `buy`/`sell`
decision logic, and recent-days mode summary are preserved, then extended with
technical indicators, machine learning, backtesting, CSV exports, and a
Streamlit dashboard.

> This project is for education and research only. It is not financial advice.

## Features

- Download OHLCV market data with `yfinance`
- Preserve the original MACD system:
  - EMA 12 minus EMA 26
  - 9-period EMA signal line
  - `buy` when MACD is above signal, `sell` otherwise
  - bullish and bearish crossover detection
  - mode of recent MACD decisions
- Add RSI, SMA 20, SMA 50, EMA 12, EMA 26, Bollinger Bands, ATR, Stochastic RSI,
  OBV, ADI, volume ratios, momentum, returns, and volatility
- Generate `Strong Buy`, `Buy`, `Hold`, `Sell`, and `Strong Sell` signals
- Train ML models for next-day direction:
  - Random Forest
  - Logistic Regression
  - XGBoost when installed
- Report model accuracy, confusion matrix, bullish probability, and latest trend
- Backtest a long-only trading strategy with capital, fees, win rate, Sharpe
  ratio, max drawdown, and portfolio value
- Visualize price, MACD, RSI, predictions, and strategy performance with Plotly
- Export indicator values, signals, ML predictions, backtest trades, metrics, and
  charts
- Compare multiple tickers from the dashboard

## Folder Structure

```text
AI-Stock-Analyzer/
├── app/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── backtest.py
│   ├── dashboard.py
│   ├── indicators.py
│   ├── ml_model.py
│   ├── signals.py
│   └── utils.py
├── charts/
├── data/
├── models/
├── notebooks/
├── outputs/
├── main.py
├── README.md
└── requirements.txt
```

## Installation

Open VS Code in the project folder:

```bash
cd "C:\Users\koushik\OneDrive\Desktop\SNEAKER WEBSITE\AI-Stock-Analyzer(gpt)\AI-Stock-Analyzer"
code .
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
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
to train ML models. Results are saved automatically into `outputs/`, `charts/`,
`data/`, and `models/`.

## Run the CLI Pipeline

Analyze the default stock list:

```bash
python main.py
```

Analyze selected tickers:

```bash
python main.py --tickers AAPL MSFT NVDA --period 5y --initial-capital 10000
```

Skip ML training for a faster run:

```bash
python main.py --tickers AAPL --period 2y --skip-ml
```

## Example Outputs

After running the dashboard or CLI, the project generates files like:

```text
data/AAPL_raw.csv
outputs/AAPL_indicators_signals.csv
outputs/AAPL_ml_predictions.csv
outputs/AAPL_backtest_equity.csv
outputs/AAPL_backtest_trades.csv
outputs/AAPL_backtest_metrics.csv
outputs/AAPL_feature_importance.csv
charts/AAPL_original_macd.png
models/AAPL_Random_Forest.joblib
```

## Screenshots

Add dashboard screenshots here after your first run:

```text
screenshots/dashboard_overview.png
screenshots/macd_chart.png
screenshots/backtest_results.png
```

## Core Modules

- `app/indicators.py` keeps the original MACD math and adds advanced indicators.
- `app/signals.py` keeps the original MACD `buy`/`sell` decision and adds a
  multi-factor signal score.
- `app/ml_model.py` trains and evaluates classification models for next-day
  direction.
- `app/backtest.py` simulates a long-only trading strategy.
- `app/analyzer.py` orchestrates download, indicators, signals, ML, backtesting,
  exports, and chart generation.
- `app/dashboard.py` provides the interactive Streamlit interface.

## Project Explanation For Review

This project is an AI-powered stock analysis dashboard built in Python. It
downloads live and historical stock data, calculates technical indicators,
generates buy/sell signals, trains machine learning models to predict next-day
movement, backtests a trading strategy, and displays the results in a Streamlit
dashboard.

The project is based on the original `macd-stock-analyzer` repository. The core
MACD logic is preserved: compare the MACD line with the signal line and generate
buy/sell decisions from that relationship.

### Software Used

```text
Python          Main programming language
Streamlit       Web dashboard
yfinance        Downloads stock data from Yahoo Finance
pandas          DataFrame and table handling
numpy           Numerical calculations
matplotlib      Static MACD chart export
plotly          Interactive dashboard charts
scikit-learn    Machine learning models and metrics
ta              Technical analysis indicators
xgboost         Optional advanced ML classifier
joblib          Saves trained ML models
```

### Main Workflow

```text
1. User enters a stock ticker such as AAPL
2. yfinance downloads OHLCV stock data
3. Technical indicators are calculated
4. Buy/sell signals are generated
5. ML models predict next-day direction
6. Backtesting simulates trading performance
7. Charts and metrics are displayed in Streamlit
8. CSV files, charts, and model files are saved
```

OHLCV means:

```text
Open
High
Low
Close
Volume
```

### Original MACD Logic

The most important inherited part is the MACD system.

```text
EMA_12 = 12-period exponential moving average of Close
EMA_26 = 26-period exponential moving average of Close

MACD = EMA_12 - EMA_26
Signal Line = 9-period EMA of MACD
MACD Histogram = MACD - Signal Line
```

EMA formula:

```text
EMA_today = alpha * Price_today + (1 - alpha) * EMA_yesterday

alpha = 2 / (period + 1)
```

Original decision logic:

```text
If MACD > Signal Line:
    original decision = buy

If MACD < Signal Line:
    original decision = sell
```

Crossover logic:

```text
Bullish crossover:
MACD crosses above Signal Line

Bearish crossover:
MACD crosses below Signal Line
```

In simple terms, MACD measures momentum by comparing short-term and long-term
exponential moving averages. When MACD rises above the signal line, it indicates
bullish momentum. When MACD falls below the signal line, it indicates bearish
momentum.

### Technical Indicators

The project calculates:

```text
MACD
Signal Line
MACD Histogram
RSI
SMA 20
SMA 50
EMA 12
EMA 26
Bollinger Bands
ATR
Stochastic RSI
Volume SMA 20
Volume Ratio
OBV
ADI
Daily Returns
Momentum 5-day
Momentum 10-day
Volatility 20-day
```

Important indicator meanings:

```text
RSI:
Measures overbought or oversold conditions.
Above 70 usually means overbought.
Below 30 usually means oversold.

SMA 20 and SMA 50:
Simple moving averages used to detect trend direction.

Golden Cross:
SMA 20 crosses above SMA 50.
This is treated as bullish.

Death Cross:
SMA 20 crosses below SMA 50.
This is treated as bearish.

Bollinger Bands:
Middle band = SMA 20.
Upper band = SMA 20 + 2 standard deviations.
Lower band = SMA 20 - 2 standard deviations.

ATR:
Average True Range.
Measures volatility.

Stochastic RSI:
Shows RSI momentum on a 0 to 100 scale.

Volume Ratio:
Current volume compared to average volume.
Used to detect volume spikes.
```

### Smart Signal Engine

The project combines multiple indicators into a signal score. Final signal types
are:

```text
Strong Buy
Buy
Hold
Sell
Strong Sell
```

The signal engine considers:

```text
MACD above or below signal line
Bullish or bearish MACD crossover
RSI oversold or overbought
SMA golden cross or death cross
Trend confirmation
Volume spike
Price momentum
```

Example scoring behavior:

```text
Bullish MACD crossover = positive score
Bearish MACD crossover = negative score
RSI below 30 = bullish score
RSI above 70 = bearish score
Golden cross = bullish score
Death cross = bearish score
Uptrend = bullish score
Downtrend = bearish score
```

### Machine Learning

The ML module predicts whether the stock will move up or down on the next
trading day.

Target variable:

```text
If tomorrow's Close > today's Close:
    Target = 1, Bullish

Else:
    Target = 0, Bearish
```

Models used:

```text
RandomForestClassifier
LogisticRegression
XGBoost, if installed
```

The project trains the available models and picks the best one based on accuracy.

Features used by ML:

```text
RSI
MACD
Signal Line
MACD Histogram
Volume
EMA difference
Momentum
Daily returns
SMA 20
SMA 50
ATR
Stochastic RSI
Volume Ratio
Volatility
Close compared to SMA 20
Close compared to SMA 50
```

ML outputs:

```text
Best model name
Accuracy score
Confusion matrix
Bullish probability
Bearish or bullish prediction
Feature importance
Saved model file
```

In review terms: the machine learning system converts technical indicators into
features and tries to classify whether the next day will be bullish or bearish.
It uses historical data, trains models chronologically, and reports accuracy and
prediction confidence.

### Backtesting

Backtesting checks how the strategy would have performed historically.

The simulator:

```text
Starts with initial capital, for example $10,000
Buys when signal is Buy or Strong Buy
Sells when signal is Sell or Strong Sell
Tracks cash, shares, fees, and portfolio value
```

Metrics calculated:

```text
Final portfolio value
Total return
Annualized return
Benchmark return
Sharpe ratio
Maximum drawdown
Trade count
Winning trades
Win rate
```

Important formulas:

```text
Total Return = Final Value / Initial Capital - 1

Drawdown = Portfolio Value / Previous Peak - 1

Sharpe Ratio = Mean Daily Return / Standard Deviation of Daily Return * sqrt(252)
```

Backtesting allows the project to test the generated strategy on historical data
before trusting it as an analysis signal.

### Dashboard Charts

The Streamlit dashboard includes:

```text
Candlestick chart
SMA 20 and SMA 50 overlays
Bollinger Bands
Buy/sell markers
MACD chart
MACD histogram
RSI chart
Stochastic RSI chart
Portfolio performance chart
Prediction confidence chart
Feature importance table
Confusion matrix
```

Chart meanings:

```text
Candlestick:
Shows open, high, low, and close price movement.

MACD chart:
Shows momentum and crossover behavior.

MACD histogram:
Shows the distance between MACD and the signal line.

RSI chart:
Shows overbought and oversold zones.

Portfolio chart:
Compares strategy performance with buy-and-hold.

Prediction confidence chart:
Shows ML bullish probability over time.
```

### Short Explanation

This project is an AI-powered stock analysis dashboard. It downloads stock data
using `yfinance`, calculates technical indicators like MACD, RSI, moving
averages, Bollinger Bands, ATR, and volume indicators, then generates smart
buy/sell signals. It also trains machine learning models like Random Forest,
Logistic Regression, and XGBoost to predict next-day stock direction. Finally,
it backtests the strategy and visualizes everything using Streamlit and Plotly.

## Future Improvements

- Add optional LSTM forecasting when TensorFlow is available
- Add news sentiment scoring from a market news API
- Add Telegram or Discord alerts for new strong signals
- Add paper-trading broker integration
- Add portfolio-level risk controls and position sizing
