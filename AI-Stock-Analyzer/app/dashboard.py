"""Streamlit dashboard for the AI Stock Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyzer import StockAnalyzer, StockAnalysisResult
from app.utils import dataframe_to_csv_bytes


def price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )
    for column, color in [("SMA_20", "#2563eb"), ("SMA_50", "#9333ea")]:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode="lines",
                name=column.replace("_", " "),
                line={"color": color, "width": 1.4},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_Upper"],
            mode="lines",
            name="BB Upper",
            line={"color": "#94a3b8", "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_Lower"],
            mode="lines",
            name="BB Lower",
            line={"color": "#94a3b8", "width": 1, "dash": "dot"},
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.12)",
        )
    )

    buys = data.dropna(subset=["Buy_Marker"])
    sells = data.dropna(subset=["Sell_Marker"])
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["Buy_Marker"],
            mode="markers",
            name="Buy",
            marker={"symbol": "triangle-up", "size": 11, "color": "#16a34a"},
            text=buys["Trade_Signal"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["Sell_Marker"],
            mode="markers",
            name="Sell",
            marker={"symbol": "triangle-down", "size": 11, "color": "#dc2626"},
            text=sells["Trade_Signal"],
        )
    )
    fig.update_layout(
        title=f"{ticker} Price",
        height=620,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )
    return fig


def macd_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    colors = np.where(data["MACD_Histogram"] >= 0, "#16a34a", "#dc2626")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35])
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
            line={"color": "#111827", "width": 1.3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["MACD_Histogram"],
            marker_color=colors,
            name="Histogram",
            opacity=0.45,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            mode="lines",
            name="MACD",
            line={"color": "#15803d", "width": 1.4},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Signal_Line"],
            mode="lines",
            name="Signal Line",
            line={"color": "#b91c1c", "width": 1.2},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"{ticker} Original MACD System",
        height=560,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend_orientation="h",
    )
    return fig


def rsi_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line={"color": "#2563eb", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["StochRSI"],
            mode="lines",
            name="Stoch RSI",
            line={"color": "#f97316", "width": 1.2},
        )
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626")
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a")
    fig.update_layout(
        title="Momentum",
        height=360,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        yaxis_range=[0, 100],
        legend_orientation="h",
    )
    return fig


def portfolio_chart(result: StockAnalysisResult) -> go.Figure:
    curve = result.backtest.equity_curve
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve["Portfolio_Value"],
            mode="lines",
            name="Strategy",
            line={"color": "#0f766e", "width": 2},
        )
    )
    benchmark = (
        result.data["Close"] / result.data["Close"].iloc[0]
    ) * result.backtest.metrics["initial_capital"]
    fig.add_trace(
        go.Scatter(
            x=result.data.index,
            y=benchmark,
            mode="lines",
            name="Buy and Hold",
            line={"color": "#64748b", "width": 1.4, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Portfolio Performance",
        height=420,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        legend_orientation="h",
    )
    return fig


def prediction_chart(result: StockAnalysisResult) -> go.Figure:
    frame = result.ml.prediction_frame.tail(120) if result.ml else pd.DataFrame()
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["Bullish_Probability"] * 100,
                mode="lines+markers",
                name="Bullish Probability",
                line={"color": "#7c3aed", "width": 1.6},
            )
        )
    fig.add_hline(y=50, line_dash="dash", line_color="#64748b")
    fig.update_layout(
        title="Prediction Confidence",
        height=360,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        yaxis_title="Probability %",
        yaxis_range=[0, 100],
    )
    return fig


def metric_value(value: float, suffix: str = "") -> str:
    return f"{value:,.2f}{suffix}"


st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("AI Stock Analyzer")

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=3)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    initial_capital = st.number_input(
        "Initial capital", min_value=1000.0, value=10_000.0, step=1000.0
    )
    run_ml = st.toggle("Train ML models", value=True)
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    st.divider()
    compare_symbols = st.text_input("Compare", value="MSFT, NVDA, AMZN")
    compare_clicked = st.button("Run comparison", use_container_width=True)

analyzer = StockAnalyzer()

if compare_clicked:
    symbols = [symbol.strip() for symbol in compare_symbols.split(",") if symbol.strip()]
    with st.spinner("Running comparison"):
        comparison = analyzer.compare_stocks(symbols, period=period, interval=interval)
    st.subheader("Multi-stock Comparison")
    st.dataframe(comparison, use_container_width=True)

if analyze_clicked or "analysis_result" not in st.session_state:
    with st.spinner("Analyzing market data"):
        try:
            st.session_state.analysis_result = analyzer.analyze(
                ticker=ticker,
                period=period,
                interval=interval,
                initial_capital=initial_capital,
                run_ml=run_ml,
                export=True,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

result: StockAnalysisResult = st.session_state.analysis_result
data = result.data
latest = data.iloc[-1]
snapshot = result.latest_snapshot

cols = st.columns(5)
cols[0].metric("Close", f"${snapshot['close']:,.2f}", f"{snapshot['change_pct']:.2f}%")
cols[1].metric("Smart Signal", str(latest["Trade_Signal"]), f"{latest['Signal_Score']:.0f}")
cols[2].metric(
    "Original MACD",
    result.macd_summary["latest_decision"].upper(),
    f"{result.macd_summary['lookback_days']}d mode: {result.macd_summary['mode_decision']}",
)
cols[3].metric("RSI", metric_value(float(latest["RSI"])))
if result.ml:
    cols[4].metric(
        "ML Trend",
        result.ml.latest_prediction,
        f"{result.ml.latest_probability * 100:.1f}%",
    )
else:
    cols[4].metric("ML Trend", "Skipped")

overview_tab, charts_tab, ml_tab, backtest_tab, exports_tab = st.tabs(
    ["Overview", "Charts", "ML", "Backtest", "Exports"]
)

with overview_tab:
    left, right = st.columns([0.64, 0.36])
    with left:
        st.plotly_chart(price_chart(data, result.ticker), use_container_width=True)
    with right:
        st.subheader("Latest Signal")
        st.write(
            pd.DataFrame(
                [
                    {
                        "Ticker": result.ticker,
                        "Close": latest["Close"],
                        "Signal": latest["Trade_Signal"],
                        "Score": latest["Signal_Score"],
                        "Reason": latest["Signal_Reason"],
                        "Trend": latest["Trend_Confirmed"],
                    }
                ]
            )
        )
        st.subheader("Recent Rows")
        st.dataframe(
            data[
                [
                    "Close",
                    "RSI",
                    "MACD",
                    "Signal_Line",
                    "Original_MACD_Decision",
                    "Trade_Signal",
                    "Signal_Reason",
                ]
            ].tail(12),
            use_container_width=True,
        )

with charts_tab:
    st.plotly_chart(macd_chart(data, result.ticker), use_container_width=True)
    st.plotly_chart(rsi_chart(data), use_container_width=True)

with ml_tab:
    if result.ml is None:
        st.info("ML was skipped. Use a longer period or enable ML training.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Best Model", result.ml.model_name)
        m2.metric("Accuracy", f"{result.ml.accuracy * 100:.2f}%")
        st.plotly_chart(prediction_chart(result), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Confusion Matrix")
            st.dataframe(result.ml.confusion_matrix, use_container_width=True)
        with c2:
            st.subheader("Feature Importance")
            st.dataframe(result.ml.feature_importance.head(12), use_container_width=True)

with backtest_tab:
    metrics = result.backtest.metrics
    bcols = st.columns(5)
    bcols[0].metric("Final Value", f"${metrics['final_value']:,.2f}")
    bcols[1].metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
    bcols[2].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    bcols[3].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    bcols[4].metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
    st.plotly_chart(portfolio_chart(result), use_container_width=True)
    st.subheader("Trades")
    st.dataframe(result.backtest.trades, use_container_width=True)

with exports_tab:
    st.subheader("Generated Files")
    export_table = pd.DataFrame(
        [{"Name": name, "Path": str(path)} for name, path in result.export_paths.items()]
    )
    st.dataframe(export_table, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download analysis CSV",
        dataframe_to_csv_bytes(result.data),
        file_name=f"{result.ticker}_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "Download backtest CSV",
        dataframe_to_csv_bytes(result.backtest.equity_curve),
        file_name=f"{result.ticker}_backtest.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if result.ml:
        d3.download_button(
            "Download predictions CSV",
            dataframe_to_csv_bytes(result.ml.prediction_frame),
            file_name=f"{result.ticker}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
