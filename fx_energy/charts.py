"""Plotly figures with explicit units and observation-based horizons."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fx_energy.analysis import Analysis
from fx_energy.backtest import paired_predictions

COLORS = {"Naiwny": "#94A3B8", "ARIMA": "#60A5FA", "GARCH": "#FBBF24", "XGBoost": "#2DD4BF"}


def _style(figure: go.Figure, ylabel: str, xlabel: str = "") -> go.Figure:
    figure.update_layout(
        height=420,
        margin=dict(l=16, r=16, t=24, b=16),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def history_chart(prices: pd.Series, unit: str) -> go.Figure:
    figure = go.Figure(
        go.Scatter(x=prices.index, y=prices, name="Close", line=dict(color="#2DD4BF", width=2))
    )
    return _style(figure, unit)


def forecast_chart(prices: pd.Series, analysis: Analysis, band_model: str, unit: str) -> go.Figure:
    history = prices.iloc[-60:]
    figure = go.Figure(
        go.Scatter(
            x=np.arange(-len(history) + 1, 1),
            y=history,
            name="Historia",
            line=dict(color="#E2E8F0"),
        )
    )
    band = analysis.forecasts[band_model].values
    figure.add_trace(
        go.Scatter(
            x=band["step"], y=band["upper"], mode="lines", line=dict(width=0), showlegend=False
        )
    )
    figure.add_trace(
        go.Scatter(
            x=band["step"],
            y=band["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(96,165,250,0.18)",
            name=f"Przedzial: {band_model}",
        )
    )
    for name, forecast in analysis.forecasts.items():
        values = forecast.values
        figure.add_trace(
            go.Scatter(
                x=[0, *values["step"]],
                y=[prices.iloc[-1], *values["forecast"]],
                name=name,
                mode="lines+markers",
                line=dict(color=COLORS[name], width=2),
            )
        )
    figure.add_vline(x=0, line_dash="dot", line_color="#64748B")
    return _style(figure, unit, "Sesje wzgledem ostatniej obserwacji (0)")


def backtest_chart(analysis: Analysis, step: int, unit: str) -> go.Figure:
    frame = paired_predictions(analysis.backtest, step)
    figure = go.Figure()
    if frame.empty:
        return _style(figure, unit)
    actual = frame.drop_duplicates("target_date").sort_values("target_date")
    figure.add_trace(
        go.Scatter(
            x=actual["target_date"],
            y=actual["actual"],
            name="Rzeczywiste",
            line=dict(color="#E2E8F0", width=3),
        )
    )
    for name, group in frame.groupby("model", sort=False):
        figure.add_trace(
            go.Scatter(
                x=group["target_date"],
                y=group["forecast"],
                name=name,
                mode="lines+markers",
                line=dict(color=COLORS[name], dash="dot"),
            )
        )
    return _style(figure, unit, f"Data realizacji prognozy t+{step}")
