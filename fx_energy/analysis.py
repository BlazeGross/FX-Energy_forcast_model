"""Reusable analysis orchestration and reproducible downloadable reports."""

import json
import logging
from dataclasses import asdict, dataclass
from importlib.metadata import version
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from fx_energy.backtest import BacktestConfig, BacktestResult, rolling_backtest, score_backtest
from fx_energy.models import Forecast, ModelConfig, ModelError, predict

logger = logging.getLogger(__name__)


@dataclass
class Analysis:
    backtest: BacktestResult
    forecasts: dict[str, Forecast]
    failures: dict[str, str]


def run_analysis(
    prices: pd.Series, backtest_config: BacktestConfig, model_config: ModelConfig
) -> Analysis:
    backtest = rolling_backtest(prices, backtest_config, model_config)
    forecasts, failures = {}, {}
    for model in backtest_config.models:
        try:
            forecasts[model] = predict(
                prices.iloc[-backtest_config.window :], model, backtest_config.horizon, model_config
            )
        except (ModelError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            logger.warning("Biezaca prognoza / %s: %s", model, exc)
            failures[model] = str(exc)
    return Analysis(backtest, forecasts, failures)


def forecast_table(analysis: Analysis) -> pd.DataFrame:
    frames = [
        forecast.values.assign(model=name, interval_method=forecast.interval_method)
        for name, forecast in analysis.forecasts.items()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_report(
    analysis: Analysis,
    prices: pd.Series,
    asset: str,
    source: str,
    notes: tuple[str, ...],
    backtest_config: BacktestConfig,
    model_config: ModelConfig,
) -> bytes:
    scores = []
    for step in range(1, backtest_config.horizon + 1):
        frame = score_backtest(analysis.backtest, step)
        if not frame.empty:
            scores.append(frame.reset_index().assign(step=step))
    score_frame = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    metadata = {
        "asset": asset,
        "source": source,
        "source_notes": notes,
        "last_observation": prices.index[-1].isoformat(),
        "forecast_time_unit": "next observed trading session, not calendar day",
        "price_forecast_statistic": "conditional median / exp(predicted log price)",
        "backtest": asdict(backtest_config),
        "model": asdict(model_config),
        "latest_forecast_failures": analysis.failures,
        "versions": {
            name: version(name)
            for name in ("fx-energy-forecast", "numpy", "pandas", "statsmodels", "arch", "xgboost")
        },
    }
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for filename, frame in {
            "input_prices.csv": prices.rename(asset).to_frame(),
            "backtest_predictions.csv": analysis.backtest.predictions,
            "backtest_failures.csv": analysis.backtest.failures,
            "metrics_by_horizon.csv": score_frame,
            "forecast.csv": forecast_table(analysis),
        }.items():
            archive.writestr(
                filename,
                frame.to_csv(index=filename == "input_prices.csv", index_label="date"),
            )
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
    return buffer.getvalue()
