"""Fixed-size rolling windows, paired model comparisons and explicit failures."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fx_energy.data import validate_series
from fx_energy.models import MODEL_NAMES, ModelConfig, ModelError, predict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    window: int = 252
    horizon: int = 5
    stride: int = 10
    max_origins: int = 12
    models: tuple[str, ...] = MODEL_NAMES

    def __post_init__(self) -> None:
        if self.window < 100 or not 1 <= self.horizon <= 30:
            raise ValueError("Okno musi miec >=100 sesji, a horyzont od 1 do 30.")
        if self.stride < 1 or self.max_origins < 1:
            raise ValueError("Krok i liczba okien musza byc dodatnie.")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Wybierz co najmniej jeden model, bez powtorzen.")
        if any(model not in MODEL_NAMES for model in self.models):
            raise ValueError("Nieznany model w konfiguracji backtestu.")
        if "Naiwny" not in self.models:
            raise ValueError("Backtest wymaga modelu naiwnego jako punktu odniesienia.")


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    failures: pd.DataFrame
    origins: int
    models: tuple[str, ...]


def rolling_backtest(
    prices: pd.Series,
    config: BacktestConfig,
    model_config: ModelConfig | None = None,
) -> BacktestResult:
    prices = validate_series(prices, config.window + config.horizon)
    origins = list(range(config.window, len(prices) - config.horizon + 1, config.stride))
    origins = origins[-config.max_origins :]
    records, failures = [], []
    for stop in origins:
        train = prices.iloc[stop - config.window : stop]
        origin = train.index[-1]
        actual = prices.iloc[stop : stop + config.horizon]
        scale = float(np.abs(np.diff(train.to_numpy())).mean())
        for model in config.models:
            try:
                forecast = predict(train, model, config.horizon, model_config)
            except (ModelError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                logger.warning("%s / %s: %s", origin, model, exc)
                failures.append({"origin": origin, "model": model, "error": str(exc)})
                continue
            for i, row in forecast.values.iterrows():
                records.append(
                    {
                        "origin": origin,
                        "train_start": train.index[0],
                        "target_date": actual.index[i],
                        "model": model,
                        "step": int(row["step"]),
                        "actual": float(actual.iloc[i]),
                        "last_price": float(train.iloc[-1]),
                        "forecast": float(row["forecast"]),
                        "lower": float(row["lower"]),
                        "upper": float(row["upper"]),
                        "sigma_return": float(row["sigma_return"]),
                        "actual_return": float(
                            np.log(prices.iloc[stop + i] / prices.iloc[stop + i - 1])
                        ),
                        "mase_scale": scale,
                    }
                )
    return BacktestResult(
        pd.DataFrame.from_records(records),
        pd.DataFrame(failures, columns=["origin", "model", "error"]),
        len(origins),
        config.models,
    )


def paired_predictions(result: BacktestResult, step: int) -> pd.DataFrame:
    if result.predictions.empty:
        return result.predictions.copy()
    frame = result.predictions.loc[result.predictions["step"] == step].copy()
    counts = frame.groupby("origin")["model"].nunique()
    common_origins = counts[counts == len(result.models)].index
    return frame.loc[frame["origin"].isin(common_origins)].copy()


def score_backtest(result: BacktestResult, step: int) -> pd.DataFrame:
    frame = paired_predictions(result, step)
    if frame.empty:
        return pd.DataFrame()
    metrics = []
    for model, group in frame.groupby("model", sort=False):
        error = group["forecast"] - group["actual"]
        scale = group["mase_scale"].where(group["mase_scale"] > 0)
        actual_direction = np.sign(group["actual"] - group["last_price"])
        predicted_direction = np.sign(group["forecast"] - group["last_price"])
        sigma2 = group["sigma_return"] ** 2
        valid_vol = sigma2.notna() & (sigma2 > 0)
        # Squared daily returns are a noisy proxy, not observed realized volatility.
        qlike = (
            float(
                (
                    np.log(sigma2[valid_vol])
                    + group.loc[valid_vol, "actual_return"] ** 2 / sigma2[valid_vol]
                ).mean()
            )
            if valid_vol.all()
            else np.nan
        )
        metrics.append(
            {
                "model": model,
                "N": len(group),
                "MAE": float(error.abs().mean()),
                "RMSE": float(np.sqrt((error**2).mean())),
                "MAPE %": float((error.abs() / group["actual"]).mean() * 100),
                "sMAPE %": float(
                    (2 * error.abs() / (group["actual"] + group["forecast"])).mean() * 100
                ),
                "MASE": float((error.abs() / scale).mean()),
                "Kierunek %": float((actual_direction == predicted_direction).mean() * 100),
                "Pokrycie %": float(
                    group["actual"].between(group["lower"], group["upper"]).mean() * 100
                ),
                "Szerokosc": float((group["upper"] - group["lower"]).mean()),
                "QLIKE": qlike,
            }
        )
    scores = pd.DataFrame(metrics).set_index("model")
    baseline = scores.loc["Naiwny", "RMSE"]
    scores["RMSE vs naiwny %"] = (1 - scores["RMSE"] / baseline) * 100 if baseline > 0 else np.nan
    return scores.sort_values("RMSE")
