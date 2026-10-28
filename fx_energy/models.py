"""Price forecasts in log space and conditional return volatility."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from fx_energy.data import validate_series

MODEL_NAMES = ("Naiwny", "ARIMA", "GARCH", "XGBoost")


class ModelError(RuntimeError):
    """A model could not produce a valid forecast."""


@dataclass(frozen=True)
class ModelConfig:
    arima_order: tuple[int, int, int] = (1, 1, 1)
    confidence: float = 0.95
    n_estimators: int = 120
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1:
            raise ValueError("Poziom przedzialu musi nalezec do (0.5, 1).")
        if len(self.arima_order) != 3 or any(
            not isinstance(v, int) or v < 0 for v in self.arima_order
        ):
            raise ValueError("ARIMA wymaga trzech nieujemnych liczb calkowitych.")
        if self.n_estimators < 1:
            raise ValueError("Liczba drzew musi byc dodatnia.")


@dataclass
class Forecast:
    model: str
    values: pd.DataFrame
    interval_method: str


def make_features(prices: pd.Series) -> pd.DataFrame:
    log_prices = np.log(prices)
    returns = log_prices.diff()
    features = pd.DataFrame(index=prices.index)
    for lag in (0, 1, 2, 3, 4, 9, 19):
        features[f"return_lag_{lag}"] = returns.shift(lag)
    for window in (5, 10, 20):
        features[f"return_mean_{window}"] = returns.rolling(window).mean()
        features[f"return_std_{window}"] = returns.rolling(window).std()
        features[f"relative_level_{window}"] = log_prices - log_prices.rolling(window).mean()
    return features


def supervised_data(prices: pd.Series, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    if horizon < 1:
        raise ValueError("Horyzont musi byc dodatni.")
    features = make_features(prices)
    target = (np.log(prices.shift(-horizon)) - np.log(prices)).rename("target")
    joined = features.join(target).dropna()
    return joined.drop(columns="target"), joined["target"]


def _xgb(config: ModelConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=config.seed,
        n_jobs=1,
    )


def predict(
    prices: pd.Series,
    model: str,
    horizon: int,
    config: ModelConfig | None = None,
) -> Forecast:
    config = config or ModelConfig()
    prices = validate_series(prices)
    if model not in MODEL_NAMES:
        raise ValueError(f"Nieznany model: {model}.")
    if not isinstance(horizon, int) or not 1 <= horizon <= 30:
        raise ValueError("Horyzont musi byc liczba calkowita od 1 do 30.")
    log_prices = np.log(prices.to_numpy())
    log_returns = np.diff(log_prices)
    last_log = log_prices[-1]
    steps = np.arange(1, horizon + 1)
    z = NormalDist().inv_cdf((1 + config.confidence) / 2)
    sigma_return = np.full(horizon, np.nan)

    if model == "Naiwny":
        sigma_return[:] = np.std(log_returns, ddof=1)
        center = np.full(horizon, last_log)
        spread = z * sigma_return * np.sqrt(steps)
        method = "Losowy spacer: normalne log-zwroty, stala wariancja."
    elif model == "ARIMA":
        # Percentage-scale increments keep FX likelihood optimization well-conditioned.
        fitted = ARIMA(log_prices * 100, order=config.arima_order, trend="n").fit(
            method_kwargs={"maxiter": 300}
        )
        if not fitted.mle_retvals.get("converged", False):
            raise ModelError("ARIMA: optymalizacja nie osiagnela zbieznosci.")
        prediction = fitted.get_forecast(steps=horizon)
        center = np.asarray(prediction.predicted_mean) / 100
        spread = z * np.sqrt(np.asarray(prediction.var_pred_mean)) / 100
        method = "ARIMA: parametryczny przedzial w przestrzeni log-cen."
    elif model == "GARCH":
        if np.std(log_returns) < 1e-10:
            raise ModelError("GARCH wymaga niestalej serii zwrotow.")
        fitted = arch_model(
            log_returns * 100,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        ).fit(disp="off", show_warning=False)
        if fitted.convergence_flag != 0:
            raise ModelError(f"GARCH: brak zbieznosci: {fitted.optimization_result.message}")
        prediction = fitted.forecast(horizon=horizon, reindex=False)
        mean_returns = prediction.mean.iloc[-1].to_numpy() / 100
        variance = prediction.residual_variance.iloc[-1].to_numpy() / 10_000
        sigma_return = np.sqrt(variance)
        center = last_log + np.cumsum(mean_returns)
        spread = z * np.sqrt(np.cumsum(variance))
        method = "GARCH(1,1): przyblizenie normalne sumy log-zwrotow, bez niepewnosci parametrow."
    else:
        centers, spreads = [], []
        future_features = make_features(prices).iloc[[-1]]
        for step in steps:
            features, target = supervised_data(prices, int(step))
            calibration_size = max(20, len(features) // 5)
            split = len(features) - calibration_size
            # Purge overlapping labels before chronological interval calibration.
            train_end = split - int(step) + 1
            if train_end < 30:
                raise ModelError("XGBoost: za malo danych do treningu i kalibracji. Zwieksz okno.")
            regressor = _xgb(config)
            regressor.fit(features.iloc[:train_end], target.iloc[:train_end])
            residuals = np.abs(
                target.iloc[split:].to_numpy() - regressor.predict(features.iloc[split:])
            )
            radius = float(np.quantile(residuals, config.confidence, method="higher"))
            regressor.fit(features, target)
            centers.append(last_log + float(regressor.predict(future_features)[0]))
            spreads.append(radius)
        center, spread = np.asarray(centers), np.asarray(spreads)
        method = (
            "XGBoost direct: empiryczne kwantyle bledow na koncowym 20% okna "
            "(min. 20 prob), z purgingiem. Pokrycie nie jest gwarantowane."
        )

    with np.errstate(over="raise", invalid="raise"):
        values = pd.DataFrame(
            {
                "step": steps,
                "forecast": np.exp(center),
                "lower": np.exp(center - spread),
                "upper": np.exp(center + spread),
                "sigma_return": sigma_return,
            }
        )
    price_columns = values[["forecast", "lower", "upper"]]
    if (
        not np.isfinite(price_columns).all().all()
        or (price_columns <= 0).any().any()
        or (values["lower"] > values["forecast"]).any()
        or (values["upper"] < values["forecast"]).any()
    ):
        raise ModelError(f"{model}: niepoprawne wartosci prognozy.")
    return Forecast(model, values, method)
