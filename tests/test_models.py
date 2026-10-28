import numpy as np
import pandas as pd
import pytest

from fx_energy.data import demo_data
from fx_energy.models import (
    MODEL_NAMES,
    ModelConfig,
    ModelError,
    make_features,
    predict,
    supervised_data,
)


@pytest.mark.parametrize("asset", ["EURPLN", "TTF"])
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_all_models_produce_valid_forecasts(asset, model):
    prices = demo_data(280).prices[asset]
    forecast = predict(prices, model, 3, ModelConfig(n_estimators=30))
    frame = forecast.values
    assert list(frame["step"]) == [1, 2, 3]
    assert list(frame) == ["step", "forecast", "lower", "upper", "sigma_return"]
    assert np.isfinite(frame[["forecast", "lower", "upper"]]).all().all()
    assert (frame["lower"] > 0).all()
    assert (frame["lower"] <= frame["forecast"]).all()
    assert (frame["forecast"] <= frame["upper"]).all()
    assert forecast.interval_method
    if model in ("GARCH", "Naiwny"):
        assert (frame["sigma_return"] > 0).all()


def test_naive_exact_price_and_square_root_time_interval():
    prices = demo_data(200).prices["EURPLN"]
    frame = predict(prices, "Naiwny", 4).values
    np.testing.assert_allclose(frame["forecast"], prices.iloc[-1])
    spread = np.log(frame["upper"]) - np.log(frame["forecast"])
    np.testing.assert_allclose(spread.iloc[-1] / spread.iloc[0], 2)


def test_features_are_causal_and_targets_have_exact_horizon():
    prices = demo_data(200).prices["TTF"]
    changed = prices.copy()
    changed.iloc[130:] *= 10
    pd.testing.assert_frame_equal(
        make_features(prices).iloc[:130], make_features(changed).iloc[:130]
    )
    features, target = supervised_data(prices, 5)
    assert features.index[-1] == prices.index[-6]
    expected = np.log(prices.shift(-5) / prices).loc[target.index]
    np.testing.assert_allclose(target, expected)
    pd.testing.assert_series_equal(features.iloc[-1], make_features(prices.iloc[:-5]).iloc[-1])


def test_xgboost_is_reproducible():
    prices = demo_data(160).prices["TTF"]
    config = ModelConfig(n_estimators=20)
    pd.testing.assert_frame_equal(
        predict(prices, "XGBoost", 2, config).values,
        predict(prices, "XGBoost", 2, config).values,
    )


def test_garch_rejects_degenerate_returns():
    prices = pd.Series(4.3, index=pd.bdate_range("2025-01-01", periods=120))
    with pytest.raises(ModelError, match="niestalej"):
        predict(prices, "GARCH", 2)


def test_xgboost_rejects_insufficient_calibration_history():
    with pytest.raises(ModelError, match="za malo danych"):
        predict(demo_data(100).prices["TTF"], "XGBoost", 30)


@pytest.mark.parametrize("horizon", [0, 31, 1.5])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="Horyzont"):
        predict(demo_data(100).prices["TTF"], "ARIMA", horizon)
