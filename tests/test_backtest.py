import json
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from fx_energy.analysis import export_report, run_analysis
from fx_energy.backtest import BacktestConfig, rolling_backtest, score_backtest
from fx_energy.data import demo_data
from fx_energy.models import ModelConfig, ModelError, predict


def test_rolling_windows_have_exact_size_and_exclude_target(monkeypatch):
    prices = demo_data(150).prices["EURPLN"]
    observed = []

    def spy(train, model, horizon, config=None):
        observed.append(train.copy())
        return predict(train, model, horizon, config)

    monkeypatch.setattr("fx_energy.backtest.predict", spy)
    config = BacktestConfig(window=100, horizon=3, stride=7, max_origins=3, models=("Naiwny",))
    result = rolling_backtest(prices, config)
    stops = list(range(100, 148, 7))[-3:]
    assert result.origins == 3
    assert len(result.predictions) == 9
    for train, stop in zip(observed, stops, strict=True):
        pd.testing.assert_series_equal(train, prices.iloc[stop - 100 : stop])
    assert (result.predictions["origin"] < result.predictions["target_date"]).all()
    assert result.failures.empty


@pytest.mark.parametrize("asset", ["EURPLN", "TTF"])
def test_real_models_end_to_end_and_export(asset):
    market = demo_data(210)
    prices = market.prices[asset]
    config = BacktestConfig(window=160, horizon=2, stride=10, max_origins=2)
    model_config = ModelConfig(n_estimators=20)
    analysis = run_analysis(prices, config, model_config)
    assert analysis.backtest.failures.empty, analysis.backtest.failures.to_dict()
    assert not analysis.failures
    assert len(analysis.forecasts) == 4
    assert len(analysis.backtest.predictions) == 16
    scores = score_backtest(analysis.backtest, 2)
    assert scores["N"].eq(2).all()
    assert scores["RMSE"].notna().all()
    assert scores.loc["Naiwny", "RMSE vs naiwny %"] == 0
    with ZipFile(
        BytesIO(
            export_report(
                analysis, prices, asset, market.source, market.notes, config, model_config
            )
        )
    ) as report:
        metadata = json.loads(report.read("metadata.json"))
        assert metadata["asset"] == asset
        assert metadata["backtest"]["window"] == 160
        assert "xgboost" in metadata["versions"]
        exported_scores = pd.read_csv(report.open("metrics_by_horizon.csv"))
        assert len(exported_scores) == 8
        exported_prices = pd.read_csv(report.open("input_prices.csv"))
        assert list(exported_prices) == ["date", asset]
        assert len(exported_prices) == len(prices)


def test_failed_model_not_replaced_and_comparison_is_paired(monkeypatch):
    prices = demo_data(150).prices["EURPLN"]
    config = BacktestConfig(
        window=100, horizon=2, stride=10, max_origins=3, models=("Naiwny", "ARIMA")
    )
    failed_origin = prices.index[129]

    def fail_one(train, model, horizon, model_config=None):
        if model == "ARIMA" and train.index[-1] == failed_origin:
            raise ModelError("deliberate failure")
        return predict(train, "Naiwny", horizon, model_config)

    monkeypatch.setattr("fx_energy.backtest.predict", fail_one)
    result = rolling_backtest(prices, config)
    assert len(result.failures) == 1
    assert result.failures.iloc[0]["model"] == "ARIMA"
    scores = score_backtest(result, 2)
    assert scores["N"].eq(2).all()
    assert scores.loc["ARIMA", "RMSE"] == scores.loc["Naiwny", "RMSE"]


def test_metrics_match_hand_calculation():
    prices = pd.Series(np.arange(1.0, 151.0), index=pd.bdate_range("2025-01-01", periods=150))
    config = BacktestConfig(window=100, horizon=2, max_origins=3, models=("Naiwny",))
    result = rolling_backtest(prices, config)
    scores = score_backtest(result, 2)
    assert scores.loc["Naiwny", "MAE"] == pytest.approx(2)
    assert scores.loc["Naiwny", "RMSE"] == pytest.approx(2)
    assert scores.loc["Naiwny", "MASE"] == pytest.approx(2)
    assert scores.loc["Naiwny", "Kierunek %"] == 0


def test_changing_future_does_not_change_earlier_forecast():
    prices = demo_data(150).prices["TTF"]
    config = BacktestConfig(window=100, horizon=2, stride=10, max_origins=10, models=("Naiwny",))
    original = rolling_backtest(prices, config).predictions
    changed = prices.copy()
    changed.iloc[120:] *= 2
    altered = rolling_backtest(changed, config).predictions
    earlier = original["origin"] < prices.index[120]
    np.testing.assert_allclose(original.loc[earlier, "forecast"], altered.loc[earlier, "forecast"])
    np.testing.assert_allclose(original.loc[earlier, "lower"], altered.loc[earlier, "lower"])


def test_no_common_success_yields_no_scores(monkeypatch):
    def fail(*args, **kwargs):
        raise ModelError("failure")

    monkeypatch.setattr("fx_energy.backtest.predict", fail)
    config = BacktestConfig(window=100, horizon=1, max_origins=1)
    result = rolling_backtest(demo_data(150).prices["TTF"], config)
    assert result.predictions.empty
    assert len(result.failures) == 4
    assert score_backtest(result, 1).empty


def test_backtest_config_validation():
    with pytest.raises(ValueError, match="punktu odniesienia"):
        BacktestConfig(models=("ARIMA",))
    with pytest.raises(ValueError, match="Krok"):
        replace(BacktestConfig(), stride=0)


def test_arima_converges_for_small_fx_returns_in_default_windows():
    prices = demo_data().prices["EURPLN"]
    result = rolling_backtest(prices, BacktestConfig(models=("Naiwny", "ARIMA")))
    assert result.failures.empty, result.failures.to_dict()
    assert score_backtest(result, 5)["N"].eq(12).all()
