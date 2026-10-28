from datetime import date

import numpy as np
import pandas as pd
import pytest

from fx_energy.data import DataError, demo_data, download_yahoo, read_csv, validate_series


def test_demo_reproducible_and_explicit():
    first = demo_data(300)
    pd.testing.assert_frame_equal(first.prices, demo_data(300).prices)
    assert "DEMO" in first.source
    assert list(first.prices) == ["EURPLN", "TTF"]
    assert (first.prices > 0).all().all()
    for asset in first.prices:
        validate_series(first.prices[asset])


def test_csv_roundtrip_sorted_and_missing_sessions_not_filled():
    frame = demo_data(140).prices.iloc[::-1].copy()
    frame.iloc[2, 0] = np.nan
    result = read_csv(frame.to_csv(index_label="date").encode())
    assert result.prices.index.is_monotonic_increasing
    assert result.prices["EURPLN"].isna().sum() == 1
    assert len(result.prices["EURPLN"].dropna()) == 139
    assert result.notes


def test_csv_single_asset_semicolon():
    frame = demo_data(100).prices[["TTF"]]
    result = read_csv(frame.to_csv(index_label="date", sep=";").encode())
    assert list(result.prices) == ["TTF"]
    pd.testing.assert_frame_equal(result.prices, frame, check_freq=False)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), "bad"])
def test_csv_rejects_invalid_prices(bad):
    frame = demo_data(100).prices.astype(object)
    frame.iloc[5, 0] = bad
    with pytest.raises(DataError):
        read_csv(frame.to_csv(index_label="date").encode())


def test_csv_rejects_duplicate_dates_and_columns():
    frame = demo_data(100).prices
    duplicated = pd.concat([frame, frame.iloc[[-1]]])
    with pytest.raises(DataError, match="zduplikowane daty"):
        read_csv(duplicated.to_csv(index_label="date").encode())
    with pytest.raises(DataError, match="nazwy kolumn"):
        read_csv(b"date,EURPLN,eurpln\n2026-01-01,4.2,4.3\n")


def test_validate_rejects_unsorted_and_missing_prices():
    prices = demo_data(100).prices["EURPLN"]
    with pytest.raises(DataError, match="rosnaco"):
        validate_series(prices.iloc[::-1])
    prices.iloc[0] = np.nan
    with pytest.raises(DataError, match="bez brakow"):
        validate_series(prices)


def test_yahoo_multiindex_close_and_no_synthetic_fallback(monkeypatch):
    frame = demo_data(100).prices[["EURPLN"]]
    frame.columns = pd.MultiIndex.from_tuples([("Close", "EURPLN=X")])
    monkeypatch.setattr("fx_energy.data.yf.download", lambda *args, **kwargs: frame)
    result = download_yahoo("EURPLN", date(2025, 1, 1), date(2026, 1, 1))
    assert result.source == "Yahoo Finance / EURPLN=X"
    assert len(result.prices) == 100
    monkeypatch.setattr("fx_energy.data.yf.download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(DataError, match="Nie uzyto danych demo"):
        download_yahoo("TTF", date(2025, 1, 1), date(2026, 1, 1))
