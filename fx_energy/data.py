"""Explicit, validated market-data sources; never substitute synthetic data."""

from dataclasses import dataclass
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = {
    "EURPLN": {"label": "EUR/PLN", "unit": "PLN za EUR", "ticker": "EURPLN=X"},
    "TTF": {"label": "TTF futures", "unit": "EUR/MWh", "ticker": "TTF=F"},
}


class DataError(ValueError):
    """Unusable or unavailable source data."""


@dataclass
class MarketData:
    prices: pd.DataFrame
    source: str
    notes: tuple[str, ...] = ()


def validate_series(series: pd.Series, min_points: int = 80) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataError("Indeks cen musi zawierac daty.")
    if series.index.hasnans or series.index.has_duplicates:
        raise DataError("Daty nie moga byc puste ani zduplikowane.")
    if not series.index.is_monotonic_increasing:
        raise DataError("Daty musza byc uporzadkowane rosnaco.")
    values = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataError("Ceny musza byc skonczone, dodatnie i bez brakow.")
    if len(values) < min_points:
        raise DataError(f"Za malo obserwacji: {len(values)}; wymagane co najmniej {min_points}.")
    return values


def read_csv(content: bytes) -> MarketData:
    try:
        frame = pd.read_csv(BytesIO(content), sep=None, engine="python")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"Nie mozna odczytac CSV: {exc}") from exc
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    if frame.columns.has_duplicates:
        raise DataError("CSV zawiera zduplikowane nazwy kolumn.")
    if "DATE" not in frame or not any(asset in frame for asset in ASSETS):
        raise DataError("CSV wymaga kolumny date oraz EURPLN i/lub TTF.")
    try:
        dates = pd.to_datetime(frame.pop("DATE"), format="ISO8601", errors="raise")
        index = pd.DatetimeIndex(dates)
        if index.tz is not None:
            index = index.tz_localize(None)
        frame.index = index.normalize()
    except (ValueError, TypeError) as exc:
        raise DataError("Daty musza byc w formacie ISO, np. 2025-01-31.") from exc
    if frame.index.has_duplicates or frame.index.hasnans:
        raise DataError("CSV zawiera puste lub zduplikowane daty dzienne.")
    frame = frame.sort_index()
    prices = pd.DataFrame(index=frame.index)
    notes = []
    for asset in ASSETS:
        if asset not in frame:
            continue
        try:
            values = pd.to_numeric(frame[asset], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise DataError(f"{asset}: kolumna zawiera wartosci nienumeryczne.") from exc
        missing = int(values.isna().sum())
        validate_series(values.dropna())
        prices[asset] = values
        if missing:
            notes.append(f"{asset}: pominieto {missing} brakujacych cen; bez interpolacji.")
    prices.index.name = "date"
    return MarketData(prices, "CSV uzytkownika", tuple(notes))


def download_yahoo(asset: str, start: date, end: date) -> MarketData:
    if asset not in ASSETS:
        raise DataError(f"Nieznany instrument: {asset}.")
    if start >= end:
        raise DataError("Poczatek zakresu musi poprzedzac jego koniec.")
    ticker = ASSETS[asset]["ticker"]
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    if raw.empty:
        raise DataError(
            f"Yahoo nie zwrocilo danych dla {ticker}. Sprawdz zakres, limit zapytan "
            "i dostepnosc instrumentu lub wczytaj CSV. Nie uzyto danych demo."
        )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    missing = int(close.isna().sum())
    prices = validate_series(close.dropna()).rename(asset).to_frame()
    prices.index.name = "date"
    notes = [
        "Dane dzienne Close, bez korekty. Koniec zakresu jest wylaczny.",
        f"Pominieto {missing} pustych cen; bez uzupelniania swiat i sesji.",
    ]
    if asset == "TTF":
        notes.append(
            "TTF=F to seria futures dostawcy, nie spot. Zmiany kontraktu moga "
            "powodowac skoki; seria nie jest tu korygowana o rolowanie."
        )
    return MarketData(prices, f"Yahoo Finance / {ticker}", tuple(notes))


def demo_data(points: int = 1100, seed: int = 42) -> MarketData:
    if points < 80:
        raise DataError("Demo wymaga co najmniej 80 obserwacji.")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((points, 2))
    shocks[:, 1] = 0.2 * shocks[:, 0] + np.sqrt(0.96) * shocks[:, 1]
    prices = {}
    for column, asset, initial, base_sigma in [
        (0, "EURPLN", 4.35, 0.0035),
        (1, "TTF", 38.0, 0.035),
    ]:
        returns = np.zeros(points)
        variance = base_sigma**2
        for t in range(1, points):
            variance = 0.04 * base_sigma**2 + 0.08 * returns[t - 1] ** 2 + 0.88 * variance
            returns[t] = -0.06 * returns[t - 1] + np.sqrt(variance) * shocks[t, column]
        prices[asset] = initial * np.exp(np.cumsum(returns))
    index = pd.bdate_range(end="2026-10-23", periods=points, name="date")
    return MarketData(
        pd.DataFrame(prices, index=index),
        "DEMO / dane syntetyczne",
        (f"Dane symulowane, seed={seed}. Nie sa to historyczne notowania ani dane inwestycyjne.",),
    )
