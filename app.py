"""Streamlit entry point. Run with: streamlit run app.py."""

import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from fx_energy.analysis import export_report, forecast_table, run_analysis
from fx_energy.backtest import BacktestConfig, score_backtest
from fx_energy.charts import backtest_chart, forecast_chart, history_chart
from fx_energy.data import ASSETS, DataError, demo_data, download_yahoo, read_csv, validate_series
from fx_energy.models import MODEL_NAMES, ModelConfig

st.set_page_config(
    page_title="FX & Energy | Forecasting engine",
    page_icon=":material/monitoring:",
    layout="wide",
)


@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def cached_yahoo(asset: str, start: date, end: date):
    return download_yahoo(asset, start, end)


@st.cache_data(max_entries=4, show_spinner=False)
def cached_csv(content: bytes):
    return read_csv(content)


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def cached_analysis(prices: pd.Series, config: BacktestConfig, model_config: ModelConfig):
    return run_analysis(prices, config, model_config)


st.caption("RESEARCH LAB  /  FX + EUROPEAN ENERGY")
st.title("FX & Energy forecasting engine")
st.markdown(
    "EUR/PLN i TTF · **ARIMA / GARCH / XGBoost** · prognozy cen, ryzyko i uczciwy test poza proba"
)

with st.sidebar:
    st.header("Dane rynkowe", divider="gray")
    source = st.selectbox("Zrodlo danych", ["Demo", "Yahoo Finance", "CSV"], key="source")
    asset = st.selectbox(
        "Instrument", list(ASSETS), format_func=lambda x: ASSETS[x]["label"], key="asset"
    )
    st.caption(ASSETS[asset]["unit"])
    if source == "Yahoo Finance":
        with st.form("yahoo_form"):
            start = st.date_input("Od", date.today() - timedelta(days=5 * 365))
            end = st.date_input("Do (wylacznie)", date.today())
            fetch = st.form_submit_button("Pobierz notowania", width="stretch")
        if fetch:
            try:
                with st.spinner("Pobieranie danych Yahoo Finance..."):
                    st.session_state["yahoo_data"] = (
                        (asset, start, end),
                        cached_yahoo(asset, start, end),
                    )
            except (DataError, ValueError, OSError) as exc:
                st.session_state.pop("yahoo_data", None)
                st.error(str(exc))
        stored = st.session_state.get("yahoo_data")
        if stored is None or stored[0] != (asset, start, end):
            st.info("Ustaw zakres i pobierz notowania. Nie ma automatycznego przejscia na demo.")
            st.stop()
        market = stored[1]
    elif source == "CSV":
        uploaded = st.file_uploader("CSV: date, EURPLN i/lub TTF", type=["csv"])
        st.download_button(
            "Przyklad CSV (syntetyczny)",
            demo_data(300).prices.to_csv(index_label="date"),
            "synthetic_example.csv",
            "text/csv",
            width="stretch",
        )
        if uploaded is None:
            st.info("Wczytaj co najmniej 100 + horyzont obserwacji. Ceny musza byc dodatnie.")
            st.stop()
        try:
            market = cached_csv(uploaded.getvalue())
        except (DataError, ValueError) as exc:
            st.error(str(exc))
            st.stop()
    else:
        market = demo_data()
    st.divider()
    st.caption(
        "TTF: EUR/MWh, futures, nie cena spot. "
        "Dane Yahoo moga byc opoznione, niedostepne lub zawierac skoki przy zmianie kontraktu."
    )

if asset not in market.prices:
    st.error(f"Brak {asset} w danych. Dostepne kolumny: {', '.join(market.prices.columns)}.")
    st.stop()
try:
    prices = validate_series(market.prices[asset].dropna())
except DataError as exc:
    st.error(str(exc))
    st.stop()

unit = ASSETS[asset]["unit"]
if source == "Demo":
    st.warning("TRYB DEMO: dane syntetyczne. Wyniki nie opisuja rzeczywistego rynku.")
else:
    st.info(f"Zrodlo: {market.source}. Ostatnia obserwacja: {prices.index[-1]:%Y-%m-%d}.")
for note in market.notes:
    st.caption(note)
st.caption(
    f"Zakres: {prices.index[0]:%Y-%m-%d} - {prices.index[-1]:%Y-%m-%d} · "
    f"{len(prices):,} obserwacji · bez interpolacji i uzupelniania brakujacych sesji"
)

returns = np.log(prices).diff().dropna()
with st.container(horizontal=True):
    st.metric("Ostatnie Close", f"{prices.iloc[-1]:.4f}", border=True)
    st.metric("Zmiana ostatniej sesji", f"{100 * np.expm1(returns.iloc[-1]):+.2f}%", border=True)
    st.metric(
        "Zmiennosc roczna (hist.)",
        f"{returns.iloc[-60:].std() * np.sqrt(252) * 100:.1f}%",
        border=True,
        help="Odchylenie log-zwrotow z ostatnich 60 obserwacji x sqrt(252).",
    )
    st.metric("Instrument", ASSETS[asset]["label"], border=True)

st.plotly_chart(
    history_chart(prices, unit),
    width="stretch",
    alt=f"Historia dziennych cen {ASSETS[asset]['label']}, {unit}",
)

with st.form("analysis_form", border=True):
    st.subheader("Eksperyment prognostyczny")
    left, middle, right, last = st.columns(4)
    window = left.number_input("Okno treningowe (sesje)", 100, 2000, 252, step=20)
    horizon = middle.slider("Horyzont (sesje)", 1, 30, 5)
    stride = right.number_input("Krok przesuwania okna", 1, 100, 10)
    max_origins = last.number_input("Maks. liczba okien", 1, 100, 8)
    models = st.multiselect(
        "Modele (naiwny zawsze wlaczony)",
        ["ARIMA", "GARCH", "XGBoost"],
        default=["ARIMA", "GARCH", "XGBoost"],
    )
    a, b, c, d = st.columns(4)
    arima_p = a.number_input("ARIMA p", 0, 5, 1)
    arima_d = b.number_input("ARIMA d", 0, 2, 1)
    arima_q = c.number_input("ARIMA q", 0, 5, 1)
    confidence = d.selectbox("Poziom przedzialu", [0.8, 0.9, 0.95], index=2)
    st.caption(
        "Stale okno kroczace; ostatnie dostepne punkty startowe. "
        "Kazdy model jest trenowany od nowa w kazdym oknie. "
        "Wiecej okien i dluzszy horyzont zwiekszaja czas obliczen."
    )
    submitted = st.form_submit_button(
        "Uruchom analize", type="primary", width="stretch", key="run_analysis"
    )

config = BacktestConfig(
    window=int(window),
    horizon=int(horizon),
    stride=int(stride),
    max_origins=int(max_origins),
    models=tuple(name for name in MODEL_NAMES if name == "Naiwny" or name in models),
)
model_config = ModelConfig(
    arima_order=(int(arima_p), int(arima_d), int(arima_q)), confidence=float(confidence)
)
signature = hashlib.sha256(
    pd.util.hash_pandas_object(prices).values.tobytes()
    + repr((asset, market.source, config, model_config)).encode()
).hexdigest()
if submitted:
    if len(prices) < config.window + config.horizon:
        st.error(
            f"Potrzeba {config.window + config.horizon} obserwacji, dostepne {len(prices)}. "
            "Zmniejsz okno/horyzont albo dodaj dane."
        )
        st.stop()
    with st.spinner("Trening modeli i rolling-window backtest..."):
        analysis = cached_analysis(prices, config, model_config)
        st.session_state["analysis"] = (signature, analysis)

stored_analysis = st.session_state.get("analysis")
if stored_analysis is None or stored_analysis[0] != signature:
    st.info("Uruchom analize, aby zobaczyc prognozy i porownanie na aktualnych ustawieniach.")
    st.stop()
analysis = stored_analysis[1]
for model, message in analysis.failures.items():
    st.error(f"Prognoza biezaca / {model}: {message}")
if not analysis.backtest.failures.empty:
    st.warning(
        f"Nieudane dopasowania w backtescie: {len(analysis.backtest.failures)}. "
        "Wyniki porownujemy tylko na wspolnych, poprawnie obliczonych oknach."
    )

view = st.segmented_control(
    "Widok analizy",
    ["Prognoza", "Backtest", "Zmiennosc", "Dane i metodologia"],
    default="Prognoza",
    key="analysis_view",
)
if view == "Prognoza":
    st.subheader(f"Prognoza {ASSETS[asset]['label']} / {horizon} kolejnych sesji")
    st.caption(
        f"Trening na ostatnich {window} obserwacjach do {prices.index[-1]:%Y-%m-%d}. "
        "Linie to exp(prognozy log-ceny), czyli prognozy median, nie srednich cen. "
        "Nie przypisujemy przyszlym sesjom fikcyjnych dat kalendarzowych."
    )
    if analysis.forecasts:
        band_model = st.selectbox("Przedzial na wykresie", list(analysis.forecasts))
        st.plotly_chart(
            forecast_chart(prices, analysis, band_model, unit),
            width="stretch",
            alt="Prognozy cen oraz przedzial predykcyjny wybranego modelu",
        )
        st.caption(
            f"Nominalny poziom: {confidence:.0%}. {analysis.forecasts[band_model].interval_method}"
        )
        st.dataframe(
            forecast_table(analysis).drop(columns=["interval_method"]),
            hide_index=True,
            alt="Prognozy cen i granice przedzialow wedlug modelu i horyzontu",
        )
    else:
        st.error("Zaden model nie dostarczyl prognozy. Sprawdz komunikaty o dopasowaniu.")
elif view == "Backtest":
    st.subheader("Porownanie poza proba")
    step = st.select_slider("Oceniany horyzont t+h", range(1, horizon + 1), value=horizon)
    scores = score_backtest(analysis.backtest, step)
    st.caption(
        f"Przetestowano {analysis.backtest.origins} okien. "
        "N = liczba wspolnych punktow startowych dla wszystkich wybranych modeli. "
        "Nizsze MAE / RMSE / MAPE / MASE sa lepsze; dodatnie RMSE vs naiwny % oznacza poprawe."
    )
    if scores.empty:
        st.error("Brak wspolnych poprawnych prognoz. Sprawdz bledy lub zmien konfiguracje.")
    else:
        winner = scores.index[0]
        with st.container(horizontal=True):
            st.metric("Najnizsze RMSE w tej probie", winner, border=True)
            st.metric("RMSE", f"{scores.iloc[0]['RMSE']:.4f}", border=True)
            st.metric("Wspolne okna", str(int(scores.iloc[0]["N"])), border=True)
        st.dataframe(scores.round(4), alt="Ranking modeli na identycznych punktach testowych")
        st.plotly_chart(
            backtest_chart(analysis, int(step), unit),
            width="stretch",
            alt="Prognozy poza proba na tle rzeczywistych cen",
        )
        st.caption(
            "Kierunek porownuje znak zmiany wzgledem ostatniej ceny w oknie; "
            "model naiwny przewiduje brak zmiany. MASE uzywa sredniej bezwzglednej "
            "zmiany ceny w danym oknie treningowym (jednokrokowy mianownik). "
            "MASE nie jest dostepne przy zerowym mianowniku. "
            "Wybieranie zwyciezcy po tym samym backtescie nie stanowi niezaleznego testu."
        )
    if not analysis.backtest.failures.empty:
        st.dataframe(
            analysis.backtest.failures,
            hide_index=True,
            alt="Bledy dopasowania wedlug modelu i poczatku prognozy",
        )
elif view == "Zmiennosc":
    st.subheader("Ryzyko / warunkowa zmiennosc log-zwrotow")
    vol = {
        name: forecast.values.set_index("step")["sigma_return"] * 100
        for name, forecast in analysis.forecasts.items()
        if forecast.values["sigma_return"].notna().all()
    }
    if vol:
        st.line_chart(
            pd.DataFrame(vol),
            x_label="Sesja t+h",
            y_label="Odchylenie dziennego log-zwrotu (%)",
            alt="Prognoza dziennej zmiennosci GARCH i stala zmiennosc modelu naiwnego",
        )
    else:
        st.error("Brak poprawnych prognoz zmiennosci.")
    st.info(
        "GARCH(1,1) modeluje wariancje zwrotow, nie gwarantuje trafnego kierunku ceny. "
        "QLIKE porownuje prognoze dziennej wariancji z kwadratem pojedynczego "
        "log-zwrotu (szumny proxy, nie zrealizowana zmiennosc intraday). "
        "ARIMA i XGBoost nie dostarczaja tu prognozy dziennej wariancji."
    )
    vol_scores = score_backtest(analysis.backtest, 1)
    if not vol_scores.empty:
        st.dataframe(
            vol_scores[["N", "QLIKE"]].dropna(subset=["QLIKE"]),
            alt="Jednokrokowy QLIKE; nizsza wartosc jest lepsza",
        )
else:
    st.subheader("Metodologia i ograniczenia")
    st.markdown(
        """
- **ARIMA(p,d,q)**: log-ceny, bez trendu deterministycznego, parametry z formularza.
- **GARCH(1,1)**: procentowe log-zwroty, stala srednia, rozklad normalny.
  Przedzial wielokrokowy jest przyblizeniem; nie uwzglednia niepewnosci parametrow.
- **XGBoost direct**: osobny model na kazdy horyzont, opoznienia zwrotow,
  kroczace srednie, odchylenia i relatywne poziomy. Brak zewnetrznych zmiennych.
  Ostatnie 20% prob (min. 20) sluzy do chronologicznej kalibracji przedzialow;
  nakladajace sie etykiety sa wykluczane z czesci treningowej kalibratora.
  Model punktowy jest potem dopasowywany do calego okna.
- **Naiwny**: ostatnia cena; przedzialy losowego spaceru ze stala wariancja.
- **Rolling window**: wyklucza wszystkie dane od daty realizacji prognozy wzwyz.
  Brak skalowania lub doboru hiperparametrow na pelnym zbiorze. Dla uczciwego
  porownania metryki korzystaja z identycznych poprawnych punktow startowych.
- **Sesja** oznacza kolejna dostepna obserwacje danego instrumentu.
  Swieta, luki i zawieszenia moga zmieniac rzeczywisty odstep kalendarzowy.
- **Ograniczenia**: brak korekt rolowania TTF, kosztow transakcyjnych, spreadow
  i doboru kontraktu. Brak gwarancji pokrycia przedzialow i przewagi nad benchmarkiem.
  Nakladajace sie horyzonty daja zalezne bledy. Male N jest tylko demonstracja.
  Przed uzyciem produkcyjnym potrzebne sa licencjonowane dane, kontrola jakosci,
  tuning w zagniezdzonym backtescie i osobny, nietkniety okres testowy.
"""
    )
    st.dataframe(
        prices.rename(ASSETS[asset]["label"]).to_frame(),
        alt="Wejsciowe ceny po walidacji, bez uzupelniania brakow",
    )

st.divider()
st.download_button(
    "Pobierz raport ZIP",
    export_report(analysis, prices, asset, market.source, market.notes, config, model_config),
    f"forecast_{asset}_{prices.index[-1]:%Y%m%d}.zip",
    "application/zip",
    icon=":material/download:",
)
st.caption(
    "Raport: dane wejsciowe, prognozy, wszystkie horyzonty backtestu, bledy, konfiguracja "
    "i wersje bibliotek. Material badawczy; nie stanowi rekomendacji inwestycyjnej."
)
