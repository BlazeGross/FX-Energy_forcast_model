# FX & Energy forecasting engine

Pythonowy silnik prognoz EUR/PLN i TTF: ARIMA, GARCH(1,1), XGBoost,
benchmark naiwny, backtesting rolling-window i dashboard Streamlit.

## Uruchomienie (Windows / PowerShell)

W katalogu projektu:

```powershell
uv sync --extra dev
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Alternatywnie bez `uv`, przy Pythonie 3.11-3.14:

```powershell
py -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Dashboard jest dostepny pod adresem http://localhost:8501.
W VS Code mozna tez uruchomic zadanie **Dashboard**.
Wybierz instrument, zrodlo i parametry, a nastepnie **Uruchom analize**.
Domyslnie aplikacja nie wykonuje zadnych zapytan sieciowych.

## Zrodla danych

- **Demo**: powtarzalne dane syntetyczne z grupowaniem zmiennosci, seed=42,
  1100 dni roboczych do 2026-10-23. Nie sa to historyczne notowania.
- **Yahoo Finance**: `EURPLN=X` (PLN za EUR), `TTF=F` (futures TTF, EUR/MWh).
  Dzienne `Close`, `auto_adjust=False`. Zakres konczy sie wylacznie przed data
  wskazana w polu **Do**. Zapytania sa cache'owane przez 15 minut.
  Dostepnosc zalezy od dostawcy; blad lub pusty wynik nie wlacza demo.
- **CSV**: kolumna `date` (ISO `YYYY-MM-DD`) i `EURPLN` i/lub `TTF`.
  Separator przecinek lub srednik; kropka dziesietna. W panelu dostepny jest
  poprawny przyklad CSV z danymi **syntetycznymi**.

Przyklad schematu (do analizy potrzeba wiecej wierszy):

```csv
date,EURPLN,TTF
2025-01-02,4.2800,48.20
2025-01-03,4.2710,47.85
```

CSV jest sortowany chronologicznie. Zduplikowane daty i nienumeryczne,
niedodatnie lub nieskonczone ceny sa odrzucane. Brakujace ceny sa pomijane
osobno dla kazdego instrumentu, z informacja o ich liczbie. Nie interpolujemy
ani nie wypelniamy cen na swieta. Podstawowa walidacja wymaga 80 cen; backtest
wymaga co najmniej `window + horizon` cen, przy `window >= 100`.
Dla XGBoost z dlugim horyzontem potrzebne jest wieksze okno do kalibracji.

TTF jest seria futures dostawcy, **nie spot**. Nie wykonujemy korekty
zmiany kontraktu ani rekonstrukcji szeregu constant-maturity. Skoki rolowania
moga pogarszac prognozy i zawyzac zmiennosc. Do zastosowan komercyjnych
sprawdz licencje danych; darmowe notowania nie zastepuja danych gieldowych.

## Modele i przedzialy

| Model | Zmienna modelowana | Prognoza i przedzial |
|---|---|---|
| Naiwny | Ostatnia log-cena | Ostatnia cena, normalny przedzial losowego spaceru |
| ARIMA(p,d,q) | Log-ceny, bez trendu | `exp()` prognozy log-ceny i jej przedzialu |
| GARCH(1,1) | Procentowe log-zwroty, stala srednia, rozklad normalny | Zmiennosc dziennych zwrotow; sciezka ceny z kumulacji srednich |
| XGBoost direct | Skumulowany przyszly log-zwrot, osobno dla kazdego h | Opoznione zwroty, kroczace srednie i odchylenia, poziom wzgledem sredniej |

Prognozy cen sa `exp(prognoza log-ceny)` (mediany w modelach
log-normalnych), **nie wartosciami oczekiwanymi ceny**.
ARIMA estymuje log-ceny pomnozone przez 100 dla stabilnosci numerycznej
przy malych zmianach FX; prognozy i odchylenia sa przeliczane z powrotem.
GARCH sluzy glownie do prognozy ryzyka; nie jest gwarancja trafnego kierunku.
Jego wielokrokowy przedzial jest przyblizeniem normalnym sumy log-zwrotow
i nie uwzglednia niepewnosci estymacji.

XGBoost kalibruje szerokosc przedzialu empirycznym kwantylem bezwzglednych
bledow na koncowych 20% prob w oknie (min. 20 prob). Trening kalibratora
wyklucza etykiety konczace sie po pierwszym punkcie kalibracyjnym (purging).
Model punktowy jest nastepnie dopasowywany do wszystkich dostepnych prob.
To heurystyczny przedzial po ponownym treningu, bez gwarancji pokrycia
conformal przy zaleznosciach czasowych lub zmianie rezimu.

## Uczciwy backtesting

Implementacja w [fx_energy/backtest.py](fx_energy/backtest.py):

1. Stale okno `window` obserwacji konczy sie przed pierwsza prognozowana cena.
2. Kazdy model jest ponownie trenowany, a cechy sa liczone tylko wewnatrz okna.
   Nie ma globalnego skalowania, dopasowania ani tuningu na przyszlych danych.
3. Przesuwamy poczatek prognozy o `stride` obserwacji. Ze wszystkich mozliwych
   punktow wybieramy ostatnie `max_origins` (domyslnie 8, dla szybkiego startu).
4. Zapisujemy wszystkie horyzonty `1..horizon`. Nieudane dopasowania sa
   raportowane, nie zastepowane prognoza innego modelu.
5. Metryki dla danego h liczymy na **identycznych, wspolnych punktach startowych**
   wszystkich wybranych modeli. Dlatego N moze byc mniejsze niz liczba okien.
   Wybor zestawu modeli moze zmienic probke wspolna.

**Metryki:** MAE, RMSE, MAPE, sMAPE, MASE, trafnosc kierunku, pokrycie
przedzialu, srednia szerokosc i poprawa RMSE wzgledem benchmarku.
MASE dzieli blad przez srednia bezwzgledna jednokrokowa zmiane ceny
**w danym oknie treningowym**; przy zerowym mianowniku jest niedostepne.
Kierunek to znak zmiany od ostatniej ceny w oknie (zero jest osobna klasa).
Przedzialy mierza niepewnosc przyszlej obserwacji, nie przedzial ufnosci sredniej.

QLIKE = `mean(log(sigma^2) + r^2 / sigma^2)` dla GARCH i benchmarku.
`r^2` jest szumnym proxy wariancji, nie zmiennoscia intraday.
Nizszy QLIKE jest lepszy; poziom moze byc ujemny. Wykres ryzyka prezentuje
nieannualizowane dzienne odchylenie log-zwrotow w procentach.
Historyczny KPI roczny uzywa 60 obserwacji i mnoznika `sqrt(252)`.

Horyzonty sa liczone w kolejnych **dostepnych obserwacjach**, a nie dniach
kalendarzowych. Nie zgadujemy dat przyszlych sesji. Raport nie jest strategia
transakcyjna: nie obejmuje spreadow, prowizji, roll yield ani P&L.
Nakladajace sie prognozy maja zalezne bledy; mala probka nie daje istotnosci
statystycznej. Ranking na tym samym backtescie nie jest niezaleznym holdoutem.
Wdrozenie wymaga osobnego nietknietego testu i zagniezdzonego tuningu.

## Eksport i uzycie w Pythonie

Przycisk **Pobierz raport ZIP** zapisuje dane wejsciowe, konfiguracje,
wersje bibliotek, biezace prognozy, wszystkie prognozy backtestu, metryki
dla kazdego h i bledy estymacji. Pliki pozostaja w pamieci sesji, dopoki
uzytkownik ich nie pobierze. Cache wynikow jest ograniczony do 4 analiz / 1 h.
Zmiana danych lub konfiguracji ukrywa nieaktualny wynik do ponownego uruchomienia.

```python
from fx_energy.data import demo_data
from fx_energy.analysis import run_analysis
from fx_energy.backtest import BacktestConfig, score_backtest
from fx_energy.models import ModelConfig

prices = demo_data().prices["EURPLN"]
analysis = run_analysis(
    prices,
    BacktestConfig(window=252, horizon=5, stride=10, max_origins=8),
    ModelConfig(arima_order=(1, 1, 1), confidence=0.95),
)
print(score_backtest(analysis.backtest, step=5))
print(analysis.forecasts["GARCH"].values)
```

## Testy

```powershell
& ".\.venv\Scripts\python.exe" -m pytest
& ".\.venv\Scripts\python.exe" -m ruff check .
```

Testy obejmuja oba instrumenty i wszystkie modele, izolacje czasowa cech
i okien, reczne wartosci metryk, obsluge bledow, import CSV, ZIP i widoki
dashboardu przez Streamlit AppTest. Testy nie wymagaja internetu.
Dokladne wersje zaleznosci zapisuje `uv.lock`.

Material badawczy i edukacyjny. Nie jest to rekomendacja inwestycyjna.
