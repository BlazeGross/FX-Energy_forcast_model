from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_dashboard_analysis_and_all_views():
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert any("TRYB DEMO" in item.value for item in app.warning)
    assert len(app.metric) == 4
    app.slider[0].set_value(2)
    app.number_input[0].set_value(160)
    app.number_input[2].set_value(2)
    app.button(key="run_analysis").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe[0].value) == 8
    for view in ("Backtest", "Zmiennosc", "Dane i metodologia", "Prognoza"):
        app.segmented_control(key="analysis_view").set_value(view).run()
        assert not app.exception, view
        assert not app.error, view
    app.selectbox(key="asset").select("TTF").run()
    assert not app.exception
    assert any("aktualnych ustawieniach" in item.value for item in app.info)


def test_insufficient_history_is_visible():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.number_input[0].set_value(2000)
    app.button(key="run_analysis").click().run()
    assert not app.exception
    assert any("Potrzeba" in item.value for item in app.error)


def test_csv_source_requires_explicit_upload():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.selectbox(key="source").select("CSV").run()
    assert not app.exception
    assert not app.metric
    assert any("Wczytaj" in item.value for item in app.info)
