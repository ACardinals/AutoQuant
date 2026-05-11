from datetime import datetime, timedelta

from market_monitor.dashboard import (
    backtest_symbol,
    candles_to_chart_rows,
    compare_symbol_strategies,
    compare_watchlist_strategies,
    create_candlestick_figure,
    create_full_a_share_watchlist,
    download_symbols_from_watchlist,
    ensure_default_watchlist,
    evaluate_symbol_ml,
    filter_screen_rows,
    filter_watchlist_rows,
    flatten_screen_row,
    load_available_watchlist_candles,
    rank_watchlist_ml_candidates,
    screen_rows,
    write_filtered_watchlist,
)
from market_monitor.models import Candle


def _write_candles(path, symbol, count=80):
    rows = ["timestamp,symbol,open,high,low,close,volume"]
    start = datetime(2024, 1, 1)
    for i in range(count):
        cycle = i % 12
        price = 100 + i * 0.08 + (cycle - 6) * 0.25
        day = start + timedelta(days=i)
        rows.append(f"{day:%Y-%m-%d}T00:00:00Z,{symbol},{price},{price + 1},{price - 1},{price},{100 + i * 3}")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_candles_to_chart_rows():
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1),
            symbol="TEST",
            open=1,
            high=2,
            low=0.5,
            close=1.5,
            volume=100,
        )
    ]

    rows = candles_to_chart_rows(candles)

    assert rows[0]["timestamp"] == datetime(2024, 1, 1)
    assert rows[0]["close"] == 1.5


def test_screen_rows_returns_metadata(tmp_path):
    csv_path = tmp_path / "test.csv"
    _write_candles(csv_path, "TEST")
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nTEST,Test Asset,unit,test.csv\n",
        encoding="utf-8",
    )

    rows = screen_rows(watchlist_path, "ma_trend")

    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["name"] == "Test Asset"
    assert rows[0]["market"] == "unit"


def test_backtest_symbol_returns_chart_and_backtest(tmp_path):
    csv_path = tmp_path / "test.csv"
    _write_candles(csv_path, "TEST")
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nTEST,Test Asset,unit,test.csv\n",
        encoding="utf-8",
    )

    details = backtest_symbol(watchlist_path, "TEST", "ma_trend", 10_000)

    assert details["symbol"] == "TEST"
    assert details["candles"]
    assert "metrics" in details["backtest"]
    assert "equity_curve" in details["backtest"]


def test_flatten_screen_row_extracts_risk_and_reason():
    row = flatten_screen_row(
        {
            "symbol": "TEST",
            "signal": "buy_candidate",
            "confidence": 0.9,
            "reasons": ["first", "second"],
            "risk": {"stop_loss": 9.5, "take_profit": 11, "max_position_pct": 0.1},
        }
    )

    assert row["stop_loss"] == 9.5
    assert row["take_profit"] == 11
    assert row["reason"] == "first"


def test_ensure_default_watchlist_creates_curated_symbols(tmp_path):
    watchlist_path = tmp_path / "a_share.csv"

    summary = ensure_default_watchlist(watchlist_path, "data/a_share")

    assert summary["count"] >= 10
    assert watchlist_path.exists()
    assert "600519.SH" in watchlist_path.read_text(encoding="utf-8")


def test_filter_screen_rows():
    rows = [
        {"symbol": "A", "signal": "buy_candidate", "confidence": 0.8},
        {"symbol": "B", "signal": "hold", "confidence": 0.4},
    ]

    assert [row["symbol"] for row in filter_screen_rows(rows, "buy_candidate", 0.5)] == ["A"]
    assert [row["symbol"] for row in filter_screen_rows(rows, "all", 0.5)] == ["A"]


def test_create_candlestick_figure():
    rows = [
        {"timestamp": datetime(2024, 1, 1), "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
        {"timestamp": datetime(2024, 1, 2), "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 1200},
    ]

    figure = create_candlestick_figure(rows, "TEST")

    assert len(figure.data) == 2
    assert figure.data[0].type == "candlestick"


def test_create_full_a_share_watchlist_from_mocked_universe(tmp_path, monkeypatch):
    output = tmp_path / "all.csv"

    def fake_universe():
        from market_monitor.data.watchlist import WatchlistItem
        from pathlib import Path

        return [
            WatchlistItem(symbol="000001.SZ", name="平安银行", market="A股", csv_path=Path("000001.SZ.csv")),
            WatchlistItem(symbol="600519.SH", name="贵州茅台", market="A股", csv_path=Path("600519.SH.csv")),
        ]

    monkeypatch.setattr("market_monitor.dashboard.fetch_a_share_spot_universe", fake_universe)

    summary = create_full_a_share_watchlist(output, "data/a_share")

    assert summary["count"] == 2
    text = output.read_text(encoding="utf-8")
    assert "000001.SZ" in text
    assert "贵州茅台" in text


def test_load_available_watchlist_candles_skips_missing_csv(tmp_path):
    csv_path = tmp_path / "available.csv"
    _write_candles(csv_path, "HAVE")
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\n"
        "HAVE,Available,unit,available.csv\n"
        "MISS,Missing,unit,missing.csv\n",
        encoding="utf-8",
    )

    items, candles_by_symbol, missing_items = load_available_watchlist_candles(watchlist_path)

    assert [item.symbol for item in items] == ["HAVE"]
    assert "HAVE" in candles_by_symbol
    assert [item.symbol for item in missing_items] == ["MISS"]


def test_create_full_a_share_watchlist_respects_limit(tmp_path, monkeypatch):
    output = tmp_path / "limited.csv"

    def fake_universe():
        from market_monitor.data.watchlist import WatchlistItem
        from pathlib import Path

        return [
            WatchlistItem(symbol="000001.SZ", name="平安银行", market="A股", csv_path=Path("000001.SZ.csv")),
            WatchlistItem(symbol="600519.SH", name="贵州茅台", market="A股", csv_path=Path("600519.SH.csv")),
        ]

    monkeypatch.setattr("market_monitor.dashboard.fetch_a_share_spot_universe", fake_universe)

    summary = create_full_a_share_watchlist(output, "data/a_share", limit=1)

    assert summary["count"] == 1
    text = output.read_text(encoding="utf-8")
    assert "000001.SZ" in text


def test_filter_watchlist_rows_combines_search_market_signal_and_confidence():
    rows = [
        {"symbol": "000001.SZ", "name": "平安银行", "market": "银行", "signal": "buy_candidate", "confidence": 0.8},
        {"symbol": "600519.SH", "name": "贵州茅台", "market": "白酒", "signal": "hold", "confidence": 0.4},
    ]

    filtered = filter_watchlist_rows(rows, "平安", "银行", "buy_candidate", 0.5)

    assert [row["symbol"] for row in filtered] == ["000001.SZ"]


def test_write_filtered_watchlist(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    output = tmp_path / "filtered.csv"
    source.write_text(
        "symbol,name,market,csv\n"
        "000001.SZ,平安银行,银行,000001.SZ.csv\n"
        "600519.SH,贵州茅台,白酒,600519.SH.csv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("market_monitor.dashboard.resolve_dashboard_path", lambda path: path)

    summary = write_filtered_watchlist(source, output, ["600519.SH"])

    assert summary["count"] == 1
    text = output.read_text(encoding="utf-8")
    assert "600519.SH" in text
    assert "000001.SZ" not in text


def test_compare_symbol_strategies_returns_ranked_rows(tmp_path):
    csv_path = tmp_path / "test.csv"
    _write_candles(csv_path, "TEST", count=90)
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nTEST,Test Asset,unit,test.csv\n",
        encoding="utf-8",
    )

    rows = compare_symbol_strategies(watchlist_path, "TEST", 10_000)

    assert rows
    assert "score" in rows[0]
    assert rows[0]["score"] >= rows[-1]["score"]


def test_compare_watchlist_strategies_returns_metadata(tmp_path):
    csv_path = tmp_path / "test.csv"
    _write_candles(csv_path, "TEST", count=90)
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nTEST,Test Asset,unit,test.csv\n",
        encoding="utf-8",
    )

    rows = compare_watchlist_strategies(watchlist_path, 10_000, top_n=3)

    assert rows
    assert len(rows) <= 3
    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["name"] == "Test Asset"


def test_evaluate_symbol_ml_returns_metrics(tmp_path):
    csv_path = tmp_path / "test.csv"
    _write_candles(csv_path, "TEST", count=90)
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nTEST,Test Asset,unit,test.csv\n",
        encoding="utf-8",
    )

    result = evaluate_symbol_ml(watchlist_path, "TEST", "logistic_regression", horizon=5, splits=3, threshold=0)

    assert result["model"] == "logistic_regression"
    assert result["folds"]
    assert "accuracy" in result["metrics"]


def test_rank_watchlist_ml_candidates_returns_candidates(tmp_path):
    csv_path = tmp_path / "test.csv"
    _write_candles(csv_path, "TEST", count=90)
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nTEST,Test Asset,unit,test.csv\n",
        encoding="utf-8",
    )

    rows = rank_watchlist_ml_candidates(watchlist_path, "logistic_regression", horizon=5, threshold=0, top_n=5)

    assert rows
    assert rows[0]["symbol"] == "TEST"
    assert "probability" in rows[0]
