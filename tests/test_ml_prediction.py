from datetime import datetime, timedelta
from pathlib import Path

from market_monitor.data.watchlist import WatchlistItem
from market_monitor.ml.prediction import format_ml_rank_table, predict_latest_probability, rank_watchlist_ml
from market_monitor.models import Candle


def _candles(symbol="TEST", count=90):
    start = datetime(2024, 1, 1)
    prices = []
    for index in range(count):
        cycle = index % 12
        prices.append(100 + index * 0.08 + (cycle - 6) * 0.25)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            symbol=symbol,
            open=price - 0.2,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=1000 + (index % 9) * 30,
        )
        for index, price in enumerate(prices)
    ]


def test_predict_latest_probability_returns_current_features():
    result = predict_latest_probability(_candles(), "logistic_regression", horizon=5)

    assert 0 <= result["probability"] <= 1
    assert result["sample_count"] > 0
    assert "rsi_14" in result
    assert "macd_histogram" in result


def test_rank_watchlist_ml_sorts_candidates():
    items = [
        WatchlistItem(symbol="AAA", name="Asset A", market="unit", csv_path=Path("a.csv")),
        WatchlistItem(symbol="BBB", name="Asset B", market="unit", csv_path=Path("b.csv")),
    ]
    rows = rank_watchlist_ml(
        items,
        {"AAA": _candles("AAA"), "BBB": _candles("BBB")},
        "logistic_regression",
        horizon=5,
        top_n=2,
    )

    assert rows
    assert rows[0]["probability"] >= rows[-1]["probability"]
    assert rows[0]["symbol"] in {"AAA", "BBB"}


def test_format_ml_rank_table():
    table = format_ml_rank_table(
        [
            {
                "symbol": "AAA",
                "name": "Asset A",
                "market": "unit",
                "probability": 0.75,
                "sample_count": 40,
                "latest_date": "2024-01-01",
                "rsi_14": 55.0,
                "macd_histogram": 0.1,
                "volume_ratio_20": 1.2,
                "return_5": 2.0,
            }
        ]
    )

    assert "probability" in table
    assert "AAA" in table
