from datetime import datetime, timedelta
from pathlib import Path

from market_monitor.data.watchlist import WatchlistItem
from market_monitor.evaluation import (
    compare_strategies,
    compare_watchlist,
    format_strategy_comparison_table,
    format_watchlist_comparison_table,
    score_strategy_row,
)
from market_monitor.models import Candle


def _candles(prices):
    start = datetime(2024, 1, 1)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            symbol="TEST",
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=100 + index,
        )
        for index, price in enumerate(prices)
    ]


def test_compare_strategies_returns_sorted_summary_rows():
    rows = compare_strategies(_candles([100] * 22 + [110, 112, 115]), ["breakout", "ma_trend"], 10_000)

    assert [row["strategy"] for row in rows]
    assert rows[0]["score"] >= rows[-1]["score"]
    assert "max_drawdown_pct" in rows[0]
    assert "average_trade_return_pct" in rows[0]


def test_score_strategy_row_rewards_return_and_penalizes_drawdown():
    strong = {
        "total_return_pct": 10,
        "max_drawdown_pct": 3,
        "win_rate_pct": 60,
        "profit_factor": 1.8,
        "average_trade_return_pct": 1.2,
        "trades": 8,
    }
    weak = {**strong, "total_return_pct": 2, "max_drawdown_pct": 12, "profit_factor": 0.8}

    assert score_strategy_row(strong) > score_strategy_row(weak)


def test_compare_watchlist_enriches_and_sorts_rows():
    items = [WatchlistItem(symbol="TEST", name="Test Asset", market="unit", csv_path=Path("test.csv"))]
    rows = compare_watchlist(items, {"TEST": _candles([100] * 22 + [110, 112, 115])}, ["breakout", "ma_trend"])

    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["name"] == "Test Asset"
    assert rows[0]["market"] == "unit"
    assert rows[0]["score"] >= rows[-1]["score"]


def test_format_strategy_comparison_table():
    table = format_strategy_comparison_table(
        [
            {
                "strategy": "ma_trend",
                "score": 3.2,
                "final_equity": 10_100.0,
                "total_return_pct": 1.0,
                "max_drawdown_pct": 0.5,
                "trades": 2,
                "win_rate_pct": 50.0,
                "profit_factor": 1.2,
                "average_trade_return_pct": 0.5,
            }
        ]
    )

    assert "score" in table
    assert "ma_trend" in table


def test_format_watchlist_comparison_table():
    table = format_watchlist_comparison_table(
        [
            {
                "symbol": "TEST",
                "name": "Test Asset",
                "market": "unit",
                "strategy": "ma_trend",
                "score": 3.2,
                "total_return_pct": 1.0,
                "max_drawdown_pct": 0.5,
                "trades": 2,
                "win_rate_pct": 50.0,
                "profit_factor": 1.2,
            }
        ]
    )

    assert "symbol" in table
    assert "Test Asset" in table
