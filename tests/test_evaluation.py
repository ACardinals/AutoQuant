from datetime import datetime, timedelta

from market_monitor.evaluation import compare_strategies, format_strategy_comparison_table
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
    assert rows[0]["total_return_pct"] >= rows[-1]["total_return_pct"]
    assert "max_drawdown_pct" in rows[0]
    assert "average_trade_return_pct" in rows[0]


def test_format_strategy_comparison_table():
    table = format_strategy_comparison_table(
        [
            {
                "strategy": "ma_trend",
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

    assert "strategy" in table
    assert "ma_trend" in table
