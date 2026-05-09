from datetime import datetime

from market_monitor.models import Candle
from market_monitor.strategies.moving_average import MovingAverageTrendStrategy
from market_monitor.strategies.rsi_rebound import RsiReboundStrategy


def _trend_candles():
    return [
        Candle(
            timestamp=datetime(2024, 1, 1, i % 24),
            symbol="TEST",
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100 + i,
            volume=100 + i * 3,
        )
        for i in range(25)
    ]


def test_moving_average_trend_strategy_generates_signal():
    signal = MovingAverageTrendStrategy().generate_signal(_trend_candles())

    assert signal.action == "buy_candidate"
    assert signal.confidence >= 0.65


def test_rsi_rebound_strategy_generates_signal():
    closes = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 75, 74, 76, 78, 80, 82, 84, 86]
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, i % 24),
            symbol="TEST",
            open=close - 0.5,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=100,
        )
        for i, close in enumerate(closes)
    ]

    signal = RsiReboundStrategy().generate_signal(candles)

    assert signal.action == "buy_candidate"
    assert signal.confidence >= 0.65
