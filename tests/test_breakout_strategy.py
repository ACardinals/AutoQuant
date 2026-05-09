from market_monitor.models import Candle
from market_monitor.strategies.breakout import BreakoutStrategy


def test_breakout_strategy_generates_signal():
    candles = [
        Candle(
            timestamp=__import__("datetime").datetime(2024, 1, 1, i % 24),
            symbol="BTCUSDT",
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100 + i,
            volume=100,
        )
        for i in range(25)
    ]
    candles[-1] = Candle(
        timestamp=candles[-1].timestamp,
        symbol="BTCUSDT",
        open=130,
        high=135,
        low=129,
        close=134,
        volume=300,
    )

    signal = BreakoutStrategy().generate_signal(candles)

    assert signal.action == "buy_candidate"
    assert signal.confidence > 0
