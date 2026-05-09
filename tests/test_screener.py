from datetime import datetime

from market_monitor.models import Candle
from market_monitor.signals.screener import screen_watchlist
from market_monitor.strategies.breakout import BreakoutStrategy


def _candles(symbol: str, breakout: bool) -> list[Candle]:
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, i % 24),
            symbol=symbol,
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100 + i,
            volume=100,
        )
        for i in range(25)
    ]
    if breakout:
        candles[-1] = Candle(
            timestamp=candles[-1].timestamp,
            symbol=symbol,
            open=130,
            high=135,
            low=129,
            close=134,
            volume=300,
        )
    return candles


def test_screen_watchlist_sorts_buy_candidates_first():
    signals = screen_watchlist(
        {
            "ETHUSDT": _candles("ETHUSDT", breakout=False),
            "BTCUSDT": _candles("BTCUSDT", breakout=True),
        },
        BreakoutStrategy(),
    )

    assert [signal.symbol for signal in signals] == ["BTCUSDT", "ETHUSDT"]
    assert signals[0].action == "buy_candidate"
