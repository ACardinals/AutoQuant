from datetime import datetime, timedelta

from market_monitor.models import Candle
from market_monitor.strategies.bollinger_reversion import BollingerReversionStrategy
from market_monitor.strategies.macd_trend import MacdTrendStrategy
from market_monitor.strategies.volume_pullback import VolumePullbackStrategy


def _candles(closes, volumes=None):
    start = datetime(2024, 1, 1)
    volumes = volumes or [100] * len(closes)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            symbol="TEST",
            open=close - 0.5,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=volumes[index],
        )
        for index, close in enumerate(closes)
    ]


def test_macd_trend_strategy_generates_signal():
    closes = [float(value) for value in range(1, 45)] + [46, 49, 53, 58, 64]
    volumes = [100] * 40 + [130] * 9

    signal = MacdTrendStrategy().generate_signal(_candles(closes, volumes))

    assert signal.action == "buy_candidate"
    assert signal.confidence >= 0.65


def test_bollinger_reversion_strategy_generates_signal():
    closes = [100] * 20 + [80, 82, 84, 86, 88]

    signal = BollingerReversionStrategy().generate_signal(_candles(closes))

    assert signal.action == "buy_candidate"
    assert signal.confidence >= 0.65


def test_volume_pullback_strategy_generates_signal():
    closes = [100 + index for index in range(20)] + [118, 116, 117, 119]
    volumes = [100] * 20 + [80, 75, 85, 95]

    signal = VolumePullbackStrategy().generate_signal(_candles(closes, volumes))

    assert signal.action == "buy_candidate"
    assert signal.confidence >= 0.65
