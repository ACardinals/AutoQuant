from datetime import datetime

from market_monitor.indicators import (
    average_true_range,
    bollinger_bands,
    exponential_moving_average,
    moving_average_convergence_divergence,
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)
from market_monitor.models import Candle


def _candles(closes):
    return [
        Candle(
            timestamp=datetime(2024, 1, 1, i % 24),
            symbol="TEST",
            open=close - 0.5,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=100 + i,
        )
        for i, close in enumerate(closes)
    ]


def test_simple_moving_average():
    assert simple_moving_average([1, 2, 3, 4], 3) == 3
    assert simple_moving_average([1, 2], 3) is None


def test_exponential_moving_average():
    assert exponential_moving_average([1, 2, 3, 4, 5], 3) > 3
    assert exponential_moving_average([1, 2], 3) is None


def test_macd_and_bollinger_need_history():
    assert moving_average_convergence_divergence([1, 2, 3]) is None
    assert bollinger_bands([1, 2, 3], 20) is None


def test_macd_detects_positive_trend():
    values = [float(value) for value in range(1, 45)] + [48, 53, 59, 66, 74, 83]
    macd = moving_average_convergence_divergence(values)

    assert macd is not None
    assert macd.macd > macd.signal
    assert macd.histogram > 0


def test_bollinger_bands():
    bands = bollinger_bands([float(value) for value in range(1, 21)], 20)

    assert bands is not None
    assert bands.upper > bands.middle > bands.lower


def test_relative_strength_index_and_atr_need_history():
    candles = _candles([1, 2, 3])

    assert relative_strength_index(candles, 14) is None
    assert average_true_range(candles, 14) is None


def test_relative_strength_index_and_volume_ratio():
    candles = _candles([10, 9, 8, 9, 10, 11])

    assert relative_strength_index(candles, 5) > 50
    assert volume_ratio(candles, 3) > 1
