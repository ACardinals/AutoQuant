from __future__ import annotations

from dataclasses import dataclass

from market_monitor.models import Candle


@dataclass(frozen=True)
class MacdPoint:
    macd: float
    signal: float
    histogram: float


@dataclass(frozen=True)
class BollingerBands:
    middle: float
    upper: float
    lower: float


def simple_moving_average(values: list[float], window: int) -> float | None:
    if window <= 0 or len(values) < window:
        return None
    return sum(values[-window:]) / window


def exponential_moving_average(values: list[float], window: int) -> float | None:
    if window <= 0 or len(values) < window:
        return None
    multiplier = 2 / (window + 1)
    ema = sum(values[:window]) / window
    for value in values[window:]:
        ema = (value - ema) * multiplier + ema
    return ema


def moving_average_convergence_divergence(
    values: list[float],
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> MacdPoint | None:
    if min(fast_window, slow_window, signal_window) <= 0 or fast_window >= slow_window:
        return None
    if len(values) < slow_window + signal_window:
        return None

    macd_values = []
    for end in range(slow_window, len(values) + 1):
        subset = values[:end]
        fast = exponential_moving_average(subset, fast_window)
        slow = exponential_moving_average(subset, slow_window)
        if fast is not None and slow is not None:
            macd_values.append(fast - slow)

    signal = exponential_moving_average(macd_values, signal_window)
    if not macd_values or signal is None:
        return None
    macd = macd_values[-1]
    return MacdPoint(macd=macd, signal=signal, histogram=macd - signal)


def bollinger_bands(values: list[float], window: int = 20, standard_deviations: float = 2.0) -> BollingerBands | None:
    if window <= 0 or len(values) < window:
        return None
    recent = values[-window:]
    middle = sum(recent) / window
    variance = sum((value - middle) ** 2 for value in recent) / window
    deviation = variance**0.5
    return BollingerBands(
        middle=middle,
        upper=middle + deviation * standard_deviations,
        lower=middle - deviation * standard_deviations,
    )


def relative_strength_index(candles: list[Candle], window: int = 14) -> float | None:
    if window <= 0 or len(candles) < window + 1:
        return None

    gains = 0.0
    losses = 0.0
    for previous, current in zip(candles[-window - 1 : -1], candles[-window:]):
        change = current.close - previous.close
        if change > 0:
            gains += change
        else:
            losses += abs(change)

    average_gain = gains / window
    average_loss = losses / window
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def average_true_range(candles: list[Candle], window: int = 14) -> float | None:
    if window <= 0 or len(candles) < window + 1:
        return None

    true_ranges = []
    for previous, current in zip(candles[-window - 1 : -1], candles[-window:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return sum(true_ranges) / window


def volume_ratio(candles: list[Candle], window: int = 20) -> float | None:
    if window <= 0 or len(candles) < window + 1:
        return None
    prior = candles[-window - 1 : -1]
    average_volume = sum(candle.volume for candle in prior) / window
    if average_volume <= 0:
        return None
    return candles[-1].volume / average_volume
