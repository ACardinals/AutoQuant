from __future__ import annotations

from datetime import datetime

from market_monitor.indicators import (
    average_true_range,
    bollinger_bands,
    moving_average_convergence_divergence,
    relative_strength_index,
    simple_moving_average,
    volume_ratio,
)
from market_monitor.models import Candle

FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "sma_5_distance",
    "sma_20_distance",
    "rsi_14",
    "atr_14_pct",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_position",
    "volume_ratio_20",
]


def build_feature_rows(candles: list[Candle], min_history: int = 35) -> list[dict]:
    rows = []
    for index in range(min_history, len(candles)):
        history = candles[: index + 1]
        row = feature_row(history)
        if row is not None:
            rows.append(row)
    return rows


def feature_row(candles: list[Candle]) -> dict | None:
    if len(candles) < 35:
        return None

    latest = candles[-1]
    closes = [candle.close for candle in candles]
    sma_5 = simple_moving_average(closes, 5)
    sma_20 = simple_moving_average(closes, 20)
    rsi = relative_strength_index(candles, 14)
    atr = average_true_range(candles, 14)
    macd = moving_average_convergence_divergence(closes)
    bands = bollinger_bands(closes, 20)
    vol_ratio = volume_ratio(candles, 20)

    if None in (sma_5, sma_20, rsi, atr, macd, bands, vol_ratio):
        return None

    return {
        "timestamp": latest.timestamp,
        "symbol": latest.symbol,
        "return_1": _period_return(closes, 1),
        "return_5": _period_return(closes, 5),
        "return_10": _period_return(closes, 10),
        "return_20": _period_return(closes, 20),
        "sma_5_distance": latest.close / sma_5 - 1,
        "sma_20_distance": latest.close / sma_20 - 1,
        "rsi_14": rsi,
        "atr_14_pct": atr / latest.close if latest.close else 0.0,
        "macd": macd.macd,
        "macd_signal": macd.signal,
        "macd_histogram": macd.histogram,
        "bollinger_position": _bollinger_position(latest.close, bands.lower, bands.upper),
        "volume_ratio_20": vol_ratio,
    }


def feature_matrix(rows: list[dict]) -> list[list[float]]:
    return [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows]


def _period_return(closes: list[float], periods: int) -> float:
    if len(closes) <= periods or closes[-periods - 1] == 0:
        return 0.0
    return closes[-1] / closes[-periods - 1] - 1


def _bollinger_position(close: float, lower: float, upper: float) -> float:
    width = upper - lower
    if width == 0:
        return 0.5
    return (close - lower) / width
