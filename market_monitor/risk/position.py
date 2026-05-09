from __future__ import annotations

from market_monitor.models import Candle


def simple_position_size(equity: float, max_position_pct: float, price: float) -> float:
    if equity <= 0 or price <= 0:
        return 0.0
    return equity * max_position_pct / price


def recent_drawdown(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    peak = candles[0].close
    max_drawdown = 0.0
    for candle in candles:
        peak = max(peak, candle.close)
        if peak > 0:
            max_drawdown = min(max_drawdown, candle.close / peak - 1)
    return abs(max_drawdown)
