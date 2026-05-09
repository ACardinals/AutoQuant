from __future__ import annotations

from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


def screen_watchlist(candles_by_symbol: dict[str, list[Candle]], strategy: Strategy) -> list[StrategySignal]:
    signals = [strategy.generate_signal(candles) for candles in candles_by_symbol.values() if candles]
    return sorted(signals, key=_signal_sort_key)


def _signal_sort_key(signal: StrategySignal) -> tuple[int, float, str]:
    priority = 0 if signal.action == "buy_candidate" else 1
    return priority, -signal.confidence, signal.symbol
