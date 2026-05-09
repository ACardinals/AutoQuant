from __future__ import annotations

from market_monitor.strategies.base import Strategy
from market_monitor.strategies.bollinger_reversion import BollingerReversionStrategy
from market_monitor.strategies.breakout import BreakoutStrategy
from market_monitor.strategies.macd_trend import MacdTrendStrategy
from market_monitor.strategies.moving_average import MovingAverageTrendStrategy
from market_monitor.strategies.rsi_rebound import RsiReboundStrategy
from market_monitor.strategies.volume_pullback import VolumePullbackStrategy


_STRATEGIES = {
    "bollinger_reversion": BollingerReversionStrategy,
    "breakout": BreakoutStrategy,
    "ma_trend": MovingAverageTrendStrategy,
    "macd_trend": MacdTrendStrategy,
    "rsi_rebound": RsiReboundStrategy,
    "volume_pullback": VolumePullbackStrategy,
}


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES)


def create_strategy(name: str) -> Strategy:
    try:
        return _STRATEGIES[name]()
    except KeyError as exc:
        available = ", ".join(available_strategies())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}") from exc
