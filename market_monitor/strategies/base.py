from __future__ import annotations

from abc import ABC, abstractmethod

from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        raise NotImplementedError
