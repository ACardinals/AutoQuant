from __future__ import annotations

from market_monitor.indicators import simple_moving_average, volume_ratio
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class BreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(self, lookback: int = 20, volume_window: int = 20) -> None:
        self.lookback = lookback
        self.volume_window = volume_window

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) < max(self.lookback, self.volume_window) + 1:
            return StrategySignal(
                symbol=candles[-1].symbol if candles else "UNKNOWN",
                action="hold",
                confidence=0.0,
                reasons=["历史K线数量不足"],
            )

        latest = candles[-1]
        prior = candles[-self.lookback - 1 : -1]
        breakout_price = max(candle.high for candle in prior)
        latest_volume_ratio = volume_ratio(candles, self.volume_window) or 0.0

        reasons = []
        confidence = 0.0

        if latest.close > breakout_price:
            reasons.append(f"收盘价突破{self.lookback}周期高点")
            confidence += 0.45

        if latest_volume_ratio >= 1.5:
            reasons.append(f"成交量是均量的{latest_volume_ratio:.2f}倍")
            confidence += 0.25

        closes = [candle.close for candle in candles]
        short_ma = simple_moving_average(closes, 5)
        long_ma = simple_moving_average(closes, 20)
        if short_ma is not None and long_ma is not None and short_ma > long_ma:
            reasons.append("短期均线高于长期均线")
            confidence += 0.2

        if confidence >= 0.65:
            return StrategySignal(
                symbol=latest.symbol,
                action="buy_candidate",
                confidence=min(confidence, 1.0),
                reasons=reasons,
                stop_loss=latest.close * 0.96,
                take_profit=latest.close * 1.08,
                max_position_pct=0.12,
            )

        return StrategySignal(
            symbol=latest.symbol,
            action="hold",
            confidence=confidence,
            reasons=reasons or ["未满足突破策略条件"],
        )


