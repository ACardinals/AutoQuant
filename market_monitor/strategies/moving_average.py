from __future__ import annotations

from market_monitor.indicators import average_true_range, simple_moving_average, volume_ratio
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class MovingAverageTrendStrategy(Strategy):
    name = "ma_trend"

    def __init__(self, short_window: int = 5, medium_window: int = 10, long_window: int = 20) -> None:
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) < self.long_window:
            return StrategySignal(
                symbol=candles[-1].symbol if candles else "UNKNOWN",
                action="hold",
                confidence=0.0,
                reasons=["历史K线数量不足"],
            )

        latest = candles[-1]
        closes = [candle.close for candle in candles]
        short_ma = simple_moving_average(closes, self.short_window)
        medium_ma = simple_moving_average(closes, self.medium_window)
        long_ma = simple_moving_average(closes, self.long_window)
        latest_volume_ratio = volume_ratio(candles, self.long_window) or 0.0
        atr = average_true_range(candles, 14)

        reasons = []
        confidence = 0.0

        if short_ma is not None and medium_ma is not None and long_ma is not None and short_ma > medium_ma > long_ma:
            reasons.append("短中长期均线多头排列")
            confidence += 0.45

        if long_ma is not None and latest.close > long_ma:
            reasons.append("收盘价位于长期均线上方")
            confidence += 0.2

        if latest_volume_ratio >= 1.2:
            reasons.append(f"成交量是均量的{latest_volume_ratio:.2f}倍")
            confidence += 0.15

        if len(candles) >= 2 and latest.close > candles[-2].close:
            reasons.append("最新收盘价继续走高")
            confidence += 0.1

        if confidence >= 0.65:
            stop_loss = latest.close - atr * 2 if atr is not None else latest.close * 0.95
            return StrategySignal(
                symbol=latest.symbol,
                action="buy_candidate",
                confidence=min(confidence, 1.0),
                reasons=reasons,
                stop_loss=stop_loss,
                take_profit=latest.close * 1.1,
                max_position_pct=0.1,
            )

        return StrategySignal(
            symbol=latest.symbol,
            action="hold",
            confidence=confidence,
            reasons=reasons or ["未满足均线趋势策略条件"],
        )
