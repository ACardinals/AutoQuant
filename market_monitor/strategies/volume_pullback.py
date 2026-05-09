from __future__ import annotations

from market_monitor.indicators import average_true_range, simple_moving_average, volume_ratio
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class VolumePullbackStrategy(Strategy):
    name = "volume_pullback"

    def __init__(self, short_window: int = 5, long_window: int = 20, volume_window: int = 20) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.volume_window = volume_window

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) < self.long_window + 3:
            return StrategySignal(
                symbol=candles[-1].symbol if candles else "UNKNOWN",
                action="hold",
                confidence=0.0,
                reasons=["历史K线数量不足"],
            )

        latest = candles[-1]
        previous = candles[-2]
        closes = [candle.close for candle in candles]
        short_ma = simple_moving_average(closes, self.short_window)
        long_ma = simple_moving_average(closes, self.long_window)
        prior_long_ma = simple_moving_average(closes[:-3], self.long_window)
        latest_volume_ratio = volume_ratio(candles, self.volume_window) or 0.0
        atr = average_true_range(candles, 14)

        reasons = []
        confidence = 0.0

        if short_ma is not None and long_ma is not None and short_ma > long_ma:
            reasons.append("短期均线仍高于长期均线")
            confidence += 0.25

        if long_ma is not None and prior_long_ma is not None and long_ma >= prior_long_ma:
            reasons.append("长期均线保持上行或走平")
            confidence += 0.2

        pullback_recently = any(candles[index].close < candles[index - 1].close for index in range(len(candles) - 3, len(candles)))
        if pullback_recently and long_ma is not None and latest.close >= long_ma * 0.98:
            reasons.append("回调后仍接近趋势均线")
            confidence += 0.2

        if 0.6 <= latest_volume_ratio <= 1.2:
            reasons.append(f"回调阶段成交量温和({latest_volume_ratio:.2f}倍)")
            confidence += 0.15

        if latest.close > previous.close and latest.close > latest.open:
            reasons.append("最新K线恢复上行")
            confidence += 0.15

        if confidence >= 0.65:
            stop_loss = latest.close - atr * 1.8 if atr is not None else latest.close * 0.95
            return StrategySignal(
                symbol=latest.symbol,
                action="buy_candidate",
                confidence=min(confidence, 1.0),
                reasons=reasons,
                stop_loss=stop_loss,
                take_profit=latest.close * 1.09,
                max_position_pct=0.09,
            )

        return StrategySignal(
            symbol=latest.symbol,
            action="hold",
            confidence=confidence,
            reasons=reasons or ["未满足缩量回调策略条件"],
        )
