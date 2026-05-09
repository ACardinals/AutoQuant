from __future__ import annotations

from market_monitor.indicators import average_true_range, relative_strength_index, simple_moving_average
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class RsiReboundStrategy(Strategy):
    name = "rsi_rebound"

    def __init__(self, rsi_window: int = 14, trend_window: int = 20) -> None:
        self.rsi_window = rsi_window
        self.trend_window = trend_window

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) < max(self.rsi_window + 1, self.trend_window):
            return StrategySignal(
                symbol=candles[-1].symbol if candles else "UNKNOWN",
                action="hold",
                confidence=0.0,
                reasons=["历史K线数量不足"],
            )

        latest = candles[-1]
        rsi = relative_strength_index(candles, self.rsi_window)
        previous_rsi = relative_strength_index(candles[:-1], self.rsi_window)
        closes = [candle.close for candle in candles]
        trend_ma = simple_moving_average(closes, self.trend_window)
        atr = average_true_range(candles, 14)

        reasons = []
        confidence = 0.0

        if rsi is not None and 30 <= rsi <= 55:
            reasons.append(f"RSI处于反弹观察区间({rsi:.1f})")
            confidence += 0.3

        if rsi is not None and previous_rsi is not None and rsi > previous_rsi:
            reasons.append("RSI较上一周期回升")
            confidence += 0.2

        if len(candles) >= 2 and latest.close > candles[-2].close:
            reasons.append("最新收盘价企稳回升")
            confidence += 0.2

        if trend_ma is not None and latest.close >= trend_ma * 0.97:
            reasons.append("价格接近或站回趋势均线")
            confidence += 0.1

        if confidence >= 0.65:
            stop_loss = latest.close - atr * 1.5 if atr is not None else latest.close * 0.95
            return StrategySignal(
                symbol=latest.symbol,
                action="buy_candidate",
                confidence=min(confidence, 1.0),
                reasons=reasons,
                stop_loss=stop_loss,
                take_profit=latest.close * 1.08,
                max_position_pct=0.08,
            )

        return StrategySignal(
            symbol=latest.symbol,
            action="hold",
            confidence=confidence,
            reasons=reasons or ["未满足RSI反弹策略条件"],
        )
