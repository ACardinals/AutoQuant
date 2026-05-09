from __future__ import annotations

from market_monitor.indicators import average_true_range, moving_average_convergence_divergence, simple_moving_average, volume_ratio
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class MacdTrendStrategy(Strategy):
    name = "macd_trend"

    def __init__(self, trend_window: int = 20, volume_window: int = 20) -> None:
        self.trend_window = trend_window
        self.volume_window = volume_window

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) < 35:
            return StrategySignal(
                symbol=candles[-1].symbol if candles else "UNKNOWN",
                action="hold",
                confidence=0.0,
                reasons=["历史K线数量不足"],
            )

        latest = candles[-1]
        closes = [candle.close for candle in candles]
        macd = moving_average_convergence_divergence(closes)
        previous_macd = moving_average_convergence_divergence(closes[:-1])
        trend_ma = simple_moving_average(closes, self.trend_window)
        latest_volume_ratio = volume_ratio(candles, self.volume_window) or 0.0
        atr = average_true_range(candles, 14)

        reasons = []
        confidence = 0.0

        if macd is not None and macd.macd > macd.signal and macd.histogram > 0:
            reasons.append("MACD位于信号线上方且柱体为正")
            confidence += 0.35

        if macd is not None and previous_macd is not None and macd.histogram > previous_macd.histogram:
            reasons.append("MACD动能继续增强")
            confidence += 0.2

        if trend_ma is not None and latest.close > trend_ma:
            reasons.append("收盘价站上趋势均线")
            confidence += 0.2

        if latest_volume_ratio >= 1.0:
            reasons.append(f"成交量不低于均量({latest_volume_ratio:.2f}倍)")
            confidence += 0.1

        if len(candles) >= 2 and latest.close > candles[-2].close:
            reasons.append("最新收盘价走强")
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
            reasons=reasons or ["未满足MACD趋势策略条件"],
        )
