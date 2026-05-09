from __future__ import annotations

from market_monitor.indicators import average_true_range, bollinger_bands, relative_strength_index
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class BollingerReversionStrategy(Strategy):
    name = "bollinger_reversion"

    def __init__(self, band_window: int = 20, rsi_window: int = 14) -> None:
        self.band_window = band_window
        self.rsi_window = rsi_window

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) < max(self.band_window, self.rsi_window + 1):
            return StrategySignal(
                symbol=candles[-1].symbol if candles else "UNKNOWN",
                action="hold",
                confidence=0.0,
                reasons=["历史K线数量不足"],
            )

        latest = candles[-1]
        previous = candles[-2]
        closes = [candle.close for candle in candles]
        bands = bollinger_bands(closes, self.band_window)
        rsi = relative_strength_index(candles, self.rsi_window)
        previous_rsi = relative_strength_index(candles[:-1], self.rsi_window)
        atr = average_true_range(candles, 14)

        reasons = []
        confidence = 0.0

        if bands is not None and _recently_rebounded_from_lower_band(closes, self.band_window):
            reasons.append("价格从布林下轨附近回升")
            confidence += 0.35

        if bands is not None and latest.close < bands.middle:
            reasons.append("价格仍处于均值回归空间")
            confidence += 0.15

        if rsi is not None and 30 <= rsi <= 55:
            reasons.append(f"RSI处于修复区间({rsi:.1f})")
            confidence += 0.2

        if rsi is not None and previous_rsi is not None and rsi > previous_rsi:
            reasons.append("RSI开始回升")
            confidence += 0.15

        if latest.close > previous.close:
            reasons.append("最新收盘价企稳")
            confidence += 0.1

        if confidence >= 0.65:
            stop_loss = latest.close - atr * 1.5 if atr is not None else latest.close * 0.95
            take_profit = bands.middle if bands is not None and bands.middle > latest.close else latest.close * 1.08
            return StrategySignal(
                symbol=latest.symbol,
                action="buy_candidate",
                confidence=min(confidence, 1.0),
                reasons=reasons,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_position_pct=0.08,
            )

        return StrategySignal(
            symbol=latest.symbol,
            action="hold",
            confidence=confidence,
            reasons=reasons or ["未满足布林均值回归策略条件"],
        )


def _recently_rebounded_from_lower_band(closes: list[float], band_window: int, lookback: int = 5) -> bool:
    start = max(band_window, len(closes) - lookback)
    touched = False
    for end in range(start, len(closes) + 1):
        bands = bollinger_bands(closes[:end], band_window)
        if bands is None:
            continue
        close = closes[end - 1]
        if close <= bands.lower * 1.02:
            touched = True
        elif touched and close > bands.lower:
            return True
    return touched and closes[-1] > closes[-2]
