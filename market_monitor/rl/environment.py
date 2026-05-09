from __future__ import annotations

from dataclasses import dataclass

from market_monitor.indicators import average_true_range, relative_strength_index, simple_moving_average
from market_monitor.models import Candle


HOLD = 0
BUY_OR_INCREASE = 1
SELL_OR_REDUCE = 2


@dataclass
class TradingEnvironmentConfig:
    window_size: int = 50
    fee_rate: float = 0.001
    initial_cash: float = 10_000.0
    trade_fraction: float = 0.25


@dataclass(frozen=True)
class TradingStepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


class TradingEnvironment:
    def __init__(self, candles: list[Candle], config: TradingEnvironmentConfig | None = None) -> None:
        if not candles:
            raise ValueError("TradingEnvironment requires at least one candle")
        self.candles = candles
        self.config = config or TradingEnvironmentConfig()
        self.current_index = 0
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.previous_equity = self.config.initial_cash

    def reset(self) -> dict:
        self.current_index = min(self.config.window_size, len(self.candles) - 1)
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.previous_equity = self._equity(self.candles[self.current_index].close)
        return self._observation()

    def step(self, action: int) -> TradingStepResult:
        if self.current_index >= len(self.candles) - 1:
            return TradingStepResult(self._observation(), 0.0, True, {"reason": "already_done"})

        candle = self.candles[self.current_index]
        fee = 0.0
        action_name = "hold"

        if action == BUY_OR_INCREASE:
            fee = self._buy(candle.close)
            action_name = "buy_or_increase"
        elif action == SELL_OR_REDUCE:
            fee = self._sell(candle.close)
            action_name = "sell_or_reduce"
        elif action != HOLD:
            raise ValueError(f"Unknown action: {action}")

        self.current_index += 1
        latest = self.candles[self.current_index]
        equity = self._equity(latest.close)
        reward = equity - self.previous_equity - fee
        self.previous_equity = equity
        done = self.current_index >= len(self.candles) - 1
        return TradingStepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info={"equity": equity, "fee": fee, "action": action_name},
        )

    def _buy(self, price: float) -> float:
        budget = self.cash * self.config.trade_fraction
        if budget <= 0 or price <= 0:
            return 0.0
        fee = budget * self.config.fee_rate
        quantity = (budget - fee) / price
        self.cash -= budget
        self.position += quantity
        return fee

    def _sell(self, price: float) -> float:
        if self.position <= 0 or price <= 0:
            return 0.0
        quantity = self.position * self.config.trade_fraction
        proceeds = quantity * price
        fee = proceeds * self.config.fee_rate
        self.position -= quantity
        self.cash += proceeds - fee
        return fee

    def _observation(self) -> dict:
        candle = self.candles[self.current_index]
        equity = self._equity(candle.close)
        technicals = self._technical_features(candle, equity)
        return {
            "step": self.current_index,
            "recent_returns": self._recent_returns(),
            "cash": self.cash,
            "position": self.position,
            "equity": equity,
            "price": candle.close,
            **technicals,
        }

    def _technical_features(self, candle: Candle, equity: float) -> dict:
        history = self.candles[: self.current_index + 1]
        closes = [item.close for item in history]
        sma_20 = simple_moving_average(closes, 20)
        rsi = relative_strength_index(history, 14)
        atr = average_true_range(history, 14)
        latest_return = closes[-1] / closes[-2] - 1 if len(closes) >= 2 and closes[-2] else 0.0
        position_value = self.position * candle.close
        return {
            "latest_return": latest_return,
            "sma_20_distance": candle.close / sma_20 - 1 if sma_20 else 0.0,
            "rsi_14": rsi if rsi is not None else 50.0,
            "atr_14_pct": atr / candle.close if atr is not None and candle.close else 0.0,
            "position_ratio": position_value / equity if equity else 0.0,
        }

    def _recent_returns(self) -> list[float]:
        start = max(1, self.current_index - self.config.window_size + 1)
        returns = []
        for index in range(start, self.current_index + 1):
            previous = self.candles[index - 1].close
            current = self.candles[index].close
            returns.append(current / previous - 1 if previous else 0.0)
        return returns

    def _equity(self, price: float) -> float:
        return self.cash + self.position * price


class TradingEnvironmentPlaceholder:
    def __init__(self, config: TradingEnvironmentConfig | None = None) -> None:
        self.config = config or TradingEnvironmentConfig()

    def describe(self) -> dict:
        return {
            "observation": [
                "recent_returns",
                "cash",
                "position",
                "equity",
                "price",
                "step",
                "latest_return",
                "sma_20_distance",
                "rsi_14",
                "atr_14_pct",
                "position_ratio",
            ],
            "actions": {HOLD: "hold", BUY_OR_INCREASE: "buy_or_increase", SELL_OR_REDUCE: "sell_or_reduce"},
            "reward": "equity_change - fee_penalty",
            "config": {
                "window_size": self.config.window_size,
                "fee_rate": self.config.fee_rate,
                "initial_cash": self.config.initial_cash,
                "trade_fraction": self.config.trade_fraction,
            },
        }
