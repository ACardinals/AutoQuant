from datetime import datetime, timedelta

import pytest

from market_monitor.models import Candle
from market_monitor.rl.environment import (
    BUY_OR_INCREASE,
    SELL_OR_REDUCE,
    TradingEnvironment,
    TradingEnvironmentConfig,
    TradingEnvironmentPlaceholder,
)


def _candles(prices):
    start = datetime(2024, 1, 1)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            symbol="TEST",
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=100,
        )
        for index, price in enumerate(prices)
    ]


def test_trading_environment_reset_observation():
    env = TradingEnvironment(_candles([100, 101, 102, 103]), TradingEnvironmentConfig(window_size=2))

    observation = env.reset()

    assert observation["step"] == 2
    assert "recent_returns" in observation
    assert "latest_return" in observation
    assert "sma_20_distance" in observation
    assert "rsi_14" in observation
    assert "atr_14_pct" in observation
    assert "position_ratio" in observation
    assert observation["cash"] == 10_000.0
    assert observation["position"] == 0.0


def test_buy_action_changes_cash_and_position():
    env = TradingEnvironment(_candles([100, 101, 102, 103]), TradingEnvironmentConfig(window_size=1, fee_rate=0.0))
    env.reset()

    result = env.step(BUY_OR_INCREASE)

    assert result.observation["cash"] < 10_000.0
    assert result.observation["position"] > 0
    assert result.info["action"] == "buy_or_increase"


def test_sell_action_reduces_position():
    env = TradingEnvironment(_candles([100, 101, 102, 103, 104]), TradingEnvironmentConfig(window_size=1, fee_rate=0.0))
    env.reset()
    env.step(BUY_OR_INCREASE)
    before_position = env.position

    result = env.step(SELL_OR_REDUCE)

    assert result.observation["position"] < before_position
    assert result.info["action"] == "sell_or_reduce"


def test_reward_reflects_price_movement_after_buy():
    env = TradingEnvironment(_candles([100, 100, 110]), TradingEnvironmentConfig(window_size=1, fee_rate=0.0))
    env.reset()

    result = env.step(BUY_OR_INCREASE)

    assert result.reward > 0


def test_environment_reaches_done():
    env = TradingEnvironment(_candles([100, 101, 102]), TradingEnvironmentConfig(window_size=1))
    env.reset()

    result = env.step(BUY_OR_INCREASE)

    assert result.done is True


def test_environment_rejects_unknown_action():
    env = TradingEnvironment(_candles([100, 101, 102]), TradingEnvironmentConfig(window_size=1))
    env.reset()

    with pytest.raises(ValueError):
        env.step(99)


def test_rl_spec_describes_concrete_environment():
    spec = TradingEnvironmentPlaceholder().describe()

    assert spec["actions"][BUY_OR_INCREASE] == "buy_or_increase"
    assert "trade_fraction" in spec["config"]
    assert "position_ratio" in spec["observation"]
