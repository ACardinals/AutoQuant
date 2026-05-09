from datetime import datetime, timedelta

from market_monitor.models import Candle
from market_monitor.rl.baselines import available_policies, buy_and_hold_policy, create_policy, evaluate_policy, hold_policy, random_policy
from market_monitor.rl.environment import TradingEnvironmentConfig


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


def test_available_policies():
    assert available_policies() == ["buy_and_hold", "hold", "random"]


def test_hold_policy_never_buys():
    summary = evaluate_policy(_candles([100, 101, 102]), hold_policy, TradingEnvironmentConfig(window_size=1))

    assert summary["action_counts"]["hold"] == 1
    assert summary["final_equity"] == 10_000.0


def test_buy_and_hold_policy_buys_once():
    summary = evaluate_policy(_candles([100, 100, 110]), buy_and_hold_policy(), TradingEnvironmentConfig(window_size=1, fee_rate=0.0))

    assert summary["action_counts"]["buy_or_increase"] == 1
    assert summary["final_equity"] > 10_000.0


def test_random_policy_is_seeded():
    candles = _candles([100, 101, 102, 103, 104])

    first = evaluate_policy(candles, random_policy(seed=7), TradingEnvironmentConfig(window_size=1))
    second = evaluate_policy(candles, random_policy(seed=7), TradingEnvironmentConfig(window_size=1))

    assert first == second


def test_create_policy_rejects_unknown_name():
    try:
        create_policy("missing")
    except ValueError as exc:
        assert "Unknown" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
