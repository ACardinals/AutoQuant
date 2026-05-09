import pytest

from market_monitor.strategies.registry import available_strategies, create_strategy


def test_available_strategies_includes_all_names():
    assert available_strategies() == [
        "bollinger_reversion",
        "breakout",
        "ma_trend",
        "macd_trend",
        "rsi_rebound",
        "volume_pullback",
    ]


def test_create_strategy_returns_strategy_instances():
    assert create_strategy("bollinger_reversion").name == "bollinger_reversion"
    assert create_strategy("breakout").name == "breakout"
    assert create_strategy("ma_trend").name == "ma_trend"
    assert create_strategy("macd_trend").name == "macd_trend"
    assert create_strategy("rsi_rebound").name == "rsi_rebound"
    assert create_strategy("volume_pullback").name == "volume_pullback"


def test_create_strategy_rejects_unknown_name():
    with pytest.raises(ValueError):
        create_strategy("missing")
