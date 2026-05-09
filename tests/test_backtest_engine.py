from datetime import datetime, timedelta

from market_monitor.backtest.engine import LongOnlyBacktester
from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


class BuyOnceStrategy(Strategy):
    name = "buy_once"

    def __init__(self, stop_loss: float | None = None, take_profit: float | None = None) -> None:
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        if len(candles) == 22:
            latest = candles[-1]
            return StrategySignal(
                symbol=latest.symbol,
                action="buy_candidate",
                confidence=0.9,
                reasons=["测试买入"],
                stop_loss=self.stop_loss,
                take_profit=self.take_profit,
                max_position_pct=0.5,
            )
        return StrategySignal(symbol=candles[-1].symbol, action="hold", confidence=0.0, reasons=["hold"])


class HoldStrategy(Strategy):
    name = "hold"

    def generate_signal(self, candles: list[Candle]) -> StrategySignal:
        return StrategySignal(symbol=candles[-1].symbol, action="hold", confidence=0.0, reasons=["hold"])


def _candles(prices: list[float]) -> list[Candle]:
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


def test_backtest_records_final_close_trade_and_metrics():
    candles = _candles([100] * 22 + [110, 112])

    result = LongOnlyBacktester(initial_cash=10_000, fee_rate=0.0).run(candles, BuyOnceStrategy())
    output = result.as_dict()

    assert output["trades"] == 1
    assert output["trade_records"][0]["exit_reason"] == "final_close"
    assert output["trade_records"][0]["pnl"] > 0
    assert output["metrics"]["win_rate_pct"] == 100.0
    assert "total_return_pct" in output
    assert "equity_curve" in output


def test_backtest_take_profit_exit_reason():
    candles = _candles([100] * 22 + [105, 108])

    result = LongOnlyBacktester(initial_cash=10_000, fee_rate=0.0).run(
        candles,
        BuyOnceStrategy(take_profit=104),
    )

    assert result.trade_records[0].exit_reason == "take_profit"
    assert result.trade_records[0].exit_price == 104


def test_backtest_stop_loss_exit_reason():
    candles = _candles([100] * 22 + [95, 94])

    result = LongOnlyBacktester(initial_cash=10_000, fee_rate=0.0).run(
        candles,
        BuyOnceStrategy(stop_loss=96),
    )

    assert result.trade_records[0].exit_reason == "stop_loss"
    assert result.trade_records[0].exit_price == 96


def test_backtest_no_trades_has_empty_records_and_none_metrics():
    candles = _candles([100] * 25)

    result = LongOnlyBacktester(initial_cash=10_000, fee_rate=0.0).run(candles, HoldStrategy())
    output = result.as_dict()

    assert output["trades"] == 0
    assert output["trade_records"] == []
    assert output["metrics"]["win_rate_pct"] is None
    assert output["metrics"]["profit_factor"] is None
    assert output["metrics"]["average_trade_return_pct"] is None
