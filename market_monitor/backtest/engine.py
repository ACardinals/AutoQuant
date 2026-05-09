from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_monitor.models import Candle
from market_monitor.signals.models import StrategySignal
from market_monitor.strategies.base import Strategy


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: float

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": round(self.equity, 2),
        }


@dataclass(frozen=True)
class TradeRecord:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float
    exit_fee: float
    pnl: float
    pnl_pct: float
    entry_reason: str
    exit_reason: str

    def as_dict(self) -> dict:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "quantity": round(self.quantity, 8),
            "fees": round(self.entry_fee + self.exit_fee, 4),
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
        }


@dataclass(frozen=True)
class _OpenTrade:
    entry_time: datetime
    entry_price: float
    quantity: float
    entry_fee: float
    invested_cash: float
    stop_loss: float | None
    take_profit: float | None
    entry_reason: str


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trades: int
    equity_curve: list[EquityPoint]
    trade_records: list[TradeRecord]
    win_rate: float | None
    annualized_return: float | None
    profit_factor: float | None

    def as_dict(self) -> dict:
        return {
            "initial_cash": round(self.initial_cash, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "trades": self.trades,
            "metrics": {
                "win_rate_pct": round(self.win_rate * 100, 2) if self.win_rate is not None else None,
                "annualized_return_pct": round(self.annualized_return * 100, 2) if self.annualized_return is not None else None,
                "profit_factor": round(self.profit_factor, 4) if self.profit_factor is not None else None,
                **_trade_metrics_as_dict(self.trade_records),
            },
            "trade_records": [trade.as_dict() for trade in self.trade_records],
            "equity_curve": [point.as_dict() for point in self.equity_curve],
        }


class LongOnlyBacktester:
    def __init__(self, initial_cash: float = 10_000.0, fee_rate: float = 0.001) -> None:
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate

    def run(self, candles: list[Candle], strategy: Strategy) -> BacktestResult:
        cash = self.initial_cash
        open_trade: _OpenTrade | None = None
        trade_records: list[TradeRecord] = []
        equity_curve: list[EquityPoint] = []

        for index in range(21, len(candles)):
            history = candles[: index + 1]
            latest = history[-1]
            signal = strategy.generate_signal(history)

            if open_trade is None and signal.action == "buy_candidate":
                open_trade, cash = _open_trade(cash, latest, signal, self.fee_rate)
            elif open_trade is not None:
                exit_price, exit_reason = _exit_trigger(open_trade, latest, candles[index - 1])
                if exit_price is not None:
                    trade, cash = _close_trade(open_trade, latest.timestamp, exit_price, exit_reason, cash, self.fee_rate)
                    trade_records.append(trade)
                    open_trade = None

            position_value = open_trade.quantity * latest.close if open_trade is not None else 0.0
            equity_curve.append(EquityPoint(timestamp=latest.timestamp, equity=cash + position_value))

        if candles and open_trade is not None:
            latest = candles[-1]
            trade, cash = _close_trade(open_trade, latest.timestamp, latest.close, "final_close", cash, self.fee_rate)
            trade_records.append(trade)
            equity_curve.append(EquityPoint(timestamp=latest.timestamp, equity=cash))

        final_equity = equity_curve[-1].equity if equity_curve else self.initial_cash
        max_drawdown = _max_drawdown([point.equity for point in equity_curve])
        total_return = final_equity / self.initial_cash - 1
        return BacktestResult(
            initial_cash=self.initial_cash,
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_drawdown,
            trades=len(trade_records),
            equity_curve=equity_curve,
            trade_records=trade_records,
            win_rate=_win_rate(trade_records),
            annualized_return=_annualized_return(total_return, candles),
            profit_factor=_profit_factor(trade_records),
        )


def _open_trade(cash: float, latest: Candle, signal: StrategySignal, fee_rate: float) -> tuple[_OpenTrade, float]:
    budget = cash * signal.max_position_pct
    entry_fee = budget * fee_rate
    quantity = (budget - entry_fee) / latest.close
    return (
        _OpenTrade(
            entry_time=latest.timestamp,
            entry_price=latest.close,
            quantity=quantity,
            entry_fee=entry_fee,
            invested_cash=budget,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_reason="; ".join(signal.reasons),
        ),
        cash - budget,
    )


def _close_trade(
    open_trade: _OpenTrade,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
    cash: float,
    fee_rate: float,
) -> tuple[TradeRecord, float]:
    proceeds = open_trade.quantity * exit_price
    exit_fee = proceeds * fee_rate
    pnl = proceeds - exit_fee - open_trade.invested_cash
    trade = TradeRecord(
        entry_time=open_trade.entry_time,
        exit_time=exit_time,
        entry_price=open_trade.entry_price,
        exit_price=exit_price,
        quantity=open_trade.quantity,
        entry_fee=open_trade.entry_fee,
        exit_fee=exit_fee,
        pnl=pnl,
        pnl_pct=pnl / open_trade.invested_cash if open_trade.invested_cash else 0.0,
        entry_reason=open_trade.entry_reason,
        exit_reason=exit_reason,
    )
    return trade, cash + proceeds - exit_fee


def _exit_trigger(open_trade: _OpenTrade, latest: Candle, previous: Candle) -> tuple[float | None, str | None]:
    if open_trade.stop_loss is not None and latest.low <= open_trade.stop_loss:
        return open_trade.stop_loss, "stop_loss"
    if open_trade.take_profit is not None and latest.high >= open_trade.take_profit:
        return open_trade.take_profit, "take_profit"
    if latest.close < previous.close * 0.97:
        return latest.close, "weakness"
    return None, None


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1)
    return abs(max_drawdown)


def _trade_metrics_as_dict(trade_records: list[TradeRecord]) -> dict:
    if not trade_records:
        return {
            "average_trade_return_pct": None,
            "average_win_pct": None,
            "average_loss_pct": None,
            "payoff_ratio": None,
            "average_holding_days": None,
        }

    returns = [trade.pnl_pct for trade in trade_records]
    wins = [trade.pnl_pct for trade in trade_records if trade.pnl > 0]
    losses = [abs(trade.pnl_pct) for trade in trade_records if trade.pnl < 0]
    holding_days = [(trade.exit_time - trade.entry_time).total_seconds() / 86400 for trade in trade_records]
    average_win = sum(wins) / len(wins) if wins else None
    average_loss = sum(losses) / len(losses) if losses else None
    return {
        "average_trade_return_pct": round(sum(returns) / len(returns) * 100, 2),
        "average_win_pct": round(average_win * 100, 2) if average_win is not None else None,
        "average_loss_pct": round(average_loss * 100, 2) if average_loss is not None else None,
        "payoff_ratio": round(average_win / average_loss, 4) if average_win is not None and average_loss not in (None, 0) else None,
        "average_holding_days": round(sum(holding_days) / len(holding_days), 2),
    }


def _win_rate(trade_records: list[TradeRecord]) -> float | None:
    if not trade_records:
        return None
    wins = sum(1 for trade in trade_records if trade.pnl > 0)
    return wins / len(trade_records)


def _profit_factor(trade_records: list[TradeRecord]) -> float | None:
    gross_profit = sum(trade.pnl for trade in trade_records if trade.pnl > 0)
    gross_loss = abs(sum(trade.pnl for trade in trade_records if trade.pnl < 0))
    if gross_profit == 0 and gross_loss == 0:
        return None
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _annualized_return(total_return: float, candles: list[Candle]) -> float | None:
    if len(candles) < 2:
        return None
    days = (candles[-1].timestamp - candles[0].timestamp).days
    if days <= 0:
        return None
    return (1 + total_return) ** (365 / days) - 1
