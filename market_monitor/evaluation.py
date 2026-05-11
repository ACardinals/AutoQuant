from __future__ import annotations

from pathlib import Path

from market_monitor.backtest.engine import LongOnlyBacktester
from market_monitor.data.watchlist import WatchlistItem
from market_monitor.models import Candle
from market_monitor.strategies.registry import available_strategies, create_strategy


def compare_strategies(
    candles: list[Candle],
    strategy_names: list[str] | None = None,
    initial_cash: float = 10_000.0,
) -> list[dict]:
    names = strategy_names or available_strategies()
    rows = []
    for name in names:
        strategy = create_strategy(name)
        result = LongOnlyBacktester(initial_cash=initial_cash).run(candles, strategy)
        output = result.as_dict()
        metrics = output["metrics"]
        row = {
            "strategy": strategy.name,
            "final_equity": output["final_equity"],
            "total_return_pct": output["total_return_pct"],
            "max_drawdown_pct": output["max_drawdown_pct"],
            "trades": output["trades"],
            "win_rate_pct": metrics["win_rate_pct"],
            "profit_factor": metrics["profit_factor"],
            "average_trade_return_pct": metrics["average_trade_return_pct"],
        }
        row["score"] = score_strategy_row(row)
        rows.append(row)
    return _sort_rows(rows)


def compare_watchlist(
    items: list[WatchlistItem],
    candles_by_symbol: dict[str, list[Candle]],
    strategy_names: list[str] | None = None,
    initial_cash: float = 10_000.0,
) -> list[dict]:
    metadata = {item.symbol: item for item in items}
    rows = []
    for symbol, candles in candles_by_symbol.items():
        item = metadata.get(symbol, WatchlistItem(symbol=symbol, csv_path=Path()))
        for row in compare_strategies(candles, strategy_names, initial_cash):
            rows.append(
                {
                    "symbol": symbol,
                    "name": item.name,
                    "market": item.market,
                    **row,
                }
            )
    return _sort_rows(rows)


def score_strategy_row(row: dict) -> float:
    total_return = _value(row.get("total_return_pct"))
    max_drawdown = _value(row.get("max_drawdown_pct"))
    win_rate = _value(row.get("win_rate_pct"), default=50.0)
    profit_factor = min(_value(row.get("profit_factor"), default=1.0), 3.0)
    average_trade_return = _value(row.get("average_trade_return_pct"))
    trades = int(row.get("trades") or 0)
    trade_factor = min(trades / 5, 1.0)

    score = total_return * 3
    score -= max_drawdown * 1.5
    score += (win_rate - 50) * 0.08
    score += (profit_factor - 1) * 4
    score += average_trade_return * 1.5
    score += trade_factor
    return round(score, 4)


def format_strategy_comparison_table(rows: list[dict]) -> str:
    headers = [
        "strategy",
        "score",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "average_trade_return_pct",
        "final_equity",
    ]
    return _format_table(rows, headers)


def format_watchlist_comparison_table(rows: list[dict]) -> str:
    headers = [
        "symbol",
        "name",
        "market",
        "strategy",
        "score",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
    ]
    return _format_table(rows, headers)


def _sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (row["score"], row["total_return_pct"], -row["max_drawdown_pct"]),
        reverse=True,
    )


def _format_table(rows: list[dict], headers: list[str]) -> str:
    table_rows = [_format_row(row, headers) for row in rows]
    widths = {header: max(len(header), *(len(row[header]) for row in table_rows)) for header in headers}
    lines = [" | ".join(header.ljust(widths[header]) for header in headers)]
    lines.append("-+-".join("-" * widths[header] for header in headers))
    lines.extend(" | ".join(row[header].ljust(widths[header]) for header in headers) for row in table_rows)
    return "\n".join(lines)


def _format_row(row: dict, headers: list[str]) -> dict[str, str]:
    return {header: _format_value(row.get(header)) for header in headers}


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _value(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)
