from __future__ import annotations

from market_monitor.backtest.engine import LongOnlyBacktester
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
        rows.append(
            {
                "strategy": strategy.name,
                "final_equity": output["final_equity"],
                "total_return_pct": output["total_return_pct"],
                "max_drawdown_pct": output["max_drawdown_pct"],
                "trades": output["trades"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "average_trade_return_pct": metrics["average_trade_return_pct"],
            }
        )
    return sorted(rows, key=lambda row: (row["total_return_pct"], -row["max_drawdown_pct"]), reverse=True)


def format_strategy_comparison_table(rows: list[dict]) -> str:
    headers = [
        "strategy",
        "total_return_pct",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "average_trade_return_pct",
        "final_equity",
    ]
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
