from __future__ import annotations

from market_monitor.signals.models import StrategySignal


def signal_with_metadata(signal: StrategySignal, metadata: dict[str, dict[str, str]]) -> dict:
    item_metadata = metadata.get(signal.symbol, {})
    output = signal.as_dict()
    output["name"] = item_metadata.get("name", "")
    output["market"] = item_metadata.get("market", "")
    return output


def format_signal_table(rows: list[dict]) -> str:
    headers = ["symbol", "name", "market", "signal", "confidence", "stop_loss", "take_profit", "reason"]
    table_rows = [_table_row(row) for row in rows]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in table_rows))
        for header in headers
    }
    lines = [" | ".join(header.ljust(widths[header]) for header in headers)]
    lines.append("-+-".join("-" * widths[header] for header in headers))
    lines.extend(" | ".join(str(row[header]).ljust(widths[header]) for header in headers) for row in table_rows)
    return "\n".join(lines)


def _table_row(row: dict) -> dict[str, str]:
    risk = row.get("risk", {})
    reasons = row.get("reasons", [])
    return {
        "symbol": str(row.get("symbol", "")),
        "name": str(row.get("name", "")),
        "market": str(row.get("market", "")),
        "signal": str(row.get("signal", "")),
        "confidence": _format_number(row.get("confidence")),
        "stop_loss": _format_number(risk.get("stop_loss")),
        "take_profit": _format_number(risk.get("take_profit")),
        "reason": "; ".join(str(reason) for reason in reasons[:2]),
    }


def _format_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float):
        return f"{value:.4g}"
    return str(value)
