from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from market_monitor.models import Candle


_TIMESTAMP_COLUMNS = ("timestamp", "date", "datetime", "time")


def load_candles_from_csv(path: str | Path, symbol: str | None = None) -> list[Candle]:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        candles = [_row_to_candle(row, symbol) for row in reader]
    return sorted(candles, key=lambda candle: candle.timestamp)


def _row_to_candle(row: dict[str, str], fallback_symbol: str | None) -> Candle:
    normalized = {_normalize_key(key): value for key, value in row.items() if key is not None}
    symbol = (normalized.get("symbol") or fallback_symbol or "UNKNOWN").upper()
    return Candle(
        timestamp=_parse_timestamp(_first_present(normalized, _TIMESTAMP_COLUMNS)),
        symbol=symbol,
        open=float(normalized["open"]),
        high=float(normalized["high"]),
        low=float(normalized["low"]),
        close=float(normalized["close"]),
        volume=float(normalized["volume"]),
        turnover=float(normalized["turnover"]) if normalized.get("turnover") else None,
    )


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _first_present(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    raise ValueError(f"CSV row is missing one of: {', '.join(keys)}")


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.isdigit():
        number = int(text)
        if number > 10_000_000_000:
            number = number // 1000
        return datetime.fromtimestamp(number, tz=timezone.utc)

    iso_text = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(iso_text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
