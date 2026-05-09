from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float | None = None

    @classmethod
    def from_binance_kline(cls, symbol: str, row: list) -> "Candle":
        return cls(
            timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            symbol=symbol,
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            turnover=float(row[7]),
        )
