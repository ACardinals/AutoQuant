from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from market_monitor.data.csv_loader import load_candles_from_csv
from market_monitor.models import Candle


WATCHLIST_COLUMNS = ("symbol", "name", "market", "csv")


@dataclass(frozen=True)
class WatchlistItem:
    symbol: str
    csv_path: Path
    name: str = ""
    market: str = ""


def load_watchlist(path: str | Path) -> list[WatchlistItem]:
    watchlist_path = Path(path)
    with watchlist_path.open(newline="", encoding="utf-8-sig") as file:
        return [_row_to_item(row, watchlist_path.parent) for row in csv.DictReader(file)]


def load_watchlist_candles(path: str | Path) -> tuple[list[WatchlistItem], dict[str, list[Candle]]]:
    items = load_watchlist(path)
    candles_by_symbol = {
        item.symbol: load_candles_from_csv(item.csv_path, symbol=item.symbol)
        for item in items
    }
    return items, candles_by_symbol


def write_watchlist(path: str | Path, items: list[WatchlistItem]) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=WATCHLIST_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "market": item.market,
                    "csv": str(item.csv_path),
                }
            )
    return {"output": str(output_path), "count": len(items)}


def sync_watchlist_paths(input_path: str | Path, output_path: str | Path, data_dir: str | Path) -> dict:
    items = [
        WatchlistItem(
            symbol=item.symbol,
            name=item.name,
            market=item.market,
            csv_path=_symbol_csv_path(data_dir, item.symbol),
        )
        for item in load_watchlist(input_path)
    ]
    summary = write_watchlist(output_path, items)
    summary["data_dir"] = str(Path(data_dir))
    return summary


def create_watchlist_from_symbols(
    symbols: list[str],
    output_path: str | Path,
    data_dir: str | Path,
    market: str = "A股",
) -> dict:
    items = [
        WatchlistItem(
            symbol=symbol.strip().upper(),
            name="",
            market=market,
            csv_path=_symbol_csv_path(data_dir, symbol.strip().upper()),
        )
        for symbol in symbols
    ]
    summary = write_watchlist(output_path, items)
    summary["data_dir"] = str(Path(data_dir))
    return summary


def _symbol_csv_path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / f"{symbol}.csv"


def _row_to_item(row: dict[str, str], base_dir: Path) -> WatchlistItem:
    normalized = {_normalize_key(key): value.strip() for key, value in row.items() if key is not None and value is not None}
    symbol = normalized["symbol"].upper()
    csv_path = Path(normalized["csv"])
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    return WatchlistItem(
        symbol=symbol,
        name=normalized.get("name", ""),
        market=normalized.get("market", ""),
        csv_path=csv_path,
    )


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")
