from __future__ import annotations

from pathlib import Path

from market_monitor.data.csv_writer import write_candle_rows
from market_monitor.data.watchlist import WatchlistItem, load_watchlist


_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "turnover",
}


def download_a_share_history(
    symbol: str,
    start_date: str,
    end_date: str,
    output: str | Path,
    adjust: str = "qfq",
) -> dict:
    rows = fetch_a_share_history(symbol, start_date, end_date, adjust)
    rows_written = write_candle_rows(output, rows)
    return {
        "symbol": normalize_a_share_symbol(symbol),
        "output": str(Path(output)),
        "rows": rows_written,
        "start_date": start_date,
        "end_date": end_date,
        "adjust": adjust,
    }


def download_a_share_watchlist(
    watchlist: str | Path,
    start_date: str,
    end_date: str,
    output_dir: str | Path = Path("data") / "a_share",
    adjust: str = "qfq",
    max_symbols: int | None = None,
) -> dict:
    output_base = Path(output_dir)
    items = load_watchlist(watchlist)
    if max_symbols is not None and max_symbols > 0:
        items = items[:max_symbols]
    results = [
        download_a_share_history(
            item.symbol,
            start_date,
            end_date,
            output_base / f"{normalize_a_share_symbol(item.symbol)}.csv",
            adjust,
        )
        for item in items
    ]
    return {
        "watchlist": str(Path(watchlist)),
        "output_dir": str(output_base),
        "count": len(results),
        "results": results,
    }




def fetch_a_share_spot_universe() -> list[WatchlistItem]:
    import akshare as ak

    frame = ak.stock_zh_a_spot_em()
    return normalize_a_share_spot_rows(frame)


def normalize_a_share_spot_rows(rows) -> list[WatchlistItem]:
    items = []
    for row in _iter_rows(rows):
        normalized = {_normalize_key(key): value for key, value in row.items()}
        code = str(_first_present(normalized, ("代码", "code", "证券代码"))).zfill(6)
        name = str(_first_present(normalized, ("名称", "name", "证券简称")))
        symbol = normalize_a_share_symbol(code)
        items.append(WatchlistItem(symbol=symbol, name=name, market="A股", csv_path=Path(f"{symbol}.csv")))
    return items


def _normalize_key(key: str) -> str:
    return key.strip()


def _first_present(row: dict, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    raise KeyError(f"Missing one of columns: {', '.join(keys)}")


def fetch_a_share_history(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> list[dict]:
    import akshare as ak

    frame = ak.stock_zh_a_hist(
        symbol=to_akshare_symbol(symbol),
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )
    return normalize_akshare_history_rows(symbol, frame)


def normalize_akshare_history_rows(symbol: str, rows) -> list[dict]:
    normalized_symbol = normalize_a_share_symbol(symbol)
    normalized_rows = []
    for row in _iter_rows(rows):
        normalized = {_COLUMN_MAP.get(key, key): value for key, value in row.items()}
        normalized_rows.append(
            {
                "date": str(normalized["date"]),
                "symbol": normalized_symbol,
                "open": float(normalized["open"]),
                "high": float(normalized["high"]),
                "low": float(normalized["low"]),
                "close": float(normalized["close"]),
                "volume": float(normalized["volume"]),
                "turnover": float(normalized["turnover"]) if normalized.get("turnover") not in (None, "") else "",
            }
        )
    return normalized_rows


def normalize_a_share_symbol(symbol: str) -> str:
    text = symbol.strip().upper()
    if "." in text:
        code, exchange = text.split(".", 1)
        return f"{code}.{exchange}"
    exchange = "SH" if text.startswith(("5", "6", "9")) else "SZ"
    return f"{text}.{exchange}"


def to_akshare_symbol(symbol: str) -> str:
    return normalize_a_share_symbol(symbol).split(".", 1)[0]


def default_a_share_output_path(symbol: str) -> Path:
    return Path("data") / "a_share" / f"{normalize_a_share_symbol(symbol)}.csv"


def _iter_rows(rows):
    if hasattr(rows, "to_dict"):
        yield from rows.to_dict("records")
        return
    yield from rows
