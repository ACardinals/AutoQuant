from __future__ import annotations

import csv
from datetime import date, datetime, time, timezone
from pathlib import Path

from market_monitor.data.csv_loader import load_candles_from_csv
from market_monitor.models import Candle
from market_monitor.research_export import write_dict_rows_csv, write_json


def review_research_snapshot(
    snapshot_dir: str | Path,
    data_dir: str | Path,
    horizons: list[int],
    threshold: float = 0.0,
    output_dir: str | Path | None = None,
) -> dict:
    snapshot = Path(snapshot_dir)
    rows = review_ai_candidates(snapshot / "ai_candidates.csv", data_dir, horizons, threshold)
    summary = summarize_review(rows, horizons)
    target_dir = Path(output_dir) if output_dir is not None else snapshot
    files = {
        "review": write_dict_rows_csv(target_dir / "review.csv", rows),
        "summary": write_json(target_dir / "review_summary.json", summary),
    }
    return {"output_dir": str(target_dir), "rows": rows, "summary": summary, "files": files}


def review_ai_candidates(
    candidates_path: str | Path,
    data_dir: str | Path,
    horizons: list[int],
    threshold: float = 0.0,
) -> list[dict]:
    candidates = _read_candidates(candidates_path)
    rows = []
    for candidate in candidates:
        symbol = candidate["symbol"].upper()
        try:
            candles = load_candles_from_csv(Path(data_dir) / f"{symbol}.csv", symbol=symbol)
            rows.append(review_candidate(candidate, candles, horizons, threshold))
        except (FileNotFoundError, ValueError, IndexError) as exc:
            rows.append({**_base_candidate_row(candidate), "error": str(exc)})
    return rows


def review_candidate(candidate: dict, candles: list[Candle], horizons: list[int], threshold: float = 0.0) -> dict:
    if not horizons:
        raise ValueError("horizons must not be empty")
    entry_date = _parse_date(candidate["latest_date"])
    entry_index = _entry_index(candles, entry_date)
    entry = candles[entry_index]
    row = {
        **_base_candidate_row(candidate),
        "entry_date": entry.timestamp.date().isoformat(),
        "entry_price": round(entry.close, 4),
    }
    max_horizon = max(horizons)
    window = candles[entry_index : min(len(candles), entry_index + max_horizon + 1)]
    row["max_drawdown_pct"] = _max_drawdown_pct(entry.close, window)
    for horizon in horizons:
        future_index = entry_index + horizon
        if future_index >= len(candles):
            row[f"return_{horizon}d_pct"] = None
            row[f"hit_{horizon}d"] = None
            continue
        future_return = candles[future_index].close / entry.close - 1 if entry.close else 0.0
        row[f"return_{horizon}d_pct"] = round(future_return * 100, 2)
        row[f"hit_{horizon}d"] = future_return > threshold
    return row


def summarize_review(rows: list[dict], horizons: list[int]) -> dict:
    summary = {"count": len(rows), "reviewed_count": sum(1 for row in rows if "error" not in row)}
    for horizon in horizons:
        returns = [row.get(f"return_{horizon}d_pct") for row in rows if row.get(f"return_{horizon}d_pct") is not None]
        hits = [row.get(f"hit_{horizon}d") for row in rows if row.get(f"hit_{horizon}d") is not None]
        summary[f"average_return_{horizon}d_pct"] = round(sum(returns) / len(returns), 2) if returns else None
        summary[f"hit_rate_{horizon}d_pct"] = round(sum(1 for hit in hits if hit) / len(hits) * 100, 2) if hits else None
    return summary


def format_review_table(rows: list[dict], horizons: list[int]) -> str:
    headers = ["symbol", "name", "probability", "entry_date", "entry_price"]
    for horizon in horizons:
        headers.extend([f"return_{horizon}d_pct", f"hit_{horizon}d"])
    headers.extend(["max_drawdown_pct", "error"])
    table_rows = [_format_row(row, headers) for row in rows]
    widths = {header: max(len(header), *(len(row[header]) for row in table_rows)) for header in headers}
    lines = [" | ".join(header.ljust(widths[header]) for header in headers)]
    lines.append("-+-".join("-" * widths[header] for header in headers))
    lines.extend(" | ".join(row[header].ljust(widths[header]) for header in headers) for row in table_rows)
    return "\n".join(lines)


def _read_candidates(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _base_candidate_row(candidate: dict) -> dict:
    return {
        "symbol": candidate.get("symbol", ""),
        "name": candidate.get("name", ""),
        "market": candidate.get("market", ""),
        "probability": _float_or_original(candidate.get("probability")),
        "model": candidate.get("model", ""),
        "latest_date": candidate.get("latest_date", ""),
    }


def _entry_index(candles: list[Candle], entry_date: date) -> int:
    entry_time = datetime.combine(entry_date, time.min, tzinfo=timezone.utc)
    for index, candle in enumerate(candles):
        if candle.timestamp >= entry_time:
            return index
    raise ValueError("entry date is after available candle history")


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value.strip()).date()


def _max_drawdown_pct(entry_price: float, candles: list[Candle]) -> float | None:
    if not candles or entry_price <= 0:
        return None
    lowest = min(candle.low for candle in candles)
    return round((lowest / entry_price - 1) * 100, 2)


def _float_or_original(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _format_row(row: dict, headers: list[str]) -> dict[str, str]:
    return {header: _format_value(row.get(header)) for header in headers}


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
