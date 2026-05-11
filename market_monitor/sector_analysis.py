from __future__ import annotations

from collections import defaultdict


def score_sectors(strategy_rows: list[dict], ai_rows: list[dict] | None = None) -> list[dict]:
    ai_by_sector = _group_rows(ai_rows or [])
    rows = []
    for sector, sector_strategy_rows in _group_rows(strategy_rows).items():
        sector_ai_rows = ai_by_sector.get(sector, [])
        symbols = {row.get("symbol", "") for row in sector_strategy_rows if row.get("symbol")}
        average_strategy_score = _average(row.get("score") for row in sector_strategy_rows)
        average_probability = _average(row.get("probability") for row in sector_ai_rows)
        sector_score = _sector_score(average_strategy_score, average_probability)
        rows.append(
            {
                "sector": sector,
                "symbol_count": len(symbols),
                "strategy_rows": len(sector_strategy_rows),
                "ai_candidates": len(sector_ai_rows),
                "sector_score": sector_score,
                "average_strategy_score": round(average_strategy_score, 4) if average_strategy_score is not None else None,
                "best_strategy_score": _max_value(row.get("score") for row in sector_strategy_rows),
                "average_total_return_pct": _round_optional(_average(row.get("total_return_pct") for row in sector_strategy_rows)),
                "average_max_drawdown_pct": _round_optional(_average(row.get("max_drawdown_pct") for row in sector_strategy_rows)),
                "average_probability": round(average_probability, 4) if average_probability is not None else None,
            }
        )
    return sorted(rows, key=lambda row: (row["sector_score"], row["average_strategy_score"] or 0), reverse=True)


def top_symbols_by_sector(strategy_rows: list[dict], top_n: int = 5) -> list[dict]:
    rows = []
    for sector, sector_rows in _group_rows(strategy_rows).items():
        sorted_rows = sorted(sector_rows, key=lambda row: (float(row.get("score") or 0), float(row.get("total_return_pct") or 0)), reverse=True)
        for rank, row in enumerate(sorted_rows[:top_n], start=1):
            rows.append(
                {
                    "sector": sector,
                    "rank": rank,
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "strategy": row.get("strategy", ""),
                    "score": row.get("score"),
                    "total_return_pct": row.get("total_return_pct"),
                    "max_drawdown_pct": row.get("max_drawdown_pct"),
                }
            )
    return rows


def format_sector_table(rows: list[dict]) -> str:
    headers = [
        "sector",
        "sector_score",
        "symbol_count",
        "average_strategy_score",
        "best_strategy_score",
        "average_probability",
        "average_total_return_pct",
        "average_max_drawdown_pct",
    ]
    return _format_table(rows, headers)


def format_sector_candidates_table(rows: list[dict]) -> str:
    headers = ["sector", "rank", "symbol", "name", "strategy", "score", "total_return_pct", "max_drawdown_pct"]
    return _format_table(rows, headers)


def _group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("market") or row.get("sector") or "未分类"].append(row)
    return dict(grouped)


def _sector_score(average_strategy_score: float | None, average_probability: float | None) -> float:
    score = average_strategy_score or 0.0
    if average_probability is not None:
        score += (average_probability - 0.5) * 20
    return round(score, 4)


def _average(values) -> float | None:
    present = [_to_float(value) for value in values if _to_float(value) is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _max_value(values) -> float | None:
    present = [_to_float(value) for value in values if _to_float(value) is not None]
    return round(max(present), 4) if present else None


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
