from __future__ import annotations

import csv
from pathlib import Path


STANDARD_CANDLE_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume", "turnover")


def write_candle_rows(path: str | Path, rows: list[dict]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STANDARD_CANDLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in STANDARD_CANDLE_COLUMNS})
    return len(rows)
