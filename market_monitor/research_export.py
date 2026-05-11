from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


def default_output_dir(base_dir: str | Path = "research_outputs", run_date: date | None = None) -> Path:
    current_date = run_date or date.today()
    return Path(base_dir) / current_date.isoformat()


def write_dict_rows_csv(path: str | Path, rows: list[dict]) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = _columns_from_rows(rows)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _serialize_cell(row.get(column)) for column in columns})
    return {"path": str(output_path), "rows": len(rows)}


def write_json(path: str | Path, payload: dict[str, Any]) -> dict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"path": str(output_path), "rows": 1}


def export_research_snapshot(
    output_dir: str | Path,
    strategy_screen_rows: list[dict],
    strategy_score_rows: list[dict],
    ai_candidate_rows: list[dict],
    metadata: dict[str, Any],
) -> dict:
    base = Path(output_dir)
    files = {
        "strategy_screen": write_dict_rows_csv(base / "strategy_screen.csv", strategy_screen_rows),
        "strategy_scores": write_dict_rows_csv(base / "strategy_scores.csv", strategy_score_rows),
        "ai_candidates": write_dict_rows_csv(base / "ai_candidates.csv", ai_candidate_rows),
        "metadata": write_json(base / "metadata.json", metadata),
    }
    return {"output_dir": str(base), "files": files}


def _columns_from_rows(rows: list[dict]) -> list[str]:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns or ["empty"]


def _serialize_cell(value) -> str:
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    if value is None:
        return ""
    return str(value)
