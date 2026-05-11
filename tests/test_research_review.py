from datetime import datetime, timedelta
from pathlib import Path

from market_monitor.data.csv_writer import write_candle_rows
from market_monitor.research_export import export_research_snapshot
from market_monitor.research_review import format_review_table, review_ai_candidates, review_research_snapshot, summarize_review


def _write_symbol_data(path, symbol="TEST", count=30):
    start = datetime(2024, 1, 1)
    rows = []
    for index in range(count):
        price = 100 + index
        rows.append(
            {
                "date": (start + timedelta(days=index)).date().isoformat(),
                "symbol": symbol,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1000,
                "turnover": "",
            }
        )
    write_candle_rows(path, rows)


def test_review_ai_candidates_calculates_future_returns(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_data(data_dir / "TEST.csv")
    snapshot = tmp_path / "snapshot"
    export_research_snapshot(
        snapshot,
        [],
        [],
        [{"symbol": "TEST", "name": "Test", "probability": 0.7, "latest_date": "2024-01-10", "model": "logistic"}],
        {},
    )

    rows = review_ai_candidates(snapshot / "ai_candidates.csv", data_dir, [5, 10])

    assert rows[0]["symbol"] == "TEST"
    assert rows[0]["return_5d_pct"] > 0
    assert rows[0]["hit_5d"] is True


def test_review_research_snapshot_writes_outputs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_symbol_data(data_dir / "TEST.csv")
    snapshot = tmp_path / "snapshot"
    export_research_snapshot(
        snapshot,
        [],
        [],
        [{"symbol": "TEST", "probability": 0.7, "latest_date": "2024-01-10"}],
        {},
    )

    result = review_research_snapshot(snapshot, data_dir, [5])

    assert (snapshot / "review.csv").exists()
    assert (snapshot / "review_summary.json").exists()
    assert result["summary"]["reviewed_count"] == 1


def test_summarize_review_and_format_table():
    rows = [{"symbol": "TEST", "return_5d_pct": 2.0, "hit_5d": True, "probability": 0.7, "entry_date": "2024-01-01", "entry_price": 100}]

    summary = summarize_review(rows, [5])
    table = format_review_table(rows, [5])

    assert summary["average_return_5d_pct"] == 2.0
    assert summary["hit_rate_5d_pct"] == 100.0
    assert "return_5d_pct" in table
