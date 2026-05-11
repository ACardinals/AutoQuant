from datetime import date
import json

from market_monitor.research_export import default_output_dir, export_research_snapshot, write_dict_rows_csv, write_json


def test_default_output_dir_uses_run_date():
    assert str(default_output_dir("outputs", date(2024, 1, 2))).endswith("outputs\\2024-01-02") or str(
        default_output_dir("outputs", date(2024, 1, 2))
    ).endswith("outputs/2024-01-02")


def test_write_dict_rows_csv_and_json(tmp_path):
    csv_summary = write_dict_rows_csv(tmp_path / "rows.csv", [{"symbol": "TEST", "reasons": ["a", "b"]}])
    json_summary = write_json(tmp_path / "meta.json", {"symbol": "TEST"})

    assert csv_summary["rows"] == 1
    assert "TEST" in (tmp_path / "rows.csv").read_text(encoding="utf-8-sig")
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["symbol"] == "TEST"
    assert json_summary["rows"] == 1


def test_export_research_snapshot_writes_expected_files(tmp_path):
    summary = export_research_snapshot(
        tmp_path / "snapshot",
        [{"symbol": "TEST", "signal": "hold"}],
        [{"symbol": "TEST", "score": 1.2}],
        [{"symbol": "TEST", "probability": 0.6}],
        {"strategy": "ma_trend"},
    )

    assert (tmp_path / "snapshot" / "strategy_screen.csv").exists()
    assert (tmp_path / "snapshot" / "strategy_scores.csv").exists()
    assert (tmp_path / "snapshot" / "ai_candidates.csv").exists()
    assert (tmp_path / "snapshot" / "metadata.json").exists()
    assert summary["files"]["strategy_screen"]["rows"] == 1
