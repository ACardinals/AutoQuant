from market_monitor.data.csv_loader import load_candles_from_csv
from market_monitor.data.csv_writer import write_candle_rows


def test_write_candle_rows_uses_standard_schema(tmp_path):
    output = tmp_path / "nested" / "candles.csv"

    rows_written = write_candle_rows(output, [
        {
            "date": "2024-01-02",
            "symbol": "000001.SZ",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
            "turnover": 10500.0,
        }
    ])

    candles = load_candles_from_csv(output)

    assert rows_written == 1
    assert candles[0].symbol == "000001.SZ"
    assert candles[0].close == 10.5
