from market_monitor.data.csv_loader import load_candles_from_csv


def test_load_candles_from_csv_with_symbol_column(tmp_path):
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "timestamp,symbol,open,high,low,close,volume,turnover\n"
        "2024-01-02T00:00:00Z,BTCUSDT,101,105,100,104,200,20800\n"
        "2024-01-01T00:00:00Z,BTCUSDT,100,102,99,101,120,12120\n",
        encoding="utf-8",
    )

    candles = load_candles_from_csv(csv_path)

    assert [candle.close for candle in candles] == [101.0, 104.0]
    assert candles[0].symbol == "BTCUSDT"
    assert candles[0].turnover == 12120.0


def test_load_candles_from_csv_uses_fallback_symbol(tmp_path):
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-01,100,102,99,101,120\n",
        encoding="utf-8",
    )

    candles = load_candles_from_csv(csv_path, symbol="ethusdt")

    assert candles[0].symbol == "ETHUSDT"
    assert candles[0].timestamp.tzinfo is not None
