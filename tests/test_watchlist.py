from market_monitor.data.watchlist import (
    create_watchlist_from_symbols,
    load_watchlist,
    load_watchlist_candles,
    sync_watchlist_paths,
)


def test_load_watchlist_resolves_relative_csv_paths(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    candle_path = data_dir / "pingan.csv"
    candle_path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-01,10,11,9,10.5,1000\n",
        encoding="utf-8",
    )
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\n"
        "000001.SZ,平安银行,A股,data/pingan.csv\n",
        encoding="utf-8",
    )

    items = load_watchlist(watchlist_path)
    loaded_items, candles_by_symbol = load_watchlist_candles(watchlist_path)

    assert items[0].symbol == "000001.SZ"
    assert items[0].name == "平安银行"
    assert items[0].market == "A股"
    assert items[0].csv_path == candle_path
    assert loaded_items == items
    assert candles_by_symbol["000001.SZ"][0].symbol == "000001.SZ"


def test_sync_watchlist_paths_preserves_metadata(tmp_path):
    input_path = tmp_path / "raw.csv"
    input_path.write_text(
        "symbol,name,market,csv\n"
        "000001.SZ,平安银行,A股,old.csv\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "synced.csv"

    summary = sync_watchlist_paths(input_path, output_path, "data/a_share")
    items = load_watchlist(output_path)

    assert summary["count"] == 1
    assert items[0].symbol == "000001.SZ"
    assert items[0].name == "平安银行"
    assert items[0].market == "A股"
    assert str(items[0].csv_path).replace("\\", "/").endswith("data/a_share/000001.SZ.csv")


def test_create_watchlist_from_symbols(tmp_path):
    output_path = tmp_path / "symbols.csv"

    summary = create_watchlist_from_symbols(["000001.SZ", "600519.SH"], output_path, "data/a_share")
    items = load_watchlist(output_path)

    assert summary["count"] == 2
    assert [item.symbol for item in items] == ["000001.SZ", "600519.SH"]
    assert all(item.market == "A股" for item in items)
