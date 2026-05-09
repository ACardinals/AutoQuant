from market_monitor.data.akshare_a_share import (
    default_a_share_output_path,
    download_a_share_watchlist,
    normalize_akshare_history_rows,
    normalize_a_share_spot_rows,
    normalize_a_share_symbol,
    to_akshare_symbol,
)





def test_a_share_symbol_normalization():
    assert normalize_a_share_symbol("000001.sz") == "000001.SZ"
    assert normalize_a_share_symbol("600519") == "600519.SH"
    assert normalize_a_share_symbol("000001") == "000001.SZ"
    assert to_akshare_symbol("600519.SH") == "600519"


def test_normalize_a_share_spot_rows():
    items = normalize_a_share_spot_rows([
        {"代码": "1", "名称": "平安银行"},
        {"代码": "600519", "名称": "贵州茅台"},
    ])

    assert items[0].symbol == "000001.SZ"
    assert items[0].name == "平安银行"
    assert items[1].symbol == "600519.SH"


def test_normalize_akshare_history_rows_from_records():
    rows = normalize_akshare_history_rows(
        "000001.SZ",
        [
            {
                "日期": "2024-01-02",
                "开盘": 10,
                "最高": 11,
                "最低": 9,
                "收盘": 10.5,
                "成交量": 1000,
                "成交额": 10500,
            }
        ],
    )

    assert rows == [
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
    ]


def test_default_a_share_output_path():
    assert str(default_a_share_output_path("000001.SZ")).replace("\\", "/") == "data/a_share/000001.SZ.csv"


def test_download_a_share_watchlist_writes_each_symbol(tmp_path, monkeypatch):
    watchlist = tmp_path / "watchlist.csv"
    watchlist.write_text(
        "symbol,name,market,csv\n"
        "000001.SZ,平安银行,A股,unused.csv\n"
        "600519.SH,贵州茅台,A股,unused.csv\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "downloads"

    def fake_fetch(symbol, start_date, end_date, adjust="qfq"):
        return [
            {
                "date": "2024-01-02",
                "symbol": normalize_a_share_symbol(symbol),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
                "turnover": 10500.0,
            }
        ]

    monkeypatch.setattr("market_monitor.data.akshare_a_share.fetch_a_share_history", fake_fetch)

    summary = download_a_share_watchlist(watchlist, "20240101", "20240110", output_dir)

    assert summary["count"] == 2
    assert (output_dir / "000001.SZ.csv").exists()
    assert (output_dir / "600519.SH.csv").exists()
