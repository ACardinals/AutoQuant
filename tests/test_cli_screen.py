from argparse import Namespace

from market_monitor.cli import _screen_rows
from market_monitor.strategies.breakout import BreakoutStrategy


def test_screen_rows_loads_watchlist_metadata(tmp_path):
    candle_path = tmp_path / "btc.csv"
    rows = ["timestamp,symbol,open,high,low,close,volume"]
    for i in range(25):
        rows.append(f"2024-01-{i + 1:02d}T00:00:00Z,BTCUSDT,{100 + i},{101 + i},{99 + i},{100 + i},100")
    rows[-1] = "2024-01-25T00:00:00Z,BTCUSDT,130,135,129,134,300"
    candle_path.write_text("\n".join(rows), encoding="utf-8")
    watchlist_path = tmp_path / "watchlist.csv"
    watchlist_path.write_text(
        "symbol,name,market,csv\nBTCUSDT,Bitcoin,crypto,btc.csv\n",
        encoding="utf-8",
    )

    rows = _screen_rows(Namespace(watchlist=str(watchlist_path), csv=None), BreakoutStrategy())

    assert rows[0]["symbol"] == "BTCUSDT"
    assert rows[0]["name"] == "Bitcoin"
    assert rows[0]["market"] == "crypto"
    assert rows[0]["signal"] == "buy_candidate"
