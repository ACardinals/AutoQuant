from argparse import Namespace

import pytest

from market_monitor.cli import _download_a_share, _sync_watchlist


def test_download_a_share_rejects_symbol_and_watchlist_together():
    args = Namespace(
        symbol="000001.SZ",
        watchlist="watchlist.csv",
        start_date="20240101",
        end_date="20240110",
        adjust="qfq",
        output=None,
        output_dir="data/a_share",
    )

    with pytest.raises(SystemExit):
        _download_a_share(args)


def test_download_a_share_rejects_watchlist_with_output_file():
    args = Namespace(
        symbol=None,
        watchlist="watchlist.csv",
        start_date="20240101",
        end_date="20240110",
        adjust="qfq",
        output="one.csv",
        output_dir="data/a_share",
    )

    with pytest.raises(SystemExit):
        _download_a_share(args)


def test_sync_watchlist_rejects_input_and_symbols_together():
    args = Namespace(input="raw.csv", symbol=["000001.SZ"], output="out.csv", data_dir="data/a_share", market="A股")

    with pytest.raises(SystemExit):
        _sync_watchlist(args)


def test_sync_watchlist_rejects_missing_source():
    args = Namespace(input=None, symbol=None, output="out.csv", data_dir="data/a_share", market="A股")

    with pytest.raises(SystemExit):
        _sync_watchlist(args)
