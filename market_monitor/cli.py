from __future__ import annotations

import argparse
import json

from market_monitor.backtest.engine import LongOnlyBacktester
from market_monitor.data.akshare_a_share import (
    default_a_share_output_path,
    download_a_share_history,
    download_a_share_watchlist,
)
from market_monitor.data.csv_loader import load_candles_from_csv
from market_monitor.data.watchlist import (
    WatchlistItem,
    create_watchlist_from_symbols,
    load_watchlist_candles,
    sync_watchlist_paths,
)
from market_monitor.evaluation import compare_strategies, format_strategy_comparison_table
from market_monitor.rl.baselines import available_policies, create_policy, evaluate_policy
from market_monitor.rl.environment import TradingEnvironmentConfig, TradingEnvironmentPlaceholder
from market_monitor.signals.formatters import format_signal_table, signal_with_metadata
from market_monitor.signals.screener import screen_watchlist
from market_monitor.strategies.registry import available_strategies, create_strategy


def main() -> None:
    parser = argparse.ArgumentParser(prog="market-monitor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Fetch candles and run a strategy backtest")
    backtest_parser.add_argument("--symbol", default="BTCUSDT")
    backtest_parser.add_argument("--interval", default="1h")
    backtest_parser.add_argument("--limit", type=int, default=300)
    backtest_parser.add_argument("--strategy", default="breakout")
    backtest_parser.add_argument("--initial-cash", type=float, default=10_000.0)
    backtest_parser.add_argument("--csv", help="Load candles from a local CSV instead of Binance")

    signal_parser = subparsers.add_parser("signal", help="Fetch candles and print the latest strategy signal")
    signal_parser.add_argument("--symbol", default="BTCUSDT")
    signal_parser.add_argument("--interval", default="1h")
    signal_parser.add_argument("--limit", type=int, default=120)
    signal_parser.add_argument("--strategy", default="breakout")
    signal_parser.add_argument("--csv", help="Load candles from a local CSV instead of Binance")

    screen_parser = subparsers.add_parser("screen", help="Run a strategy over a watchlist of local CSV files")
    screen_parser.add_argument("--csv", action="append", help="CSV path or SYMBOL=CSV path")
    screen_parser.add_argument("--watchlist", help="Watchlist CSV with symbol,csv,name,market columns")
    screen_parser.add_argument("--strategy", default="breakout")
    screen_parser.add_argument("--format", choices=("json", "table"), default="json")

    download_parser = subparsers.add_parser("download-a-share", help="Download A-share historical daily candles to CSV")
    download_parser.add_argument("--symbol")
    download_parser.add_argument("--watchlist")
    download_parser.add_argument("--start-date", required=True)
    download_parser.add_argument("--end-date", required=True)
    download_parser.add_argument("--adjust", default="qfq")
    download_parser.add_argument("--output")
    download_parser.add_argument("--output-dir", default="data/a_share")

    sync_parser = subparsers.add_parser("sync-watchlist", help="Create or refresh watchlist CSV paths")
    sync_parser.add_argument("--input")
    sync_parser.add_argument("--symbol", action="append")
    sync_parser.add_argument("--output", required=True)
    sync_parser.add_argument("--data-dir", default="data/a_share")
    sync_parser.add_argument("--market", default="A股")

    compare_parser = subparsers.add_parser("compare-strategies", help="Compare strategies on the same local CSV candles")
    compare_parser.add_argument("--csv", required=True)
    compare_parser.add_argument("--symbol", required=True)
    compare_parser.add_argument("--strategy", action="append", help="Strategy name to include; repeat to compare a subset")
    compare_parser.add_argument("--initial-cash", type=float, default=10_000.0)
    compare_parser.add_argument("--format", choices=("json", "table"), default="json")

    subparsers.add_parser("strategies", help="List available strategy names")

    rl_baseline_parser = subparsers.add_parser("rl-baseline", help="Evaluate a simple RL baseline policy on local CSV candles")
    rl_baseline_parser.add_argument("--csv", required=True)
    rl_baseline_parser.add_argument("--symbol", required=True)
    rl_baseline_parser.add_argument("--policy", choices=available_policies(), default="buy_and_hold")
    rl_baseline_parser.add_argument("--seed", type=int, default=0)
    rl_baseline_parser.add_argument("--window-size", type=int, default=50)
    rl_baseline_parser.add_argument("--initial-cash", type=float, default=10_000.0)

    subparsers.add_parser("rl-spec", help="Print the planned reinforcement-learning environment shape")

    args = parser.parse_args()

    if args.command == "rl-baseline":
        candles = load_candles_from_csv(args.csv, symbol=args.symbol)
        policy = create_policy(args.policy, args.seed)
        summary = evaluate_policy(
            candles,
            policy,
            TradingEnvironmentConfig(initial_cash=args.initial_cash, window_size=args.window_size),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "rl-spec":
        print(json.dumps(TradingEnvironmentPlaceholder().describe(), ensure_ascii=False, indent=2))
        return

    if args.command == "compare-strategies":
        candles = load_candles_from_csv(args.csv, symbol=args.symbol)
        rows = compare_strategies(candles, args.strategy, args.initial_cash)
        if args.format == "table":
            print(format_strategy_comparison_table(rows))
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.command == "strategies":
        print(json.dumps(available_strategies(), ensure_ascii=False, indent=2))
        return

    if args.command == "sync-watchlist":
        summary = _sync_watchlist(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "download-a-share":
        summary = _download_a_share(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    strategy = create_strategy(args.strategy)

    if args.command == "screen":
        rows = _screen_rows(args, strategy)
        if args.format == "table":
            print(format_signal_table(rows))
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    candles = _load_candles(args.symbol, args.interval, args.limit, args.csv)

    if args.command == "signal":
        signal = strategy.generate_signal(candles)
        print(json.dumps(signal.as_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "backtest":
        result = LongOnlyBacktester(initial_cash=args.initial_cash).run(candles, strategy)
        latest_signal = strategy.generate_signal(candles)
        output = {
            "symbol": candles[-1].symbol if candles else args.symbol.upper(),
            "interval": args.interval,
            "strategy": strategy.name,
            "backtest": result.as_dict(),
            "latest_signal": latest_signal.as_dict(),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


def _sync_watchlist(args):
    if args.input and args.symbol:
        raise SystemExit("sync-watchlist accepts either --input or --symbol, not both")
    if args.input:
        return sync_watchlist_paths(args.input, args.output, args.data_dir)
    if args.symbol:
        return create_watchlist_from_symbols(args.symbol, args.output, args.data_dir, args.market)
    raise SystemExit("sync-watchlist requires --input or at least one --symbol")


def _download_a_share(args):
    if args.symbol and args.watchlist:
        raise SystemExit("download-a-share accepts either --symbol or --watchlist, not both")
    if args.watchlist:
        if args.output:
            raise SystemExit("download-a-share --watchlist uses --output-dir, not --output")
        return download_a_share_watchlist(
            args.watchlist,
            args.start_date,
            args.end_date,
            args.output_dir,
            args.adjust,
        )
    if args.symbol:
        output = args.output or default_a_share_output_path(args.symbol)
        return download_a_share_history(args.symbol, args.start_date, args.end_date, output, args.adjust)
    raise SystemExit("download-a-share requires --symbol or --watchlist")


def _screen_rows(args, strategy):
    items, candles_by_symbol = _load_screen_inputs(args)
    metadata = {item.symbol: {"name": item.name, "market": item.market} for item in items}
    signals = screen_watchlist(candles_by_symbol, strategy)
    return [signal_with_metadata(signal, metadata) for signal in signals]


def _load_screen_inputs(args):
    if args.watchlist and args.csv:
        raise SystemExit("screen accepts either --watchlist or --csv, not both")
    if args.watchlist:
        return load_watchlist_candles(args.watchlist)
    if args.csv:
        return _load_screen_csvs(args.csv)
    raise SystemExit("screen requires --watchlist or at least one --csv")


def _load_candles(symbol: str, interval: str, limit: int, csv_path: str | None):
    if csv_path:
        return load_candles_from_csv(csv_path, symbol=symbol)

    from market_monitor.data.binance import BinanceClient

    return BinanceClient().fetch_klines(symbol, interval, limit)


def _load_screen_csvs(csv_specs: list[str]):
    items = []
    candles_by_symbol = {}
    for spec in csv_specs:
        if "=" in spec:
            symbol, path = spec.split("=", 1)
            candles = load_candles_from_csv(path, symbol=symbol)
        else:
            path = spec
            candles = load_candles_from_csv(path)
            symbol = candles[-1].symbol if candles else spec
        symbol = symbol.upper()
        items.append(WatchlistItem(symbol=symbol, csv_path=path))
        candles_by_symbol[symbol] = candles
    return items, candles_by_symbol


if __name__ == "__main__":
    main()
