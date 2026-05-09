# AutoQuant / Market Monitor

A Python research project for A-share and crypto market monitoring, rule-based strategy screening, backtesting, and future reinforcement-learning experiments.

The project is intentionally research-first: it focuses on data ingestion, explainable strategy signals, historical evaluation, and RL-friendly environment design before any live trading integration.

## Current capabilities

- Fetch Binance public OHLCV candles.
- Download A-share daily candles with AkShare.
- Load local CSV candles for reproducible research.
- Screen watchlists with multiple rule-based strategies.
- Backtest long-only strategy signals with risk levels and summary metrics.
- Compare strategies on the same symbol/history from the CLI.
- Run a Streamlit dashboard for A-share screening, K-line inspection, and backtest review.
- Evaluate simple RL baseline policies on the trading environment.

## Installation

```bash
cd market-monitor
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
```

On Linux/macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Run tests

```bash
python -m pytest tests
```

If running from the parent directory, target this project's tests explicitly:

```bash
python -m pytest market-monitor/tests
```

## CLI usage

List available strategies:

```bash
python -m market_monitor.cli strategies
```

Screen a watchlist with a local CSV-backed universe:

```bash
python -m market_monitor.cli screen \
  --watchlist watchlists/a_share.csv \
  --strategy ma_trend \
  --format table
```

Backtest one symbol from a local CSV:

```bash
python -m market_monitor.cli backtest \
  --csv examples/data/a_share/000001.SZ.csv \
  --symbol 000001.SZ \
  --strategy bollinger_reversion
```

Compare all strategies on the same symbol/history:

```bash
python -m market_monitor.cli compare-strategies \
  --csv examples/data/a_share/000001.SZ.csv \
  --symbol 000001.SZ \
  --format table
```

Evaluate an RL baseline policy:

```bash
python -m market_monitor.cli rl-baseline \
  --csv examples/data/a_share/000001.SZ.csv \
  --symbol 000001.SZ \
  --policy buy_and_hold
```

Inspect the RL environment shape:

```bash
python -m market_monitor.cli rl-spec
```

## Dashboard

Start the Streamlit dashboard:

```bash
streamlit run market_monitor/dashboard.py
```

The dashboard supports:

- Creating a curated sample A-share watchlist.
- Generating a broader A-share watchlist from AkShare spot data.
- Downloading or refreshing local A-share candles.
- Filtering by industry, signal type, confidence, and symbol/name search.
- Viewing K-line charts and backtest metrics for selected symbols.

## Strategies

Current strategy names:

- `ma_trend` - moving-average trend following, recommended default.
- `breakout` - price breakout with volume confirmation.
- `rsi_rebound` - RSI recovery after weak/oversold conditions.
- `macd_trend` - MACD momentum confirmation with trend and volume filters.
- `bollinger_reversion` - mean-reversion rebound around the lower Bollinger Band.
- `volume_pullback` - low-volume pullback inside a broader uptrend.

All strategies return explainable reasons plus risk hints such as stop loss, take profit, and suggested max position percentage.

## Data workflow

Generated market data should usually be treated as local research input rather than source code.

Recommended convention:

- `examples/data/` contains small tracked sample data for tests and demos.
- `data/` is for locally downloaded/generated market data and is ignored by Git by default.
- `watchlists/` can contain tracked watchlist definitions, but large generated universes should be reviewed before committing.

Download one A-share symbol:

```bash
python -m market_monitor.cli download-a-share \
  --symbol 000001.SZ \
  --start-date 20240101 \
  --end-date 20260509 \
  --output data/a_share/000001.SZ.csv
```

Download a watchlist:

```bash
python -m market_monitor.cli download-a-share \
  --watchlist watchlists/a_share.csv \
  --start-date 20240101 \
  --end-date 20260509 \
  --output-dir data/a_share
```

## Development notes

Before committing:

```bash
python -m pytest tests
python -m market_monitor.cli strategies
python -m market_monitor.cli compare-strategies \
  --csv examples/data/a_share/000001.SZ.csv \
  --symbol 000001.SZ \
  --format table
```

This project is not financial advice. Strategy outputs are research signals and should be validated with backtesting, risk controls, and out-of-sample review before any real trading use.
