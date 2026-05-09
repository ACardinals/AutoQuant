# Market Monitor

A Python research project for crypto/A-share market monitoring, strategy screening, backtesting, and future reinforcement-learning experiments.

## First MVP

- Fetch Binance public OHLCV candles
- Run rule-based strategy screening
- Backtest a simple long-only strategy
- Print structured buy/sell/hold suggestions

## Quick start

```bash
cd market-monitor
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
market-monitor backtest --symbol BTCUSDT --interval 1h --limit 300 --strategy breakout
```
