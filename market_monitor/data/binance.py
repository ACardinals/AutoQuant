from __future__ import annotations

import requests

from market_monitor.models import Candle


class BinanceClient:
    base_url = "https://api.binance.com"

    def fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 300) -> list[Candle]:
        response = requests.get(
            f"{self.base_url}/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            timeout=15,
        )
        response.raise_for_status()
        return [Candle.from_binance_kline(symbol.upper(), row) for row in response.json()]
