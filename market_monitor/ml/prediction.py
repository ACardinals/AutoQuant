from __future__ import annotations

from pathlib import Path

from market_monitor.data.watchlist import WatchlistItem
from market_monitor.ml.dataset import build_supervised_dataset
from market_monitor.ml.features import feature_matrix, feature_row
from market_monitor.ml.models import create_model
from market_monitor.models import Candle


def predict_latest_probability(
    candles: list[Candle],
    model_name: str = "hist_gradient_boosting",
    horizon: int = 10,
    threshold: float = 0.0,
) -> dict:
    if len(candles) < 40 + horizon:
        raise ValueError("not enough candles for ML ranking")

    training_candles = candles[:-1]
    dataset = build_supervised_dataset(training_candles, horizon, threshold)
    X = dataset["X"]
    y = dataset["y"]
    if len(X) < 10:
        raise ValueError("not enough labeled samples for ML ranking")
    if len(set(y)) < 2:
        raise ValueError("training labels only contain one class")

    latest_features = feature_row(candles)
    if latest_features is None:
        raise ValueError("latest feature row is unavailable")

    model = create_model(model_name)
    model.fit(X, y)
    probability = _positive_probability(model, feature_matrix([latest_features])[0])
    latest_timestamp = latest_features["timestamp"]
    return {
        "probability": round(probability, 4),
        "sample_count": len(X),
        "latest_date": latest_timestamp.date().isoformat(),
        "rsi_14": round(latest_features["rsi_14"], 2),
        "macd_histogram": round(latest_features["macd_histogram"], 4),
        "volume_ratio_20": round(latest_features["volume_ratio_20"], 4),
        "return_5": round(latest_features["return_5"] * 100, 2),
    }


def rank_watchlist_ml(
    items: list[WatchlistItem],
    candles_by_symbol: dict[str, list[Candle]],
    model_name: str = "hist_gradient_boosting",
    horizon: int = 10,
    threshold: float = 0.0,
    top_n: int | None = 20,
) -> list[dict]:
    metadata = {item.symbol: item for item in items}
    rows = []
    for symbol, candles in candles_by_symbol.items():
        item = metadata.get(symbol, WatchlistItem(symbol=symbol, csv_path=Path()))
        try:
            prediction = predict_latest_probability(candles, model_name, horizon, threshold)
        except ValueError:
            continue
        rows.append(
            {
                "symbol": symbol,
                "name": item.name,
                "market": item.market,
                "model": model_name,
                **prediction,
            }
        )
    rows = sorted(rows, key=lambda row: (row["probability"], row["sample_count"]), reverse=True)
    return rows[:top_n] if top_n is not None and top_n > 0 else rows


def format_ml_rank_table(rows: list[dict]) -> str:
    headers = [
        "symbol",
        "name",
        "market",
        "probability",
        "sample_count",
        "latest_date",
        "rsi_14",
        "macd_histogram",
        "volume_ratio_20",
        "return_5",
    ]
    table_rows = [_format_row(row, headers) for row in rows]
    widths = {header: max(len(header), *(len(row[header]) for row in table_rows)) for header in headers}
    lines = [" | ".join(header.ljust(widths[header]) for header in headers)]
    lines.append("-+-".join("-" * widths[header] for header in headers))
    lines.extend(" | ".join(row[header].ljust(widths[header]) for header in headers) for row in table_rows)
    return "\n".join(lines)


def _positive_probability(model, features: list[float]) -> float:
    if not hasattr(model, "predict_proba"):
        return float(model.predict([features])[0])
    probabilities = model.predict_proba([features])[0]
    if len(probabilities) < 2:
        return float(probabilities[0])
    return float(probabilities[1])


def _format_row(row: dict, headers: list[str]) -> dict[str, str]:
    return {header: _format_value(row.get(header)) for header in headers}


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
