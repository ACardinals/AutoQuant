from __future__ import annotations

from market_monitor.ml.features import FEATURE_COLUMNS, feature_matrix, feature_row
from market_monitor.models import Candle


def build_supervised_dataset(
    candles: list[Candle],
    horizon: int = 10,
    threshold: float = 0.0,
) -> dict:
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    rows = []
    for index in range(35, len(candles) - horizon):
        features = feature_row(candles[: index + 1])
        if features is None:
            continue
        current_close = candles[index].close
        future_close = candles[index + horizon].close
        future_return = future_close / current_close - 1 if current_close else 0.0
        rows.append(
            {
                **features,
                f"future_return_{horizon}": future_return,
                "label": 1 if future_return > threshold else 0,
            }
        )

    return {
        "rows": rows,
        "feature_columns": FEATURE_COLUMNS,
        "X": feature_matrix(rows),
        "y": [row["label"] for row in rows],
        "future_returns": [row[f"future_return_{horizon}"] for row in rows],
    }
