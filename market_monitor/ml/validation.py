from __future__ import annotations

from statistics import mean

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from market_monitor.ml.dataset import build_supervised_dataset
from market_monitor.ml.models import create_model
from market_monitor.models import Candle


def evaluate_time_series_model(
    candles: list[Candle],
    model_name: str = "hist_gradient_boosting",
    horizon: int = 10,
    splits: int = 5,
    threshold: float = 0.0,
) -> dict:
    dataset = build_supervised_dataset(candles, horizon, threshold)
    X = dataset["X"]
    y = dataset["y"]
    future_returns = dataset["future_returns"]
    if len(X) < splits + 2:
        raise ValueError("not enough samples for requested time-series splits")

    actual_splits = min(splits, len(X) - 1)
    fold_results = []
    splitter = TimeSeriesSplit(n_splits=actual_splits)
    for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        model = create_model(model_name, random_state=fold)
        X_train = [X[index] for index in train_index]
        y_train = [y[index] for index in train_index]
        X_test = [X[index] for index in test_index]
        y_test = [y[index] for index in test_index]
        returns_test = [future_returns[index] for index in test_index]

        if len(set(y_train)) < 2:
            predictions = [y_train[0]] * len(y_test)
            probabilities = None
        else:
            model.fit(X_train, y_train)
            predictions = list(model.predict(X_test))
            probabilities = _positive_probabilities(model, X_test)

        positive_returns = [future_return for prediction, future_return in zip(predictions, returns_test) if prediction == 1]
        fold_results.append(
            {
                "fold": fold,
                "train_size": len(train_index),
                "test_size": len(test_index),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "roc_auc": _roc_auc(y_test, probabilities),
                "positive_predictions": sum(1 for prediction in predictions if prediction == 1),
                "average_future_return_predicted_positive": mean(positive_returns) if positive_returns else None,
            }
        )

    return {
        "model": model_name,
        "horizon": horizon,
        "threshold": threshold,
        "samples": len(X),
        "feature_columns": dataset["feature_columns"],
        "metrics": _aggregate_metrics(fold_results),
        "folds": [_round_fold(fold) for fold in fold_results],
    }


def format_ml_evaluation_table(result: dict) -> str:
    headers = [
        "fold",
        "train_size",
        "test_size",
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "positive_predictions",
        "average_future_return_predicted_positive",
    ]
    rows = [_format_row(fold, headers) for fold in result["folds"]]
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    lines = [f"model={result['model']} horizon={result['horizon']} samples={result['samples']}"]
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    lines.extend(" | ".join(row[header].ljust(widths[header]) for header in headers) for row in rows)
    return "\n".join(lines)


def _positive_probabilities(model, X_test: list[list[float]]) -> list[float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(X_test)
    if len(probabilities[0]) < 2:
        return None
    return [float(row[1]) for row in probabilities]


def _roc_auc(y_test: list[int], probabilities: list[float] | None) -> float | None:
    if probabilities is None or len(set(y_test)) < 2:
        return None
    return roc_auc_score(y_test, probabilities)


def _aggregate_metrics(folds: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "roc_auc", "average_future_return_predicted_positive"]
    return {key: _mean_optional([fold[key] for fold in folds]) for key in keys}


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(mean(present), 4) if present else None


def _round_fold(fold: dict) -> dict:
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in fold.items()
    }


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
