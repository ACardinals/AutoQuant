from datetime import datetime, timedelta

from market_monitor.ml.dataset import build_supervised_dataset
from market_monitor.ml.features import FEATURE_COLUMNS, build_feature_rows, feature_matrix, feature_row
from market_monitor.ml.models import available_models, create_model
from market_monitor.ml.validation import evaluate_time_series_model, format_ml_evaluation_table
from market_monitor.models import Candle


def _candles(count=80):
    start = datetime(2024, 1, 1)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            symbol="TEST",
            open=100 + index * 0.2,
            high=101 + index * 0.2,
            low=99 + index * 0.2,
            close=100 + index * 0.2 + (index % 5) * 0.1,
            volume=1000 + index * 10,
        )
        for index in range(count)
    ]


def test_feature_row_contains_expected_columns():
    row = feature_row(_candles(40))

    assert row is not None
    assert row["symbol"] == "TEST"
    for column in FEATURE_COLUMNS:
        assert column in row


def test_build_feature_rows_and_matrix():
    rows = build_feature_rows(_candles(45))
    matrix = feature_matrix(rows)

    assert rows
    assert len(matrix) == len(rows)
    assert len(matrix[0]) == len(FEATURE_COLUMNS)


def test_build_supervised_dataset_creates_labels():
    dataset = build_supervised_dataset(_candles(70), horizon=5, threshold=0)

    assert dataset["rows"]
    assert len(dataset["X"]) == len(dataset["y"]) == len(dataset["future_returns"])
    assert set(dataset["y"]) <= {0, 1}


def test_model_factory_names():
    assert "hist_gradient_boosting" in available_models()
    assert create_model("logistic_regression") is not None


def test_evaluate_time_series_model_outputs_metrics():
    result = evaluate_time_series_model(_candles(90), "logistic_regression", horizon=5, splits=3)

    assert result["model"] == "logistic_regression"
    assert result["samples"] > 0
    assert len(result["folds"]) == 3
    assert "accuracy" in result["metrics"]


def test_format_ml_evaluation_table():
    result = evaluate_time_series_model(_candles(90), "logistic_regression", horizon=5, splits=3)
    table = format_ml_evaluation_table(result)

    assert "model=logistic_regression" in table
    assert "accuracy" in table
