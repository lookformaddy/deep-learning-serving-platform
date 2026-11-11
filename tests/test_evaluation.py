"""Tests for offline evaluation metrics (evaluation.py)."""
import pytest

from dlserve.evaluation import (
    accuracy,
    confusion_matrix,
    evaluate,
    latency_stats,
    log_loss,
    macro_f1,
    per_class_metrics,
    percentile,
)


def test_accuracy():
    assert accuracy([0, 1, 2, 1], [0, 1, 2, 0]) == 0.75
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])
    with pytest.raises(ValueError):
        accuracy([], [])


def test_confusion_matrix():
    matrix = confusion_matrix([0, 1, 1, 2], [0, 1, 0, 2], 3)
    assert matrix == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_per_class_metrics_perfect():
    metrics = per_class_metrics([0, 1, 2], [0, 1, 2], 3)
    for m in metrics.values():
        assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


def test_macro_f1_partial():
    # class 0: P=1.0 R=0.5 F1=2/3; class 1: P=0.5 R=1.0 F1=2/3; class 2: F1=1.0
    assert macro_f1([0, 1, 2, 1], [0, 1, 2, 0], 3) == pytest.approx(0.7778, rel=1e-3)


def test_log_loss():
    import math

    assert log_loss([[0.9, 0.1], [0.2, 0.8]], [0, 1]) == pytest.approx(
        -(math.log(0.9) + math.log(0.8)) / 2
    )


def test_percentile():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 100) == 100.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_latency_stats():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats.count == 4
    assert stats.mean_ms == 25.0
    assert stats.p50_ms == 20.0
    assert stats.max_ms == 40.0
    with pytest.raises(ValueError):
        latency_stats([])


def test_evaluate_report():
    def predict_proba(X):
        return [[0.9, 0.05, 0.05] if sum(row) > 0 else [0.05, 0.9, 0.05] for row in X]

    report = evaluate(predict_proba, [[1.0], [-1.0], [2.0]], [0, 1, 0], 3)
    assert report.accuracy == 1.0
    # Class 2 has zero support, so its F1 is 0 by convention and macro F1
    # is (1 + 1 + 0) / 3.
    assert report.macro_f1 == pytest.approx(2 / 3)
    assert "accuracy=1.0000" in report.summary()
