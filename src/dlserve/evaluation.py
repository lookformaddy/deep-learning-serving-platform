"""Offline evaluation metrics — pure stdlib.

Accuracy, per-class precision/recall/F1, confusion matrix, log loss and
latency percentiles. Nothing here depends on any ML framework.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def accuracy(preds: Sequence[int], targets: Sequence[int]) -> float:
    if len(preds) != len(targets) or not targets:
        raise ValueError("predictions and targets must be non-empty and equal-length")
    return sum(int(p == t) for p, t in zip(preds, targets)) / len(targets)


def confusion_matrix(
    preds: Sequence[int], targets: Sequence[int], n_classes: int
) -> list[list[int]]:
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for p, t in zip(preds, targets):
        matrix[t][p] += 1
    return matrix


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


def per_class_metrics(
    preds: Sequence[int], targets: Sequence[int], n_classes: int
) -> dict[int, ClassMetrics]:
    matrix = confusion_matrix(preds, targets, n_classes)
    out: dict[int, ClassMetrics] = {}
    for c in range(n_classes):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n_classes)) - tp
        fn = sum(matrix[c]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        out[c] = ClassMetrics(precision, recall, f1, support=sum(matrix[c]))
    return out


def log_loss(probs: list[list[float]], targets: Sequence[int]) -> float:
    total = 0.0
    for row, t in zip(probs, targets):
        total += -math.log(max(row[t], 1e-12))
    return total / len(targets)


def macro_f1(preds: Sequence[int], targets: Sequence[int], n_classes: int) -> float:
    metrics = per_class_metrics(preds, targets, n_classes)
    return sum(m.f1 for m in metrics.values()) / n_classes


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (pct in [0, 100])."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    return ordered[max(0, min(rank - 1, len(ordered) - 1))]


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    if not latencies_ms:
        raise ValueError("no latency samples")
    return LatencyStats(
        count=len(latencies_ms),
        mean_ms=sum(latencies_ms) / len(latencies_ms),
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        max_ms=max(latencies_ms),
    )


@dataclass
class EvalReport:
    accuracy: float
    macro_f1: float
    log_loss: float
    confusion: list[list[int]]
    per_class: dict[int, ClassMetrics]
    latency: LatencyStats | None = None

    def summary(self) -> str:
        lines = [
            f"accuracy={self.accuracy:.4f} macro_f1={self.macro_f1:.4f} "
            f"log_loss={self.log_loss:.4f}",
        ]
        for c, m in sorted(self.per_class.items()):
            lines.append(
                f"  class {c}: precision={m.precision:.3f} recall={m.recall:.3f} "
                f"f1={m.f1:.3f} support={m.support}"
            )
        if self.latency is not None:
            lines.append(
                f"  latency: mean={self.latency.mean_ms:.2f}ms "
                f"p50={self.latency.p50_ms:.2f}ms p95={self.latency.p95_ms:.2f}ms "
                f"p99={self.latency.p99_ms:.2f}ms"
            )
        return "\n".join(lines)


def evaluate(
    predict_proba_fn,
    X: list[list[float]],
    y: list[int],
    n_classes: int,
    latencies_ms: Sequence[float] | None = None,
) -> EvalReport:
    probs = predict_proba_fn(X)
    preds = [max(range(len(r)), key=lambda j: r[j]) for r in probs]
    return EvalReport(
        accuracy=accuracy(preds, y),
        macro_f1=macro_f1(preds, y, n_classes),
        log_loss=log_loss(probs, y),
        confusion=confusion_matrix(preds, y, n_classes),
        per_class=per_class_metrics(preds, y, n_classes),
        latency=latency_stats(latencies_ms) if latencies_ms else None,
    )
