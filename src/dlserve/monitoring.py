"""Drift monitoring with Population Stability Index (PSI) — pure stdlib.

PSI compares a reference distribution (training-time inputs and hidden
activations) against a live window. Thresholds follow the usual industry
convention: PSI < 0.1 = stable, 0.1–0.25 = drift worth investigating,
>= 0.25 = significant shift, act.

Two signals are tracked:
* input drift — per-feature PSI on the (standardised) feature vector;
* activation drift — PSI on the mean hidden activation of the first
  hidden layer, which catches representation shift that input PSI can
  miss (e.g. a feature interaction changing while marginals stay put).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .nn import Matrix


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index between two samples.

    Quantile bins are cut on ``expected``; empty bins are smoothed with a
    0.0001 floor so the log term stays finite. Deterministic.
    """
    if len(expected) < bins or len(actual) < bins:
        raise ValueError("need at least as many samples as bins")
    ordered = sorted(expected)
    edges = [ordered[int(i * len(ordered) / bins)] for i in range(bins)]
    edges.append(ordered[-1] + 1e-9)  # include the max in the last bin
    # De-duplicate degenerate edges (constant feature).
    edges = sorted(set(edges))
    if len(edges) < 2:
        return 0.0
    total = 0.0
    for lo, hi in zip(edges, edges[1:]):
        e_count = sum(1 for v in expected if lo <= v < hi or (v == hi and hi == edges[-1]))
        a_count = sum(1 for v in actual if lo <= v < hi or (v == hi and hi == edges[-1]))
        e_pct = max(e_count / len(expected), 0.0001)
        a_pct = max(a_count / len(actual), 0.0001)
        total += (a_pct - e_pct) * math.log(a_pct / e_pct)
    return total


def drift_status(score: float) -> str:
    if score < 0.1:
        return "OK"
    if score < 0.25:
        return "WARN"
    return "CRITICAL"


@dataclass
class DriftReport:
    feature_psi: dict[str, float]
    activation_psi: float
    worst_feature: str
    worst_psi: float
    status: str

    def summary(self) -> str:
        lines = [f"drift status={self.status} worst={self.worst_feature} "
                 f"PSI={self.worst_psi:.4f} activation_PSI={self.activation_psi:.4f}"]
        for name, score in sorted(self.feature_psi.items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"  {name}: PSI={score:.4f} ({drift_status(score)})")
        return "\n".join(lines)


class DriftMonitor:
    """Holds a reference snapshot and scores live windows against it."""

    def __init__(
        self,
        reference_inputs: Matrix,
        feature_names: list[str],
        reference_activations: Sequence[float] | None = None,
        bins: int = 10,
    ) -> None:
        if len(reference_inputs[0]) != len(feature_names):
            raise ValueError("feature_names must match input width")
        self.reference_inputs = reference_inputs
        self.feature_names = feature_names
        self.reference_activations = list(reference_activations) if reference_activations else None
        self.bins = bins

    def check(
        self,
        live_inputs: Matrix,
        live_activations: Sequence[float] | None = None,
    ) -> DriftReport:
        feature_psi: dict[str, float] = {}
        for j, name in enumerate(self.feature_names):
            ref_col = [row[j] for row in self.reference_inputs]
            live_col = [row[j] for row in live_inputs]
            feature_psi[name] = psi(ref_col, live_col, bins=self.bins)
        activation_psi = 0.0
        if self.reference_activations is not None and live_activations is not None:
            activation_psi = psi(
                self.reference_activations, list(live_activations), bins=self.bins
            )
        worst_feature = max(feature_psi, key=lambda k: feature_psi[k])
        worst = max(feature_psi[worst_feature], activation_psi)
        status = drift_status(worst)
        return DriftReport(
            feature_psi=feature_psi,
            activation_psi=activation_psi,
            worst_feature=worst_feature,
            worst_psi=worst,
            status=status,
        )


def hidden_activation_means(model, X: Matrix) -> list[float]:
    """Mean activation of the first hidden layer per sample.

    ``model`` is an :class:`~dlserve.nn.MLP`.
    """
    activations = model.first_hidden_activations(X)
    return [sum(row) / len(row) for row in activations]
