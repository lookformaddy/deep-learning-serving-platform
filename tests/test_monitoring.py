"""Tests for drift monitoring (monitoring.py)."""
import random

import pytest

from dlserve.data import generate_tabular, split_dataset, standardize_apply
from dlserve.monitoring import DriftMonitor, drift_status, hidden_activation_means, psi


def test_psi_identical_distributions_near_zero():
    rng = random.Random(0)
    sample = [rng.gauss(0, 1) for _ in range(500)]
    assert psi(sample, list(sample)) < 1e-9


def test_psi_shifted_distribution_large():
    rng = random.Random(0)
    ref = [rng.gauss(0, 1) for _ in range(500)]
    shifted = [v + 1.5 for v in ref]
    assert psi(ref, shifted) > 0.25


def test_psi_needs_enough_samples():
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [1.0, 2.0])


def test_drift_status_thresholds():
    assert drift_status(0.05) == "OK"
    assert drift_status(0.15) == "WARN"
    assert drift_status(0.30) == "CRITICAL"


def test_monitor_stable_window_ok():
    # PSI is noisy on small samples, so the stable-window test uses
    # 300-row windows with 5 bins (the production default of 10 bins
    # assumes reference windows in the thousands).
    dataset = generate_tabular(n=900, seed=42)
    train, _val = split_dataset(dataset, test_frac=0.2, seed=42)
    monitor = DriftMonitor(train.X[:300], dataset.feature_names, bins=5)
    report = monitor.check(train.X[300:600])
    assert report.status == "OK"
    assert report.worst_psi < 0.1


def test_monitor_shifted_window_critical():
    dataset = generate_tabular(n=900, seed=42)
    train, _val = split_dataset(dataset, test_frac=0.2, seed=42)
    monitor = DriftMonitor(train.X[:300], dataset.feature_names, bins=5)
    shifted = [list(row) for row in train.X[300:600]]
    for row in shifted:
        row[0] = row[0] * 2.0 + 1.0
    report = monitor.check(shifted)
    assert report.status in ("WARN", "CRITICAL")
    assert report.worst_feature == "f0"
    assert report.worst_psi >= 0.1


def test_hidden_activation_means(trained_mlp, tabular_split):
    model, _report, _job = trained_mlp
    train, _val = tabular_split
    means = hidden_activation_means(model, train.X[:10])
    assert len(means) == 10
    assert all(m >= 0.0 for m in means)  # post-ReLU activations are non-negative


def test_monitor_with_activations(tmp_path):
    # Needs 300-row windows for stable PSI (see test_monitor_stable_window_ok),
    # so this test trains its own small model on the larger dataset.
    from dlserve.training import TrainConfig, TrainingJob

    dataset = generate_tabular(n=900, seed=42)
    train, _val = split_dataset(dataset, test_frac=0.2, seed=42)
    job = TrainingJob(
        "job-act", TrainConfig(epochs=8, hidden_dims=[16, 8], seed=42), workdir=tmp_path
    )
    model, _report = job.run(train, _val)
    ref_act = hidden_activation_means(model, train.X[:300])
    monitor = DriftMonitor(train.X[:300], train.feature_names, ref_act, bins=5)
    report = monitor.check(
        train.X[300:600],
        live_activations=hidden_activation_means(model, train.X[300:600]),
    )
    assert report.activation_psi >= 0.0
    assert report.status == "OK"
    assert report.activation_psi < 0.1
