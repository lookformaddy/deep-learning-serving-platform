"""Tests for training jobs, checkpoints and resume (training.py)."""
import json
from pathlib import Path

import pytest

from dlserve.data import generate_tabular, split_dataset, standardize_apply
from dlserve.evaluation import accuracy
from dlserve.training import (
    LinearBaseline,
    TrainConfig,
    TrainingJob,
    load_scaler,
    train_linear_baseline,
)


def test_training_job_runs_and_reports(tabular_split, tmp_path):
    train, val = tabular_split
    job = TrainingJob(
        "job-a", TrainConfig(epochs=5, hidden_dims=[16, 8], seed=42), workdir=tmp_path
    )
    model, report = job.run(train, val)
    assert report.epochs_run == 5
    assert len(report.history) == 5
    assert 0.0 < report.final_train_accuracy <= 1.0
    assert 0.0 < report.final_val_accuracy <= 1.0
    assert report.checkpoint_path and Path(report.checkpoint_path).exists()
    assert model.predict(val.X[:4]) is not None


def test_checkpoint_roundtrip_preserves_predictions(trained_mlp, tabular_split):
    model, _report, job = trained_mlp
    _train, val = tabular_split
    ckpt = job.latest_checkpoint()
    assert ckpt is not None
    loaded, epoch, history, _cfg = TrainingJob.load_checkpoint(ckpt)
    assert epoch == 8
    assert len(history) == 8
    assert loaded.predict(val.X) == model.predict(val.X)


def test_resume_continues_from_checkpoint(tabular_split, tmp_path):
    train, val = tabular_split
    job = TrainingJob(
        "job-resume",
        TrainConfig(epochs=4, hidden_dims=[8], seed=42, checkpoint_every=2),
        workdir=tmp_path,
    )
    _model, report = job.run(train, val)
    assert len(report.history) == 4
    ckpt = job.latest_checkpoint()
    # Resume to 6 epochs total: history must continue, not restart.
    job2 = TrainingJob(
        "job-resume",
        TrainConfig(epochs=6, hidden_dims=[8], seed=42, checkpoint_every=2),
        workdir=tmp_path,
    )
    _model2, report2 = job2.run(train, val, resume_from=ckpt)
    assert report2.epochs_run == 6
    assert [r.epoch for r in report2.history] == [1, 2, 3, 4, 5, 6]


def test_scaler_saved_and_loaded(tabular_split, tmp_path):
    train, val = tabular_split
    job = TrainingJob(
        "job-scaler", TrainConfig(epochs=2, hidden_dims=[8], seed=42), workdir=tmp_path
    )
    job.run(train, val)
    means, stds = load_scaler(tmp_path, "job-scaler")
    assert len(means) == len(train.X[0]) == len(stds)
    scaled = standardize_apply(train.X[:5], means, stds)
    assert all(abs(v) < 10 for row in scaled for v in row)


def test_mlp_beats_linear_baseline_on_nonlinear_task(tmp_path):
    """The ring x quadrant task is linearly inseparable: the MLP must win
    by a clear margin. This is the demo's central claim."""
    dataset = generate_tabular(n=600, seed=42)
    train, val = split_dataset(dataset, test_frac=0.2, seed=42)
    job = TrainingJob(
        "job-compare",
        TrainConfig(epochs=40, lr=0.02, hidden_dims=[32, 16], seed=42),
        workdir=tmp_path,
    )
    model, _report = job.run(train, val)
    means, stds = load_scaler(tmp_path, "job-compare")
    mlp_acc = accuracy(
        model.predict(standardize_apply(val.X, means, stds)), val.y
    )
    _baseline, baseline_acc = train_linear_baseline(train, val, epochs=100, seed=42)
    assert mlp_acc > baseline_acc + 0.15, (
        f"MLP {mlp_acc:.3f} should beat linear baseline {baseline_acc:.3f} "
        "by > 0.15 on the nonlinear task"
    )
    assert mlp_acc > 0.75, f"MLP accuracy {mlp_acc:.3f} unexpectedly low"


def test_linear_baseline_state_roundtrip():
    model = LinearBaseline(4, 3, seed=1)
    restored = LinearBaseline.from_state_dict(model.state_dict())
    assert restored.predict([[0.1, 0.2, 0.3, 0.4]]) == model.predict(
        [[0.1, 0.2, 0.3, 0.4]]
    )


def test_report_json_written(tabular_split, tmp_path):
    train, val = tabular_split
    job = TrainingJob(
        "job-json", TrainConfig(epochs=2, hidden_dims=[8], seed=42), workdir=tmp_path
    )
    job.run(train, val)
    payload = json.loads((tmp_path / "job-json-report.json").read_text())
    assert payload["job_id"] == "job-json"
    assert payload["epochs_run"] == 2
