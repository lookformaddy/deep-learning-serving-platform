"""Training jobs with checkpointing and resume — pure stdlib.

A :class:`TrainingJob` trains an :class:`~dlserve.nn.MLP` (or the linear
baseline) on a seeded synthetic dataset, logs per-epoch metrics, and
writes JSON checkpoints. A crashed or stopped job resumes from the latest
checkpoint by re-loading weights and the epoch counter.

Checkpoints are plain JSON so they are inspectable and diffable; weights
are small (a few thousand parameters), so JSON is a reasonable format here
— for larger models this would move to a binary format.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .data import Dataset, standardize_apply, standardize_fit
from .evaluation import accuracy
from .nn import (
    MLP,
    MLPConfig,
    cross_entropy_loss,
    make_optimizer,
    softmax,
    softmax_ce_grad,
)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 0.02
    optimizer: str = "adam"
    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    seed: int = 42
    checkpoint_every: int = 5  # epochs between checkpoints (0 = final only)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_accuracy: float


@dataclass
class TrainReport:
    job_id: str
    epochs_run: int
    history: list[EpochRecord]
    final_train_accuracy: float
    final_val_accuracy: float
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "epochs_run": self.epochs_run,
            "final_train_accuracy": self.final_train_accuracy,
            "final_val_accuracy": self.final_val_accuracy,
            "checkpoint_path": self.checkpoint_path,
            "history": [
                {
                    "epoch": r.epoch,
                    "train_loss": r.train_loss,
                    "train_accuracy": r.train_accuracy,
                    "val_accuracy": r.val_accuracy,
                }
                for r in self.history
            ],
        }


def _batches(
    X: list[list[float]], y: list[int], batch_size: int, rng: random.Random
) -> list[tuple[list[list[float]], list[int]]]:
    idx = list(range(len(y)))
    rng.shuffle(idx)
    return [
        ([X[i] for i in idx[s : s + batch_size]], [y[i] for i in idx[s : s + batch_size]])
        for s in range(0, len(idx), batch_size)
    ]


class LinearBaseline:
    """Softmax regression trained with the same loop as the MLP.

    Serves two roles: (1) the "does deep learning actually help?" control
    in the demo — it cannot learn the ring boundary; (2) the cheap
    latency-budget fallback model at serving time.
    """

    def __init__(self, input_dim: int, n_classes: int, seed: int = 42) -> None:
        rng = random.Random(seed)
        limit = math.sqrt(6.0 / (input_dim + n_classes))
        self.weight = [[rng.uniform(-limit, limit) for _ in range(n_classes)]
                       for _ in range(input_dim)]
        self.bias = [0.0] * n_classes
        self.input_dim = input_dim
        self.n_classes = n_classes

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        out = []
        for row in X:
            logits = [
                sum(row[i] * self.weight[i][j] for i in range(self.input_dim)) + self.bias[j]
                for j in range(self.n_classes)
            ]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            total = sum(exps)
            out.append([e / total for e in exps])
        return out

    def predict(self, X: list[list[float]]) -> list[int]:
        return [max(range(len(r)), key=lambda j: r[j]) for r in self.predict_proba(X)]

    def state_dict(self) -> dict[str, Any]:
        return {"weight": self.weight, "bias": self.bias, "kind": "linear-baseline"}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "LinearBaseline":
        model = cls(len(state["weight"]), len(state["bias"]))
        model.weight = state["weight"]
        model.bias = state["bias"]
        return model


def train_linear_baseline(
    train: Dataset, val: Dataset, epochs: int = 100, lr: float = 0.05, seed: int = 42
) -> tuple[LinearBaseline, float]:
    """Full-batch gradient descent for the linear baseline."""
    model = LinearBaseline(len(train.X[0]), train.n_classes, seed=seed)
    n = len(train.y)
    for _ in range(epochs):
        probs = model.predict_proba(train.X)
        grad = softmax_ce_grad(probs, train.y)
        # dW = X^T @ grad ; db = sum(grad)
        for i in range(model.input_dim):
            for j in range(model.n_classes):
                model.weight[i][j] -= lr * sum(train.X[k][i] * grad[k][j] for k in range(n))
        for j in range(model.n_classes):
            model.bias[j] -= lr * sum(grad[k][j] for k in range(n))
    val_acc = accuracy(model.predict(val.X), val.y)
    return model, val_acc


class TrainingJob:
    """One reproducible MLP training run with checkpointing."""

    def __init__(
        self,
        job_id: str,
        config: TrainConfig,
        workdir: str | Path,
    ) -> None:
        self.job_id = job_id
        self.config = config
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.workdir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    # -- checkpoints ------------------------------------------------------
    def checkpoint_path(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"{self.job_id}-epoch-{epoch:04d}.json"

    def save_checkpoint(self, model: MLP, epoch: int, history: list[EpochRecord]) -> Path:
        path = self.checkpoint_path(epoch)
        payload = {
            "job_id": self.job_id,
            "epoch": epoch,
            "config": {
                "input_dim": model.config.input_dim,
                "hidden_dims": model.config.hidden_dims,
                "output_dim": model.config.output_dim,
                "dropout_rate": model.config.dropout_rate,
                "use_layer_norm": model.config.use_layer_norm,
            },
            "weights": model.state_dict(),
            "history": [r.__dict__ for r in history],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def latest_checkpoint(self) -> Optional[Path]:
        files = sorted(self.checkpoint_dir.glob(f"{self.job_id}-epoch-*.json"))
        return files[-1] if files else None

    @staticmethod
    def load_checkpoint(path: str | Path) -> tuple[MLP, int, list[EpochRecord], TrainConfig]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = payload["config"]
        model = MLP(
            MLPConfig(
                input_dim=cfg["input_dim"],
                hidden_dims=cfg["hidden_dims"],
                output_dim=cfg["output_dim"],
                dropout_rate=cfg["dropout_rate"],
                use_layer_norm=cfg["use_layer_norm"],
            )
        )
        model.load_state_dict(payload["weights"])
        history = [EpochRecord(**r) for r in payload["history"]]
        config = TrainConfig(
            hidden_dims=cfg["hidden_dims"],
            dropout_rate=cfg["dropout_rate"],
            use_layer_norm=cfg["use_layer_norm"],
        )
        return model, int(payload["epoch"]), history, config

    # -- training ---------------------------------------------------------
    def run(
        self,
        train: Dataset,
        val: Dataset,
        resume_from: Optional[str | Path] = None,
    ) -> tuple[MLP, TrainReport]:
        rng = random.Random(self.config.seed)
        start_epoch = 0
        history: list[EpochRecord] = []
        model: Optional[MLP] = None
        if resume_from is not None:
            model, start_epoch, history, _ = self.load_checkpoint(resume_from)
        if model is None:
            model = MLP(
                MLPConfig(
                    input_dim=len(train.X[0]),
                    hidden_dims=self.config.hidden_dims,
                    output_dim=train.n_classes,
                    dropout_rate=self.config.dropout_rate,
                    use_layer_norm=self.config.use_layer_norm,
                ),
                rng=rng,
            )
        optimizer = make_optimizer(self.config.optimizer, self.config.lr)
        # If resuming, Adam moments restart (see nn.Adam.load_state_dict
        # note) — the step counter is the only state carried over.
        means, stds = standardize_fit(train.X)
        X_train = standardize_apply(train.X, means, stds)
        X_val = standardize_apply(val.X, means, stds)

        last_ckpt: Optional[Path] = None
        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            for X_b, y_b in _batches(X_train, train.y, self.config.batch_size, rng):
                logits, caches = model.forward(X_b, training=True)
                probs = softmax(logits)
                grad = softmax_ce_grad(probs, y_b)
                grads = model.backward(grad, caches)
                optimizer.step(model, grads)
            train_logits, _ = model.forward(X_train, training=False)
            train_probs = softmax(train_logits)
            rec = EpochRecord(
                epoch=epoch,
                train_loss=cross_entropy_loss(train_probs, train.y),
                train_accuracy=accuracy(model.predict(X_train), train.y),
                val_accuracy=accuracy(model.predict(X_val), val.y),
            )
            history.append(rec)
            if self.config.checkpoint_every and epoch % self.config.checkpoint_every == 0:
                last_ckpt = self.save_checkpoint(model, epoch, history)
        # Always persist a final checkpoint at the last epoch, even when
        # it coincides with a periodic one (same path, same content).
        last_ckpt = self.save_checkpoint(model, self.config.epochs, history)

        report = TrainReport(
            job_id=self.job_id,
            epochs_run=self.config.epochs,
            history=history,
            final_train_accuracy=history[-1].train_accuracy,
            final_val_accuracy=history[-1].val_accuracy,
            checkpoint_path=str(last_ckpt),
        )
        (self.workdir / f"{self.job_id}-report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        # Persist the standardisation stats alongside the checkpoint so
        # serving applies the identical transform.
        (self.workdir / f"{self.job_id}-scaler.json").write_text(
            json.dumps({"means": means, "stds": stds}), encoding="utf-8"
        )
        return model, report


def load_scaler(workdir: str | Path, job_id: str) -> tuple[list[float], list[float]]:
    payload = json.loads(
        (Path(workdir) / f"{job_id}-scaler.json").read_text(encoding="utf-8")
    )
    return payload["means"], payload["stds"]
