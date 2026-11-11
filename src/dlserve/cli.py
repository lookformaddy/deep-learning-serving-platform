"""Command-line interface for the deep-learning serving platform.

Subcommands:
    train    Train an MLP on a synthetic dataset (with checkpoints).
    eval     Offline evaluation of a checkpoint against held-out data.
    serve    Start the FastAPI service (needs fastapi + uvicorn).
    demo     Run the end-to-end demo pipeline (same as scripts/demo.py).
    promote  Evaluate a promotion gate decision for a registry version.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import generate_image_like, generate_tabular, split_dataset, standardize_apply
from .evaluation import evaluate
from .monitoring import DriftMonitor, hidden_activation_means
from .nn import MLP
from .registry import ModelRegistry, PromotionGate
from .training import TrainConfig, TrainingJob, load_scaler


def _dataset(name: str, n: int, seed: int):
    if name == "tabular":
        return generate_tabular(n=n, seed=seed)
    if name == "image-like":
        return generate_image_like(n=n, seed=seed)
    raise ValueError(f"unknown dataset {name!r} (expected 'tabular' or 'image-like')")


def cmd_train(args: argparse.Namespace) -> int:
    dataset = _dataset(args.dataset, args.n, args.seed)
    train, val = split_dataset(dataset, test_frac=0.2, seed=args.seed)
    job = TrainingJob(
        args.job_id,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            optimizer=args.optimizer,
            hidden_dims=[int(d) for d in args.hidden.split(",")],
            dropout_rate=args.dropout,
            seed=args.seed,
        ),
        workdir=args.workdir,
    )
    _model, report = job.run(train, val, resume_from=args.resume)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"checkpoint: {report.checkpoint_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    model, epoch, _history, _cfg = TrainingJob.load_checkpoint(args.checkpoint)
    dataset = _dataset(args.dataset, args.n, args.seed)
    _train, val = split_dataset(dataset, test_frac=0.2, seed=args.seed)
    means, stds = load_scaler(Path(args.checkpoint).parent.parent, args.job_id)
    report = evaluate(
        lambda X: model.predict_proba(standardize_apply(X, means, stds)),
        val.X, val.y, val.n_classes,
    )
    print(f"checkpoint epoch {epoch} ({args.checkpoint})")
    print(report.summary())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("uvicorn is not installed; install -r requirements.txt", file=sys.stderr)
        return 1
    from .service import app

    print(f"starting API on {args.host}:{args.port} (champion mlp-v1)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    registry = ModelRegistry(args.registry)
    candidate = registry.get(args.version)
    gate = PromotionGate(
        metric=args.metric,
        min_delta=args.min_delta,
        max_p95_latency_ms=args.max_latency_ms,
    )
    _model, decision = registry.promote_to_champion(
        args.version, gate=gate, approved=args.approve
    )
    print(decision)
    return 0 if decision.approved else 1


def cmd_drift(args: argparse.Namespace) -> int:
    dataset = _dataset(args.dataset, args.n, args.seed)
    train, _val = split_dataset(dataset, test_frac=0.2, seed=args.seed)
    model, _epoch, _history, _cfg = TrainingJob.load_checkpoint(args.checkpoint)
    means, stds = load_scaler(Path(args.checkpoint).parent.parent, args.job_id)
    # Simulate a shifted live window: scale one feature up by 1.5x.
    live = [list(row) for row in train.X[:200]]
    for row in live:
        row[0] = row[0] * 1.5 + 0.5
    monitor = DriftMonitor(
        train.X[:200], dataset.feature_names,
        reference_activations=hidden_activation_means(model, train.X[:200]),
    )
    report = monitor.check(
        live, live_activations=hidden_activation_means(model, live)
    )
    print(report.summary())
    return 0 if report.status == "OK" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlserve", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train an MLP with checkpoints")
    p.add_argument("--dataset", default="tabular", choices=["tabular", "image-like"])
    p.add_argument("--job-id", default="job-1")
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--hidden", default="32,16")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workdir", default=".dlserve")
    p.add_argument("--resume", default=None, help="checkpoint JSON to resume from")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="offline-evaluate a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--job-id", default="job-1")
    p.add_argument("--dataset", default="tabular", choices=["tabular", "image-like"])
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="start the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("promote", help="evaluate the champion promotion gate")
    p.add_argument("version")
    p.add_argument("--registry", default=".dlserve/registry.json")
    p.add_argument("--metric", default="val_accuracy")
    p.add_argument("--min-delta", type=float, default=0.01)
    p.add_argument("--max-latency-ms", type=float, default=50.0)
    p.add_argument("--approve", action="store_true", help="human approval flag")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("drift", help="PSI drift check on a shifted live window")
    p.add_argument("checkpoint")
    p.add_argument("--job-id", default="job-1")
    p.add_argument("--dataset", default="tabular", choices=["tabular", "image-like"])
    p.add_argument("--n", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_drift)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
