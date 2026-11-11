"""End-to-end demo: train -> evaluate -> register -> promote -> serve -> monitor.

Runs the full platform on deterministic synthetic data:

1. Train an MLP (from scratch, pure stdlib) and a linear baseline.
2. Show the MLP beating the baseline on the nonlinear ring task.
3. Register v1 as champion, train v2, promote it through the gate.
4. Serve with canary + shadow, measure latency, and show the fallback.
5. Score PSI drift on a shifted live window.

Usage: PYTHONPATH=src python3 scripts/demo.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlserve.data import generate_tabular, split_dataset, standardize_apply  # noqa: E402
from dlserve.evaluation import evaluate, latency_stats  # noqa: E402
from dlserve.monitoring import DriftMonitor, hidden_activation_means  # noqa: E402
from dlserve.registry import ModelRegistry, ModelVersion, PromotionGate  # noqa: E402
from dlserve.serving import ServingConfig, ServingLayer  # noqa: E402
from dlserve.training import (  # noqa: E402
    TrainConfig,
    TrainingJob,
    load_scaler,
    train_linear_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".dlserve-demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    workdir = Path(args.workdir)
    rng = random.Random(args.seed)

    print("== 1. data ==")
    dataset = generate_tabular(n=900, seed=args.seed)
    train, val = split_dataset(dataset, test_frac=0.2, seed=args.seed)
    print(f"dataset {dataset.name}: {len(train)} train / {len(val)} val, "
          f"{dataset.n_classes} classes, {len(dataset.X[0])} features")

    print("\n== 2. train MLP (from scratch) vs linear baseline ==")
    job = TrainingJob(
        "demo-v2",
        TrainConfig(epochs=40, lr=0.02, hidden_dims=[32, 16], seed=args.seed),
        workdir=workdir,
    )
    model, report = job.run(train, val)
    means, stds = load_scaler(workdir, "demo-v2")

    def mlp_proba(X):
        return model.predict_proba(standardize_apply(X, means, stds))

    baseline, baseline_val_acc = train_linear_baseline(train, val, epochs=100, seed=args.seed)
    mlp_report = evaluate(mlp_proba, val.X, val.y, val.n_classes)
    print(f"MLP      val accuracy={mlp_report.accuracy:.4f} "
          f"macro_f1={mlp_report.macro_f1:.4f} log_loss={mlp_report.log_loss:.4f}")
    print(f"baseline val accuracy={baseline_val_acc:.4f} "
          f"(linear model cannot learn the ring boundary)")
    assert mlp_report.accuracy > baseline_val_acc + 0.15, (
        "MLP should beat the linear baseline by a wide margin on rings"
    )

    print("\n== 3. registry: v1 champion, v2 gated promotion ==")
    registry = ModelRegistry(workdir / "registry.json")
    v1 = ModelVersion(
        name="tabular-mlp", version="v1",
        metrics={"val_accuracy": 0.70, "p95_latency_ms": 12.0},
        artifact_path=str(workdir / "checkpoints" / "demo-v1.json"),
    )
    registry.register(v1)
    registry.promote_to_champion(
        "v1", gate=PromotionGate(), approved=True
    )
    v2 = ModelVersion(
        name="tabular-mlp", version="v2",
        metrics={"val_accuracy": mlp_report.accuracy, "p95_latency_ms": 18.0},
        artifact_path=str(report.checkpoint_path),
    )
    registry.register(v2)
    _model, decision = registry.promote_to_champion(
        "v2", gate=PromotionGate(min_delta=0.01), approved=True
    )
    print(f"v1 champion (val_accuracy 0.70) -> v2 promotion: {decision}")
    print(f"champion is now {registry.champion().version} "
          f"(val_accuracy {registry.champion().metrics['val_accuracy']:.4f})")

    print("\n== 4. serving: canary + shadow + latency budget ==")
    # The demo re-uses the v2 weights as the canary challenger so the
    # routing/shadow mechanics are exercised; a real rollout would load
    # a separately trained challenger checkpoint here.
    layer = ServingLayer(
        champion_fn=mlp_proba,
        champion_version="v2",
        config=ServingConfig(canary_pct=25.0, shadow_pct=50.0, latency_budget_ms=50.0),
        challenger_fn=lambda X: model.predict_proba(standardize_apply(X, means, stds)),
        challenger_version="v2-canary",
        fallback_fn=lambda X: baseline.predict_proba(standardize_apply(X, means, stds)),
    )
    latencies: list[float] = []
    for i, row in enumerate(val.X[:120]):
        start = time.monotonic()
        result = layer.predict_one(f"req-{i}", row)
        latencies.append((time.monotonic() - start) * 1000.0)
        assert result.served_by in ("champion", "challenger", "fallback")
    stats = latency_stats(latencies)
    print(f"120 requests: champion={layer.metrics.champion} "
          f"challenger={layer.metrics.challenger} fallback={layer.metrics.fallback} "
          f"(fallback_rate={layer.metrics.fallback_rate:.2%})")
    print(f"shadow: {layer.metrics.shadow_scored} scored, "
          f"agreement={layer.metrics.shadow_agreement_rate:.2%}")
    print(f"latency: mean={stats.mean_ms:.2f}ms p50={stats.p50_ms:.2f}ms "
          f"p95={stats.p95_ms:.2f}ms p99={stats.p99_ms:.2f}ms")

    print("\n== 5. drift: stable window vs shifted window ==")
    # 300-row windows with 5 bins: PSI is noisy on small samples, and
    # the 10-bin production default assumes reference windows in the
    # thousands (see monitoring.py).
    ref_inputs = train.X[:300]
    ref_activations = hidden_activation_means(model, standardize_apply(ref_inputs, means, stds))
    monitor = DriftMonitor(ref_inputs, dataset.feature_names, ref_activations, bins=5)
    stable_live = train.X[300:600]
    stable = monitor.check(stable_live, hidden_activation_means(
        model, standardize_apply(stable_live, means, stds)))
    print(f"stable window: status={stable.status} worst PSI={stable.worst_psi:.4f}")
    shifted = [list(row) for row in stable_live]
    for row in shifted:
        row[0] = row[0] * 1.8 + 0.6  # shift the ring-radius proxy feature
    drifted = monitor.check(shifted, hidden_activation_means(
        model, standardize_apply(shifted, means, stds)))
    print(f"shifted window: status={drifted.status} worst PSI={drifted.worst_psi:.4f} "
          f"(feature {drifted.worst_feature})")
    assert drifted.status in ("WARN", "CRITICAL"), "shifted window should trigger drift"

    print("\n== demo complete ==")
    print(f"MLP val accuracy {mlp_report.accuracy:.4f} vs baseline "
          f"{baseline_val_acc:.4f}; champion v2 live; drift detection works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
