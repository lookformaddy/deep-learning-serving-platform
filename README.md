# deep-learning-serving-platform

A deep-learning serving platform built **from scratch in pure Python
stdlib** — no torch, no numpy, no sklearn. A small MLP (manual
forward/backprop, Adam/SGD, dropout, layer norm) is trained on
deterministic synthetic data, registered behind a gated champion
promotion, and served with canary/shadow deployment, dynamic batching,
a latency budget with fallback, and PSI drift monitoring.

> **Clean-room note:** all code is original and all data is synthetic.
> Nothing here comes from any employer's codebase, data, or internal
> design.

## The enterprise problem

Serving a neural network in production is not a training problem — it
is a *lifecycle* problem. Teams need to answer, for every model version:

- Is it actually better than the current champion, on held-out data,
  by enough to matter — and was a human in the loop?
- Can it be rolled out gradually (canary), observed without affecting
  users (shadow), and rolled back?
- What happens when it gets slow — does the request fail, or fall back
  to something cheaper?
- Has the live data drifted away from what the model was trained on?

This project implements that lifecycle end to end, with the model
itself written from scratch so every step is auditable.

## What it demonstrates

- **From-scratch deep learning** — MLP with LayerNorm, dropout,
  Adam/SGD, and a finite-difference gradient check in the test suite.
- **Training jobs with checkpointing/resume** — JSON checkpoints, epoch
  history, and resume-from-checkpoint after an interruption.
- **Versioned model registry** — immutable versions, stage machine
  (staging → challenger → production → archived), and a fail-closed
  promotion gate (metric delta + latency budget + human approval).
- **Safe rollout** — deterministic hash canary routing, shadow scoring
  with agreement rates, and automatic rollback.
- **Latency-aware serving** — dynamic batching and a latency budget
  that falls back to a cheap linear model instead of failing slow.
- **Drift monitoring** — PSI on input features *and* on hidden-layer
  activations, which catches representation shift that input PSI can
  miss.
- **Offline evaluation** — accuracy, per-class precision/recall/F1,
  log loss, confusion matrix, latency percentiles.

## Demo results (seed 42)

```
MLP      val accuracy=0.7944 macro_f1=0.7949 log_loss=0.4313
baseline val accuracy=0.5111 (linear model cannot learn the ring x quadrant boundary)
v1 champion (val_accuracy 0.70) -> v2 promotion: APPROVED: all gates passed
120 requests: champion=95 challenger=25 fallback=0 (fallback_rate=0.00%)
shadow: 53 scored, agreement=100.00%
latency: mean=0.20ms p50=0.14ms p95=0.27ms p99=0.30ms
stable window: status=OK worst PSI=0.0818
shifted window: status=CRITICAL worst PSI=0.2680 (feature f0)
```

The MLP beats the linear baseline by ~28 points because the synthetic
tabular task is genuinely nonlinear: the label is
`(radius_band + half_plane) % 3`, an XOR-style structure no linear
model can separate.

## How to run

```bash
pip install -r requirements.txt   # pytest, fastapi, uvicorn, httpx
make test                          # full suite (API tests need fastapi)
make demo                          # end-to-end pipeline, ~20s
make serve                         # FastAPI on http://127.0.0.1:8000
```

The core library has **no runtime dependencies** — FastAPI is only
needed for the HTTP service, and the API tests auto-skip without it.

### CLI

```bash
PYTHONPATH=src python3 -m dlserve.cli train --dataset tabular --epochs 40
PYTHONPATH=src python3 -m dlserve.cli eval .dlserve/checkpoints/job-1-epoch-0040.json --job-id job-1
PYTHONPATH=src python3 -m dlserve.cli drift .dlserve/checkpoints/job-1-epoch-0040.json --job-id job-1
PYTHONPATH=src python3 -m dlserve.cli promote v2 --registry .dlserve/registry.json --approve
```

### API

| Endpoint | Description |
|---|---|
| `GET /health` | Status, uptime, champion/fallback versions |
| `POST /predict` | Single prediction (16 features, tabular task) |
| `POST /predict/batch` | Batched predictions |
| `GET /metrics` | Request counts, fallback rate, shadow agreement, p95 latency |

## Architecture

See [docs/architecture.md](docs/architecture.md) for the component
map, data flow, and the design decisions and trade-offs.

## Project layout

```
src/dlserve/
  nn.py          # from-scratch MLP: layers, optimizers, autograd-free backprop
  data.py        # deterministic synthetic datasets (tabular rings, image-like shapes)
  training.py    # training jobs, checkpoints/resume, linear baseline
  registry.py    # versioned registry + gated champion promotion
  serving.py     # dynamic batching, canary/shadow, latency budget + fallback
  monitoring.py  # PSI drift on inputs and hidden activations
  evaluation.py  # offline metrics (accuracy, F1, log loss, latency percentiles)
  service.py     # FastAPI adapter (optional dependency)
  cli.py         # train / eval / serve / promote / drift
scripts/demo.py  # end-to-end demo
tests/           # pytest suite (API suite auto-skips without FastAPI)
docs/
  architecture.md
  github-actions-ci.yml   # move to .github/workflows/ci.yml to activate
```

## Design decisions and trade-offs

- **Pure stdlib, no torch/numpy.** The point of this project is the
  serving lifecycle, not framework fluency — and a from-scratch MLP
  makes the maths auditable and the dependency footprint zero. The
  cost is speed: training is CPU-bound pure Python, so models stay
  small (a few thousand parameters). For real workloads this layer
  would be swapped for a framework-backed trainer behind the same
  registry/serving interfaces.
- **JSON checkpoints.** Weights are small here, so human-readable JSON
  beats binary formats for inspectability. Adam moments are *not*
  restored on resume (only the step count) — documented in
  `nn.Adam.load_state_dict` — because moments are keyed by layer
  object identity, which is not stable across processes. A resumed
  run re-warms its moments; the epoch counter and weights are exact.
- **PSI with quantile bins on the reference.** Standard industry
  practice; the demo uses 5 bins on 300-row windows because PSI is
  noisy on small samples (documented in the tests). The 10-bin
  production default assumes reference windows in the thousands.
- **Latency budget measured per request, fallback is a linear model.**
  The fallback is deliberately weak but instant; in production it
  would be a cached/quantised model or a stale-but-valid prediction.
- **Shadow agreement uses argmax equality.** Cheap and interpretable;
  a fuller comparison would track per-class agreement and
  probability calibration drift.
- **CI ships as `docs/github-actions-ci.yml`.** The GitHub App used to
  publish this repo cannot write `.github/workflows/` directly; move
  the file there to activate CI.

## Tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
# system python (no FastAPI): 67 passed, API suite skipped
# venv (FastAPI + httpx):     74 passed
```

Coverage includes a finite-difference gradient check of the manual
backprop, checkpoint/resume round-trips, promotion-gate rejection
paths, deterministic canary routing, fallback on latency breach, and
PSI drift detection on stable vs shifted windows.
