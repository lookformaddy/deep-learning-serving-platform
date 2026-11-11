# Architecture — deep-learning-serving-platform

## Component map

```
┌─────────────────────────────────────────────────────────────────┐
│  Data (synthetic, seeded)                                       │
│  data.py: generate_tabular (ring × quadrant)                     │
│           generate_image_like (8×8 shape classes)               │
└───────────────┬─────────────────────────────────────────────────┘
                │  stratified split + standardisation
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Training (training.py)                                         │
│  TrainingJob: mini-batch Adam/SGD over nn.MLP                   │
│    ├─ JSON checkpoints every N epochs + final (resume-safe)     │
│    ├─ scaler persisted next to the checkpoint                   │
│    └─ LinearBaseline: the "does deep learning help?" control    │
└───────────────┬─────────────────────────────────────────────────┘
                │  checkpoint path + held-out metrics
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Registry (registry.py)                                         │
│  ModelRegistry: immutable versions, stage machine, JSON store   │
│  PromotionGate: metric delta + p95 latency budget + approval    │
│    staging → challenger → production (champion) → archived      │
│    rollback(): explicit, audited, no gate re-run                │
└───────────────┬─────────────────────────────────────────────────┘
                │  champion_fn / challenger_fn / fallback_fn
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Serving (serving.py)                                           │
│  ServingLayer.predict_one / predict_batch                       │
│    ├─ DynamicBatcher: flush on max size or max wait             │
│    ├─ canary: SHA-256 hash bucket of request id (deterministic) │
│    ├─ shadow: challenger scores champion traffic, agreement %   │
│    └─ latency budget: slow answer → linear-baseline fallback    │
└───────────────┬─────────────────────────────────────────────────┘
                │  requests + predictions
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Monitoring (monitoring.py) + Evaluation (evaluation.py)        │
│  DriftMonitor: PSI on input features + hidden activations       │
│  evaluate(): accuracy, per-class P/R/F1, log loss, latency p50/ │
│  p95/p99                                                        │
└─────────────────────────────────────────────────────────────────┘
                ▲
                │  thin adapter, optional dependency
┌─────────────────────────────────────────────────────────────────┐
│  service.py (FastAPI) / cli.py                                  │
│  /health /predict /predict/batch /metrics · train/eval/serve/   │
│  promote/drift subcommands                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Data flow

1. **Train.** `TrainingJob.run()` standardises the training split
   (stats saved to `<job>-scaler.json`), trains the MLP in mini-batches,
   logs per-epoch loss/accuracy, and writes JSON checkpoints.
2. **Register.** The checkpoint is registered as an immutable version
   with its held-out metrics. A challenger is promoted to champion only
   through `PromotionGate`: it must beat the champion by `min_delta` on
   the chosen metric, stay inside the p95 latency budget, and carry a
   human approval flag. The superseded champion is archived, never
   deleted.
3. **Serve.** `ServingLayer` routes each request by a deterministic hash
   of its id: `canary_pct` go to the challenger, the rest to the
   champion. A sample of champion traffic is also scored by the
   challenger (shadow) and the argmax agreement rate is recorded —
   the response is never affected. If a prediction exceeds the latency
   budget, the cheap linear fallback answers instead and the fallback
   is counted.
4. **Monitor.** `DriftMonitor` holds a reference snapshot of training
   inputs and first-hidden-layer activations. Live windows are scored
   with PSI per feature (and on activations); status is OK < 0.1,
   WARN < 0.25, CRITICAL ≥ 0.25.

## The model (nn.py)

- **Architecture:** `Linear → LayerNorm → ReLU → Dropout` blocks,
  then a final `Linear` producing logits; softmax + cross-entropy.
- **Backprop is manual.** Each layer implements `forward` (returning a
  cache) and `backward` (returning input grads + parameter grads).
  The test suite verifies this against finite differences.
- **Optimizers:** SGD with momentum and Adam (bias-corrected moments).
- **Weights:** Xavier uniform init; all randomness flows through a
  caller-supplied `random.Random`, so runs reproduce from a seed.

## The datasets (data.py)

- **tabular-rings** (16 features, 3 classes): label =
  `(radius_band + half_plane) % 3` — an XOR-style structure. A linear
  model scores ~0.47–0.51; the MLP reaches ~0.79. The remaining 14
  features are correlated noise/distractors, including a radius proxy
  that alone cannot separate the classes.
- **image-like-shapes** (64 features, 3 classes): flattened 8×8
  renders of a ring, a square frame, and a diagonal cross, with pixel
  noise — the same MLP on higher-dimensional vector input.
- Splits are stratified; standardisation stats are fit on train only.

## Key design decisions

| Decision | Rationale | Cost / follow-up |
|---|---|---|
| Pure stdlib, no torch/numpy | Auditable maths, zero runtime deps; the project is about the serving lifecycle | Training is slow pure Python; models stay small. A framework trainer would sit behind the same registry/serving interfaces. |
| JSON checkpoints | Inspectable, diffable at this scale | Not viable for large models — move to a binary format + object store. |
| Adam moments not restored on resume | Moments are keyed by layer identity, unstable across processes | Resumed runs re-warm moments; weights + epoch counter are exact. |
| Deterministic hash canary routing | Same request id always routes the same way; no sticky state | A bad hash distribution could skew the split; monitored via serving metrics. |
| Latency budget → linear fallback | Never fail slow; degrade gracefully | Fallback is weak by design — in production use a cached/quantised model. |
| PSI on activations as well as inputs | Catches representation shift when marginals look fine | Activation PSI needs the model at monitor time; currently computed inline. |
| CI as `docs/github-actions-ci.yml` | The publishing GitHub App cannot write `.github/workflows/` | Move the file to activate CI. |

## Failure modes handled

- **Promotion without approval / without beating the champion / over
  the latency budget** — gate rejects, champion untouched.
- **Duplicate version registration** — rejected; versions are immutable.
- **Rollback to an archived version** — rejected; archived is terminal.
- **Slow champion** — latency budget triggers the fallback; the
  request still gets an answer and the fallback rate is visible in
  `/metrics`.
- **Drifted inputs** — PSI flags WARN/CRITICAL before accuracy is
  measured; the demo shows a shifted window hitting CRITICAL (PSI 0.27)
  while the stable window stays OK (PSI 0.08).
- **Interrupted training** — resume from the latest checkpoint;
  history continues from the saved epoch, it does not restart.

## What is deliberately out of scope

- No GPU / framework training, no distributed training (see the
  sibling `distributed-training-platform` repo for that).
- No online learning — retraining is a new version through the same
  gate (see `continual-learning-platform`).
- No auth, TLS, or multi-tenancy on the API — the service is a demo
  adapter; production would front it with a gateway.
