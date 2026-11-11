"""FastAPI service — thin HTTP adapter over :mod:`dlserve.serving`.

Optional dependency: the core library has no FastAPI requirement. The
app is built lazily by :func:`create_app` so importing this module
without FastAPI installed still works (and the API tests skip).
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from .data import generate_tabular, split_dataset, standardize_apply, standardize_fit
from .nn import MLP
from .serving import ServingConfig, ServingLayer
from .training import LinearBaseline, TrainConfig, TrainingJob, train_linear_baseline

try:  # pragma: no cover - import guard
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment,misc]
    Field = lambda *a, **k: None  # noqa: E731
    _FASTAPI_AVAILABLE = False


class PredictRequest(BaseModel):  # type: ignore[valid-type]
    request_id: str = Field(default="req-1")
    features: list[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):  # type: ignore[valid-type]
    request_id: str
    predicted_class: int
    probabilities: list[float]
    model_version: str
    served_by: str
    latency_ms: float
    shadow_agreement: bool | None = None


def _train_default_models(seed: int = 42) -> tuple[MLP, LinearBaseline, tuple[list[float], list[float]]]:
    """Train a small champion MLP + linear fallback on synthetic data.

    Kept deliberately small (fast startup) — production would load a
    checkpoint from the registry instead of training at boot.
    """
    import tempfile

    dataset = generate_tabular(n=600, seed=seed)
    train, val = split_dataset(dataset, test_frac=0.2, seed=seed)
    job = TrainingJob(
        "service-boot",
        TrainConfig(epochs=15, hidden_dims=[16, 8], seed=seed),
        workdir=os.environ.get("DLSERVE_WORKDIR", tempfile.mkdtemp(prefix="dlserve-boot-")),
    )
    model, _report = job.run(train, val)
    baseline, _acc = train_linear_baseline(train, val, epochs=50, seed=seed)
    means, stds = standardize_fit(train.X)
    return model, baseline, (means, stds)


def create_app(seed: int = 42) -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed; install -r requirements.txt")
    model, baseline, (means, stds) = _train_default_models(seed)

    def champion_fn(batch):
        return model.predict_proba(standardize_apply(batch, means, stds))

    def fallback_fn(batch):
        return baseline.predict_proba(standardize_apply(batch, means, stds))

    layer = ServingLayer(
        champion_fn=champion_fn,
        champion_version="mlp-v1",
        config=ServingConfig(
            latency_budget_ms=float(os.environ.get("DLSERVE_LATENCY_BUDGET_MS", "50")),
            canary_pct=float(os.environ.get("DLSERVE_CANARY_PCT", "0")),
            shadow_pct=0.0,
        ),
        fallback_fn=fallback_fn,
    )
    started_at = time.monotonic()
    app = FastAPI(title="deep-learning-serving-platform", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.monotonic() - started_at, 2),
            "champion": "mlp-v1",
            "fallback": "linear-baseline",
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if len(req.features) != 16:
            raise HTTPException(status_code=422, detail="expected 16 features (tabular-rings)")
        result = layer.predict_one(req.request_id, req.features)
        return PredictResponse(
            request_id=req.request_id,
            predicted_class=max(range(len(result.probabilities)), key=lambda j: result.probabilities[j]),
            probabilities=result.probabilities,
            model_version=result.model_version,
            served_by=result.served_by,
            latency_ms=round(result.latency_ms, 3),
            shadow_agreement=result.shadow_agreement,
        )

    @app.post("/predict/batch", response_model=list[PredictResponse])
    def predict_batch(reqs: list[PredictRequest]) -> list[PredictResponse]:
        if not reqs:
            raise HTTPException(status_code=422, detail="empty batch")
        for req in reqs:
            if len(req.features) != 16:
                raise HTTPException(status_code=422, detail="expected 16 features (tabular-rings)")
        results = layer.predict_batch([(r.request_id, r.features) for r in reqs])
        return [
            PredictResponse(
                request_id=r.request_id,
                predicted_class=max(range(len(res.probabilities)), key=lambda j: res.probabilities[j]),
                probabilities=res.probabilities,
                model_version=res.model_version,
                served_by=res.served_by,
                latency_ms=round(res.latency_ms, 3),
                shadow_agreement=res.shadow_agreement,
            )
            for r, res in zip(reqs, results)
        ]

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        m = layer.metrics
        return {
            "total": m.total,
            "champion": m.champion,
            "challenger": m.challenger,
            "fallback": m.fallback,
            "fallback_rate": round(m.fallback_rate, 4),
            "shadow_scored": m.shadow_scored,
            "shadow_agreement_rate": round(m.shadow_agreement_rate, 4),
            "p95_latency_ms": round(m.p95_latency_ms(), 3),
        }

    return app


app: Optional["FastAPI"] = create_app() if _FASTAPI_AVAILABLE else None
