"""Serving layer: dynamic batching, canary/shadow, latency budgets.

The serving path is deliberately framework-free: a ``PredictFn`` is any
callable from a batch of feature vectors to a batch of probability
vectors. The FastAPI service in ``service.py`` is a thin adapter over
this module.

Key behaviours:
* **Dynamic batching** — requests accumulate until ``max_batch_size``
  or ``max_wait_ms`` elapses, then flush as one model call. Batching
  amortises the (pure-Python) forward pass.
* **Latency budget with fallback** — if the champion's p95 estimate for
  a batch exceeds the budget, the request is answered by the cheap
  fallback model (a linear baseline) and counted, never dropped.
* **Canary** — a deterministic hash of the request id routes a
  configurable percentage to the challenger.
* **Shadow** — the challenger also scores a sample of champion traffic
  and the agreement rate is recorded, without affecting the response.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .nn import Matrix

PredictFn = Callable[[Matrix], list[list[float]]]


@dataclass
class ServingConfig:
    max_batch_size: int = 16
    max_wait_ms: float = 10.0
    latency_budget_ms: float = 50.0
    canary_pct: float = 10.0
    shadow_pct: float = 20.0
    fallback_enabled: bool = True


@dataclass
class PredictionResult:
    probabilities: list[float]
    model_version: str
    served_by: str  # "champion" | "challenger" | "fallback"
    latency_ms: float
    shadow_agreement: bool | None = None


@dataclass
class ServingMetrics:
    total: int = 0
    champion: int = 0
    challenger: int = 0
    fallback: int = 0
    shadow_scored: int = 0
    shadow_agreements: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def fallback_rate(self) -> float:
        return self.fallback / self.total if self.total else 0.0

    @property
    def shadow_agreement_rate(self) -> float:
        return self.shadow_agreements / self.shadow_scored if self.shadow_scored else 0.0

    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
        return ordered[idx]


def _hash_bucket(request_id: str) -> float:
    """Deterministic 0–100 bucket for a request id (stable across runs)."""
    digest = hashlib.sha256(request_id.encode()).hexdigest()
    return (int(digest[:8], 16) % 10000) / 100.0


class DynamicBatcher:
    """Collects single requests and flushes them as model batches.

    ``add`` returns ``None`` until a flush is due; ``flush`` forces the
    pending batch out. In production this would be driven by a timer;
    here ``max_wait_ms`` is enforced on wall-clock time so the behaviour
    is testable without threads.
    """

    def __init__(self, max_batch_size: int, max_wait_ms: float) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._pending: list[tuple[str, list[float]]] = []
        self._first_arrival: float | None = None
        self.batches_served = 0
        self.total_items = 0

    def add(self, request_id: str, features: list[float]) -> Matrix | None:
        now = time.monotonic()
        if not self._pending:
            self._first_arrival = now
        self._pending.append((request_id, features))
        if len(self._pending) >= self.max_batch_size:
            return self.flush()
        return None

    def flush_due(self) -> bool:
        if not self._pending or self._first_arrival is None:
            return False
        elapsed_ms = (time.monotonic() - self._first_arrival) * 1000.0
        return elapsed_ms >= self.max_wait_ms

    def flush(self) -> Matrix:
        batch = [features for _, features in self._pending]
        self._pending = []
        self._first_arrival = None
        if batch:
            self.batches_served += 1
            self.total_items += len(batch)
        return batch

    @property
    def pending(self) -> int:
        return len(self._pending)


class ServingLayer:
    """Routes requests across champion / challenger / fallback models."""

    def __init__(
        self,
        champion_fn: PredictFn,
        champion_version: str,
        config: ServingConfig | None = None,
        challenger_fn: PredictFn | None = None,
        challenger_version: str | None = None,
        fallback_fn: PredictFn | None = None,
        fallback_version: str = "linear-baseline",
    ) -> None:
        self.champion_fn = champion_fn
        self.champion_version = champion_version
        self.challenger_fn = challenger_fn
        self.challenger_version = challenger_version
        self.fallback_fn = fallback_fn
        self.fallback_version = fallback_version
        self.config = config or ServingConfig()
        self.metrics = ServingMetrics()
        self.batcher = DynamicBatcher(
            self.config.max_batch_size, self.config.max_wait_ms
        )

    # -- routing -----------------------------------------------------------
    def _route(self, request_id: str) -> str:
        if self.challenger_fn is not None and _hash_bucket(request_id) < self.config.canary_pct:
            return "challenger"
        return "champion"

    def _should_shadow(self, request_id: str) -> bool:
        return (
            self.challenger_fn is not None
            and _hash_bucket("shadow:" + request_id) < self.config.shadow_pct
        )

    # -- core --------------------------------------------------------------
    def predict_one(self, request_id: str, features: list[float]) -> PredictionResult:
        start = time.monotonic()
        route = self._route(request_id)
        fn = self.challenger_fn if route == "challenger" else self.champion_fn
        version = self.challenger_version if route == "challenger" else self.champion_version
        probs = fn([features])[0]
        latency_ms = (time.monotonic() - start) * 1000.0
        served_by = route

        # Latency budget: fall back to the cheap model instead of
        # returning a slow answer.
        if (
            self.config.fallback_enabled
            and self.fallback_fn is not None
            and latency_ms > self.config.latency_budget_ms
        ):
            fb_start = time.monotonic()
            probs = self.fallback_fn([features])[0]
            latency_ms = (time.monotonic() - fb_start) * 1000.0
            served_by = "fallback"
            version = self.fallback_version

        # Shadow scoring: challenger scores champion traffic, response
        # is untouched; only the agreement is recorded.
        shadow_agreement: bool | None = None
        if served_by == "champion" and self._should_shadow(request_id):
            shadow_probs = self.challenger_fn([features])[0]
            shadow_agreement = (
                max(range(len(shadow_probs)), key=lambda j: shadow_probs[j])
                == max(range(len(probs)), key=lambda j: probs[j])
            )
            self.metrics.shadow_scored += 1
            if shadow_agreement:
                self.metrics.shadow_agreements += 1

        self.metrics.total += 1
        if served_by == "champion":
            self.metrics.champion += 1
        elif served_by == "challenger":
            self.metrics.challenger += 1
        else:
            self.metrics.fallback += 1
        self.metrics.latencies_ms.append(latency_ms)
        return PredictionResult(
            probabilities=probs,
            model_version=version or "unknown",
            served_by=served_by,
            latency_ms=latency_ms,
            shadow_agreement=shadow_agreement,
        )

    def predict_batch(self, requests: Sequence[tuple[str, list[float]]]) -> list[PredictionResult]:
        """Serve a batch of (request_id, features) pairs, one model call
        per routing group (champion / challenger), then fallbacks."""
        results: dict[str, PredictionResult] = {}
        groups: dict[str, list[tuple[str, list[float]]]] = {"champion": [], "challenger": []}
        for request_id, features in requests:
            groups[self._route(request_id)].append((request_id, features))
        for route, items in groups.items():
            if not items:
                continue
            fn = self.challenger_fn if route == "challenger" else self.champion_fn
            version = self.challenger_version if route == "challenger" else self.champion_version
            start = time.monotonic()
            probs_batch = fn([f for _, f in items])
            batch_ms = (time.monotonic() - start) * 1000.0
            per_item_ms = batch_ms / len(items)
            features_by_id = dict(items)
            for (request_id, _), probs in zip(items, probs_batch):
                served_by = route
                item_version = version
                final_probs = probs
                latency_ms = per_item_ms
                if (
                    self.config.fallback_enabled
                    and self.fallback_fn is not None
                    and per_item_ms > self.config.latency_budget_ms
                ):
                    fb_start = time.monotonic()
                    final_probs = self.fallback_fn([features_by_id[request_id]])[0]
                    latency_ms = (time.monotonic() - fb_start) * 1000.0
                    served_by = "fallback"
                    item_version = self.fallback_version
                results[request_id] = PredictionResult(
                    probabilities=final_probs,
                    model_version=item_version or "unknown",
                    served_by=served_by,
                    latency_ms=latency_ms,
                )
                self.metrics.total += 1
                if served_by == "champion":
                    self.metrics.champion += 1
                elif served_by == "challenger":
                    self.metrics.challenger += 1
                else:
                    self.metrics.fallback += 1
                self.metrics.latencies_ms.append(latency_ms)
        # Preserve caller order.
        return [results[request_id] for request_id, _ in requests]
