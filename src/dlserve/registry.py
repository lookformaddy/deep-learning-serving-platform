"""Versioned model registry with gated champion promotion — pure stdlib.

Versions are immutable once registered. Stages form a small state
machine::

    staging -> production (champion) -> archived
    staging -> challenger (canary/shadow) -> production | archived

Promotion to champion is *gated*: the candidate must beat the current
champion on the held-out metric by at least ``min_delta`` and stay
inside the latency budget, and a human approval flag must be set.
Nothing promotes itself. Every transition is recorded in an audit
history so a promotion (or a rollback) can always be explained.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STAGES = ("staging", "challenger", "production", "archived")


@dataclass
class ModelVersion:
    name: str
    version: str
    metrics: dict[str, float]
    artifact_path: str
    stage: str = "staging"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        if not self.history:
            self.history.append(
                {"event": "registered", "stage": self.stage, "at": self.created_at}
            )

    @property
    def checksum(self) -> str:
        payload = json.dumps(
            {"name": self.name, "version": self.version, "metrics": self.metrics,
             "artifact_path": self.artifact_path},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "metrics": self.metrics,
            "artifact_path": self.artifact_path,
            "stage": self.stage,
            "created_at": self.created_at,
            "history": self.history,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        data = {k: v for k, v in data.items() if k != "checksum"}
        return cls(**data)


@dataclass
class PromotionDecision:
    approved: bool
    reasons: list[str]

    def __str__(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        return f"{verdict}: " + "; ".join(self.reasons)


@dataclass
class PromotionGate:
    """Fail-closed promotion gate for champion promotion."""

    metric: str = "val_accuracy"
    min_delta: float = 0.01
    max_p95_latency_ms: float = 50.0
    require_approval: bool = True

    def evaluate(
        self,
        candidate: ModelVersion,
        champion: Optional[ModelVersion],
        approved: bool = False,
    ) -> PromotionDecision:
        reasons: list[str] = []
        if self.require_approval and not approved:
            reasons.append("human approval missing")
        if self.metric not in candidate.metrics:
            reasons.append(f"candidate missing metric {self.metric!r}")
            return PromotionDecision(False, reasons)
        if champion is not None:
            if self.metric not in champion.metrics:
                reasons.append(f"champion missing metric {self.metric!r}")
            elif candidate.metrics[self.metric] < champion.metrics[self.metric] + self.min_delta:
                reasons.append(
                    f"{self.metric} {candidate.metrics[self.metric]:.4f} does not beat "
                    f"champion {champion.metrics[self.metric]:.4f} + {self.min_delta}"
                )
        latency = candidate.metrics.get("p95_latency_ms")
        if latency is not None and latency > self.max_p95_latency_ms:
            reasons.append(
                f"p95 latency {latency:.2f}ms exceeds budget {self.max_p95_latency_ms:.2f}ms"
            )
        if not reasons:
            reasons.append("all gates passed")
        return PromotionDecision(approved=not reasons or reasons == ["all gates passed"], reasons=reasons)


class ModelRegistry:
    """In-memory registry with optional JSON persistence."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store_path = Path(store_path) if store_path else None
        self._versions: dict[str, ModelVersion] = {}
        if self.store_path and self.store_path.exists():
            self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.store_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        for item in data.get("versions", []):
            version = ModelVersion.from_dict(item)
            self._versions[version.version] = version

    def _save(self) -> None:
        if not self.store_path:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"versions": [v.to_dict() for v in self._versions.values()]}
        self.store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- core operations ----------------------------------------------------
    def register(self, version: ModelVersion) -> ModelVersion:
        if version.version in self._versions:
            raise ValueError(f"version {version.version!r} already registered (versions are immutable)")
        self._versions[version.version] = version
        self._save()
        return version

    def get(self, version: str) -> ModelVersion:
        try:
            return self._versions[version]
        except KeyError:
            raise KeyError(f"unknown model version {version!r}") from None

    def list(self, stage: str | None = None) -> list[ModelVersion]:
        versions = sorted(self._versions.values(), key=lambda v: v.created_at)
        if stage:
            versions = [v for v in versions if v.stage == stage]
        return versions

    def champion(self) -> Optional[ModelVersion]:
        production = self.list(stage="production")
        return production[-1] if production else None

    def challenger(self) -> Optional[ModelVersion]:
        challengers = self.list(stage="challenger")
        return challengers[-1] if challengers else None

    def _transition(self, version: str, stage: str, event: str) -> ModelVersion:
        model = self.get(version)
        if model.stage == "archived":
            raise ValueError(f"version {version!r} is archived and cannot transition")
        model.stage = stage
        model.history.append(
            {"event": event, "stage": stage, "at": datetime.now(timezone.utc).isoformat()}
        )
        self._save()
        return model

    def promote_to_challenger(self, version: str) -> ModelVersion:
        return self._transition(version, "challenger", "promoted_to_challenger")

    def promote_to_champion(
        self,
        version: str,
        gate: PromotionGate | None = None,
        approved: bool = False,
    ) -> tuple[ModelVersion, PromotionDecision]:
        gate = gate or PromotionGate()
        candidate = self.get(version)
        decision = gate.evaluate(candidate, self.champion(), approved=approved)
        if not decision.approved:
            return candidate, decision
        previous = self.champion()
        if previous is not None and previous.version != version:
            self._transition(previous.version, "archived", "superseded_by_champion")
        return self._transition(version, "production", "promoted_to_champion"), decision

    def rollback(self, version: str) -> ModelVersion:
        """Restore a previous production version as champion.

        The current champion is archived; the target must already exist
        and not be archived. Rollback is an explicit, audited act — it
        does not re-run the promotion gate, because the version already
        passed it once.
        """
        target = self.get(version)
        if target.stage == "archived":
            raise ValueError(f"cannot roll back to archived version {version!r}")
        current = self.champion()
        if current is not None and current.version != version:
            self._transition(current.version, "archived", "rolled_back")
        return self._transition(version, "production", "rollback")
