"""Tests for the versioned registry and promotion gate (registry.py)."""
import pytest

from dlserve.registry import ModelRegistry, ModelVersion, PromotionGate


def _version(name, metrics, stage="staging"):
    return ModelVersion(
        name="tabular-mlp", version=name, metrics=metrics,
        artifact_path=f"/tmp/{name}.json", stage=stage,
    )


def test_register_and_get(registry):
    v = _version("v1", {"val_accuracy": 0.80})
    registry.register(v)
    assert registry.get("v1").version == "v1"
    assert registry.get("v1").checksum


def test_register_duplicate_rejected(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    with pytest.raises(ValueError):
        registry.register(_version("v1", {"val_accuracy": 0.81}))


def test_get_unknown_raises(registry):
    with pytest.raises(KeyError):
        registry.get("nope")


def test_promotion_gate_requires_approval(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.promote_to_champion("v1", approved=True)
    registry.register(_version("v2", {"val_accuracy": 0.85}))
    _model, decision = registry.promote_to_champion("v2", approved=False)
    assert not decision.approved
    assert "approval" in str(decision).lower()
    assert registry.champion().version == "v1"


def test_promotion_gate_requires_delta(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.promote_to_champion("v1", approved=True)
    registry.register(_version("v2", {"val_accuracy": 0.805}))
    _model, decision = registry.promote_to_champion(
        "v2", gate=PromotionGate(min_delta=0.01), approved=True
    )
    assert not decision.approved
    assert registry.champion().version == "v1"


def test_promotion_gate_latency_budget(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.promote_to_champion("v1", approved=True)
    registry.register(
        _version("v2", {"val_accuracy": 0.90, "p95_latency_ms": 120.0})
    )
    _model, decision = registry.promote_to_champion(
        "v2", gate=PromotionGate(max_p95_latency_ms=50.0), approved=True
    )
    assert not decision.approved
    assert "latency" in str(decision).lower()


def test_successful_promotion_supersedes_champion(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.promote_to_champion("v1", approved=True)
    registry.register(_version("v2", {"val_accuracy": 0.85, "p95_latency_ms": 20.0}))
    model, decision = registry.promote_to_champion("v2", approved=True)
    assert decision.approved
    assert model.stage == "production"
    assert registry.champion().version == "v2"
    assert registry.get("v1").stage == "archived"
    assert any(h["event"] == "promoted_to_champion" for h in model.history)


def test_rollback_restores_previous(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.promote_to_champion("v1", approved=True)
    registry.register(_version("v2", {"val_accuracy": 0.85}))
    registry.promote_to_champion("v2", approved=True)
    # v1 was archived by the promotion; re-register semantics: rollback
    # to an archived version must fail, so use a fresh v3 path instead.
    with pytest.raises(ValueError):
        registry.rollback("v1")
    registry.register(_version("v3", {"val_accuracy": 0.82}, stage="staging"))
    registry.promote_to_challenger("v3")
    rolled = registry.rollback("v3")
    assert rolled.stage == "production"
    assert registry.champion().version == "v3"


def test_archived_version_cannot_transition(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.promote_to_champion("v1", approved=True)
    registry.register(_version("v2", {"val_accuracy": 0.85}))
    registry.promote_to_champion("v2", approved=True)
    with pytest.raises(ValueError):
        registry.promote_to_challenger("v1")


def test_registry_persists_to_disk(tmp_path):
    path = tmp_path / "registry.json"
    reg = ModelRegistry(path)
    reg.register(_version("v1", {"val_accuracy": 0.80}))
    reg.promote_to_champion("v1", approved=True)
    reloaded = ModelRegistry(path)
    assert reloaded.champion().version == "v1"
    assert reloaded.get("v1").metrics["val_accuracy"] == 0.80


def test_list_by_stage(registry):
    registry.register(_version("v1", {"val_accuracy": 0.80}))
    registry.register(_version("v2", {"val_accuracy": 0.85}))
    registry.promote_to_challenger("v2")
    assert [v.version for v in registry.list(stage="challenger")] == ["v2"]
    assert len(registry.list()) == 2
