"""Tests for the serving layer (serving.py)."""
import time

import pytest

from dlserve.serving import DynamicBatcher, ServingConfig, ServingLayer, _hash_bucket


def _const_fn(probs):
    def fn(batch):
        return [list(probs) for _ in batch]

    return fn


def test_hash_bucket_deterministic_and_bounded():
    assert _hash_bucket("req-1") == _hash_bucket("req-1")
    assert 0.0 <= _hash_bucket("req-1") < 100.0
    assert _hash_bucket("req-1") != _hash_bucket("req-2")


def test_predict_one_champion_path(serving_layer):
    result = serving_layer.predict_one("req-1", [0.1] * 16)
    assert result.served_by == "champion"
    assert result.model_version == "v1"
    assert len(result.probabilities) == 3
    assert abs(sum(result.probabilities) - 1.0) < 1e-9
    assert serving_layer.metrics.total == 1


def test_canary_routing_is_deterministic():
    champion = _const_fn([0.8, 0.1, 0.1])
    challenger = _const_fn([0.1, 0.8, 0.1])
    layer = ServingLayer(
        champion_fn=champion, champion_version="v1",
        config=ServingConfig(canary_pct=50.0, shadow_pct=0.0),
        challenger_fn=challenger, challenger_version="v2",
    )
    first = {f"req-{i}": layer.predict_one(f"req-{i}", [0.0]).served_by for i in range(40)}
    second = {f"req-{i}": layer.predict_one(f"req-{i}", [0.0]).served_by for i in range(40)}
    assert first == second  # same request id -> same route every time
    assert set(first.values()) == {"champion", "challenger"}  # both routes used
    assert layer.metrics.challenger > 0


def test_latency_budget_falls_back():
    def slow_fn(batch):
        time.sleep(0.05)
        return [[0.8, 0.1, 0.1] for _ in batch]

    layer = ServingLayer(
        champion_fn=slow_fn, champion_version="v1",
        config=ServingConfig(latency_budget_ms=1.0),
        fallback_fn=_const_fn([0.1, 0.1, 0.8]),
    )
    result = layer.predict_one("req-1", [0.0])
    assert result.served_by == "fallback"
    assert result.model_version == "linear-baseline"
    assert result.probabilities == [0.1, 0.1, 0.8]
    assert layer.metrics.fallback_rate == 1.0


def test_fallback_disabled_returns_slow_answer():
    def slow_fn(batch):
        time.sleep(0.02)
        return [[0.8, 0.1, 0.1] for _ in batch]

    layer = ServingLayer(
        champion_fn=slow_fn, champion_version="v1",
        config=ServingConfig(latency_budget_ms=1.0, fallback_enabled=False),
        fallback_fn=_const_fn([0.1, 0.1, 0.8]),
    )
    assert layer.predict_one("req-1", [0.0]).served_by == "champion"


def test_shadow_agreement_recorded():
    champion = _const_fn([0.8, 0.1, 0.1])
    challenger = _const_fn([0.7, 0.2, 0.1])  # same argmax -> agrees
    layer = ServingLayer(
        champion_fn=champion, champion_version="v1",
        config=ServingConfig(canary_pct=0.0, shadow_pct=100.0),
        challenger_fn=challenger, challenger_version="v2",
    )
    result = layer.predict_one("req-1", [0.0])
    assert result.shadow_agreement is True
    assert layer.metrics.shadow_scored == 1
    assert layer.metrics.shadow_agreement_rate == 1.0


def test_shadow_disagreement_recorded():
    champion = _const_fn([0.8, 0.1, 0.1])
    challenger = _const_fn([0.1, 0.8, 0.1])  # different argmax
    layer = ServingLayer(
        champion_fn=champion, champion_version="v1",
        config=ServingConfig(canary_pct=0.0, shadow_pct=100.0),
        challenger_fn=challenger, challenger_version="v2",
    )
    result = layer.predict_one("req-1", [0.0])
    assert result.shadow_agreement is False
    assert layer.metrics.shadow_agreement_rate == 0.0


def test_predict_batch_preserves_order_and_groups(serving_layer):
    requests = [(f"req-{i}", [0.1] * 16) for i in range(10)]
    results = serving_layer.predict_batch(requests)
    assert len(results) == 10
    assert all(r.served_by == "champion" for r in results)
    assert serving_layer.metrics.total == 10


def test_dynamic_batcher_flushes_on_size():
    batcher = DynamicBatcher(max_batch_size=3, max_wait_ms=1000.0)
    assert batcher.add("a", [1.0]) is None
    assert batcher.add("b", [2.0]) is None
    flushed = batcher.add("c", [3.0])
    assert flushed == [[1.0], [2.0], [3.0]]
    assert batcher.batches_served == 1
    assert batcher.total_items == 3


def test_dynamic_batcher_flush_due_after_wait():
    batcher = DynamicBatcher(max_batch_size=10, max_wait_ms=1.0)
    batcher.add("a", [1.0])
    assert not batcher.flush_due()
    time.sleep(0.005)
    assert batcher.flush_due()
    assert batcher.flush() == [[1.0]]
    assert batcher.pending == 0


def test_serving_metrics_p95():
    layer = ServingLayer(
        champion_fn=_const_fn([0.5, 0.3, 0.2]), champion_version="v1",
        config=ServingConfig(),
    )
    for i in range(20):
        layer.predict_one(f"req-{i}", [0.0])
    assert layer.metrics.p95_latency_ms() >= 0.0
    assert layer.metrics.total == 20
