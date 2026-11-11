"""API tests — auto-skip when FastAPI/httpx are not installed."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dlserve.service import app, create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    assert app is not None
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["champion"] == "mlp-v1"


def test_predict(client):
    resp = client.post(
        "/predict", json={"request_id": "t-1", "features": [0.1] * 16}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in (0, 1, 2)
    assert len(body["probabilities"]) == 3
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-6
    assert body["served_by"] in ("champion", "challenger", "fallback")


def test_predict_wrong_feature_count(client):
    resp = client.post("/predict", json={"request_id": "t-2", "features": [0.1] * 4})
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(
        "/predict/batch",
        json=[
            {"request_id": "b-1", "features": [0.1] * 16},
            {"request_id": "b-2", "features": [-0.5] * 16},
        ],
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_predict_batch_empty(client):
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_metrics_after_traffic(client):
    client.post("/predict", json={"request_id": "m-1", "features": [0.2] * 16})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert "p95_latency_ms" in body
    assert 0.0 <= body["fallback_rate"] <= 1.0


def test_create_app_builds_independently():
    other = create_app(seed=7)
    assert other is not None
    with TestClient(other) as c:
        assert c.get("/health").status_code == 200
