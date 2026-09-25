"""API tests use a fake model, so they run in seconds without downloading
anything. We are testing OUR code (routing, validation, response shape),
not the neural network."""

from fastapi.testclient import TestClient

from app.main import app
from app.model import get_model


class FakeModel:
    def predict(self, texts):
        return [{"text": t, "label": "negative", "confidence": 0.9,
                 "scores": {"negative": 0.9, "neutral": 0.05, "positive": 0.05}}
                for t in texts]


app.dependency_overrides[get_model] = lambda: FakeModel()
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_single():
    r = client.post("/predict", json={"text": "balance kat gaya"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_empty_text_rejected():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_batch_limit():
    r = client.post("/predict/batch", json={"texts": ["a"] * 33})
    assert r.status_code == 422


def test_batch_ok():
    r = client.post("/predict/batch", json={"texts": ["acha", "bura"]})
    assert len(r.json()["predictions"]) == 2
