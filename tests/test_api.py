"""FastAPI /health, /predict 엔드포인트 테스트."""
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.inference.predict as predict_module
from src.api.main import app
from src.config import get_config
from src.inference.predict import Detection


class _FakeModel:
    def __init__(self, boxes: list[Detection], names: dict[int, str]):
        self._boxes = boxes
        self.names = names

    def detect(self, image):
        return self._boxes


def _sample_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_schema(monkeypatch, client):
    names = {0: "밥", 1: "김치"}
    boxes = [
        Detection(0, 0.95, [10.0, 20.0, 100.0, 150.0]),
        Detection(1, 0.30, [5.0, 5.0, 50.0, 50.0]),
    ]
    monkeypatch.setattr(predict_module, "get_model", lambda: _FakeModel(boxes, names))

    response = client.post("/predict", files={"image": ("food.jpg", _sample_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert "detections" in body
    assert len(body["detections"]) == 2
    for detection in body["detections"]:
        assert {"food_name", "confidence", "bbox"} <= detection.keys()


def test_predict_flags_low_confidence_as_needs_review(monkeypatch, client):
    threshold = get_config()["confidence_threshold"]
    names = {0: "밥", 1: "김치"}
    boxes = [
        Detection(0, threshold + 0.1, [10.0, 20.0, 100.0, 150.0]),
        Detection(1, threshold - 0.1, [5.0, 5.0, 50.0, 50.0]),
    ]
    monkeypatch.setattr(predict_module, "get_model", lambda: _FakeModel(boxes, names))

    response = client.post("/predict", files={"image": ("food.jpg", _sample_image_bytes(), "image/jpeg")})

    detections = {d["food_name"]: d for d in response.json()["detections"]}
    assert "needs_review" not in detections["밥"]
    assert detections["김치"]["needs_review"] is True


def test_predict_rejects_non_image_file(client):
    response = client.post("/predict", files={"image": ("not_an_image.txt", b"hello world", "text/plain")})
    assert response.status_code == 400
