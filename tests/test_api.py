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


def test_predict_rejects_too_large_upload(client):
    too_large = b"0" * (10 * 1024 * 1024 + 1)
    response = client.post("/predict", files={"image": ("big.jpg", too_large, "image/jpeg")})
    assert response.status_code == 413


def test_classes_lists_model_names_in_id_order(monkeypatch, client):
    monkeypatch.setattr("src.api.main.get_model", lambda: _FakeModel([], {1: "김치", 0: "밥"}))
    response = client.get("/classes")
    assert response.json() == {"classes": ["밥", "김치"]}


def test_demo_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_samples_lists_images_with_labels(monkeypatch, tmp_path, client):
    (tmp_path / "green_olives.jpg").write_bytes(b"")
    (tmp_path / "._green_olives.jpg").write_bytes(b"macOS metadata")
    (tmp_path / "folder.jpg").mkdir()
    (tmp_path / "notes.txt").write_text("not an image")
    monkeypatch.setattr("src.api.main.SAMPLES_DIR", tmp_path)

    response = client.get("/samples")

    assert response.json() == {"samples": [{"url": "/static/samples/green_olives.jpg", "label": "Green Olives"}]}
