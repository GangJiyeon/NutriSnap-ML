"""ONNX 추론 전처리(letterbox)/후처리(NMS, 좌표 복원) 테스트. 모델 파일 없이 돌아간다."""
import numpy as np
import pytest
from PIL import Image

from src.inference.predict import letterbox, postprocess

NUM_CLASSES = 3


def _model_output(candidates: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
    """(cx, cy, w, h, class_id, score) 후보 목록을 YOLOv8 출력 모양 (1, 4+클래스수, 후보수)으로 만든다."""
    out = np.zeros((1, 4 + NUM_CLASSES, len(candidates)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(candidates):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out


def test_letterbox_keeps_ratio_and_pads_center():
    tensor, ratio, (left, top) = letterbox(Image.new("RGB", (200, 100), "white"), 64)

    assert tensor.shape == (1, 3, 64, 64)
    assert ratio == pytest.approx(0.32)
    assert (left, top) == (0, 16)
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)  # 위쪽 패딩
    assert tensor[0, 0, 32, 32] == pytest.approx(1.0)  # 가운데는 원본(흰색)


def test_postprocess_drops_low_confidence():
    output = _model_output([(32, 32, 10, 10, 0, 0.1)])
    assert postprocess(output, 1.0, (0, 0), (64, 64)) == []


def test_postprocess_merges_overlapping_boxes_of_same_class():
    output = _model_output([(32, 32, 20, 20, 0, 0.9), (33, 33, 20, 20, 0, 0.8)])
    detections = postprocess(output, 1.0, (0, 0), (64, 64))

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)


def test_postprocess_keeps_overlapping_boxes_of_different_classes():
    output = _model_output([(32, 32, 20, 20, 0, 0.9), (33, 33, 20, 20, 1, 0.8)])
    detections = postprocess(output, 1.0, (0, 0), (64, 64))

    assert sorted(d.class_id for d in detections) == [0, 1]


def test_postprocess_maps_boxes_back_to_original_image():
    # 200x100 원본을 64로 letterbox하면 ratio 0.32, 위쪽 패딩 16
    output = _model_output([(32, 32, 32, 16, 2, 0.9)])
    (detection,) = postprocess(output, 0.32, (0, 16), (200, 100))

    assert detection.class_id == 2
    assert detection.xyxy == pytest.approx([50.0, 25.0, 150.0, 75.0])
