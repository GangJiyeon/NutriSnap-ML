"""ONNX Runtime으로 YOLOv8 음식 탐지 모델을 실행한다 (서버에 PyTorch/ultralytics 불필요)."""
import ast
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.config import get_config

# ultralytics predict 기본값과 맞춤
CANDIDATE_CONF = 0.25
NMS_IOU = 0.7
PAD_COLOR = (114, 114, 114)


@dataclass
class Detection:
    class_id: int
    confidence: float
    xyxy: list[float]


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """비율을 유지한 채 size x size로 줄이고 남는 부분은 회색으로 채운다."""
    w, h = image.size
    ratio = min(size / w, size / h)
    new_w, new_h = round(w * ratio), round(h * ratio)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    left, top = round(pad_w - 0.1), round(pad_h - 0.1)
    right, bottom = size - new_w - left, size - new_h - top

    # 학습 때와 같은 결과가 나오도록 ultralytics처럼 cv2 INTER_LINEAR로 리사이즈 (PIL은 안티앨리어싱이 달라 점수가 흔들림)
    resized = cv2.resize(np.asarray(image), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    tensor = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, ratio, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """점수 높은 박스부터 남기고, 그와 많이 겹치는 박스는 버린다."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(
    output: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    image_size: tuple[int, int],
) -> list[Detection]:
    """모델 출력 (1, 4+클래스수, 후보수)을 원본 이미지 좌표의 탐지 결과로 바꾼다."""
    preds = output[0].T  # (후보수, 4+클래스수)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(preds)), class_ids]

    mask = confidences >= CANDIDATE_CONF
    preds, class_ids, confidences = preds[mask], class_ids[mask], confidences[mask]
    if not len(preds):
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # 클래스별로 NMS 하기 위해 클래스마다 박스를 멀리 떨어뜨려 놓는 트릭
    offset_boxes = boxes + class_ids[:, None] * 4096
    keep = nms(offset_boxes, confidences, NMS_IOU)

    # letterbox 좌표 -> 원본 이미지 좌표
    left, top = pad
    img_w, img_h = image_size
    boxes = (boxes - [left, top, left, top]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)

    return [Detection(int(class_ids[i]), float(confidences[i]), boxes[i].tolist()) for i in keep]


class OnnxDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape[2]
        # ultralytics가 export 시 클래스 이름을 메타데이터에 dict 문자열로 넣어둔다
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(metadata["names"])

    def detect(self, image: Image.Image) -> list[Detection]:
        tensor, ratio, pad = letterbox(image, self.input_size)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return postprocess(output, ratio, pad, image.size)


@lru_cache
def get_model() -> OnnxDetector:
    config = get_config()
    return OnnxDetector(config["model_path"])


def predict_image(image_bytes: bytes) -> list[dict]:
    config = get_config()
    threshold = config["confidence_threshold"]
    max_detections = config["max_detections"]

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    model = get_model()

    detections = []
    for det in model.detect(image):
        detection = {
            "food_name": model.names[det.class_id],
            "confidence": round(det.confidence, 2),
            "bbox": [round(v) for v in det.xyxy],
        }
        if det.confidence < threshold:
            detection["needs_review"] = True
        detections.append(detection)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:max_detections]
