from functools import lru_cache
from io import BytesIO

from PIL import Image
from ultralytics import YOLO

from src.config import get_config


@lru_cache
def get_model() -> YOLO:
    config = get_config()
    return YOLO(config["model_path"])


def predict_image(image_bytes: bytes) -> list[dict]:
    config = get_config()
    threshold = config["confidence_threshold"]
    max_detections = config["max_detections"]

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = get_model().predict(image, verbose=False)[0]

    detections = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        detection = {
            "food_name": result.names[int(box.cls[0])],
            "confidence": round(confidence, 2),
            "bbox": [round(v) for v in box.xyxy[0].tolist()],
        }
        if confidence < threshold:
            detection["needs_review"] = True
        detections.append(detection)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:max_detections]
