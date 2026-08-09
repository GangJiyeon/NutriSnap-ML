# 추론 서버 구현 (1차)

AIHub 데이터 다운로드 승인을 기다리는 동안, 실제 음식 모델 없이도 검증 가능한 부분(추론 서버 골격)부터 먼저 구현했다. `config.yaml` → `predict.py` → `main.py`로 이어지는 파이프라인을 만들고, COCO pretrained `yolov8n.pt`로 임시 스모크 테스트까지 완료한 상태.

## 1. 구조

```
src/
├── config.py              # config.yaml 로더
├── inference/
│   └── predict.py         # 모델 로드 + 추론 로직
└── api/
    └── main.py             # FastAPI 앱 (/health, /predict)
```

호출 흐름:

```
클라이언트
  └─ POST /predict (image 파일)
       └─ main.py: predict() 가 파일을 bytes로 읽음
            └─ predict.py: predict_image(bytes) 가 추론 실행
                 └─ config.py: get_config() 로 threshold/max_detections/model_path 읽음
       └─ {"detections": [...]} 반환
```

## 2. `src/config.py`

```python
from functools import lru_cache
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

@lru_cache
def get_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- `config.yaml`(프로젝트 루트)을 읽어 dict로 반환.
- `CONFIG_PATH`를 `__file__` 기준 상대 경로로 계산해서, 어느 디렉토리에서 실행하든(`uvicorn src.api.main:app`을 루트에서 돌리든, 다른 스크립트에서 import하든) 항상 같은 `config.yaml`을 찾도록 함.
- `@lru_cache`: 첫 호출 시 파일을 읽고 이후로는 캐시된 dict를 재사용. 요청마다 파일 I/O가 발생하지 않도록 함. (단점: 런타임 중 `config.yaml`을 수정해도 서버 재시작 전까지 반영 안 됨 — 지금 단계에서는 문제 없음)

## 3. `src/inference/predict.py`

```python
@lru_cache
def get_model() -> YOLO:
    config = get_config()
    return YOLO(config["model_path"])
```

- YOLO 모델은 로드 비용이 크기 때문에(가중치 파일 읽기 + GPU/CPU 초기화) 요청마다 새로 만들지 않고, 프로세스 생애주기 동안 한 번만 로드해서 재사용.
- `config["model_path"]`가 가리키는 가중치(`models/food_yolov8.pt`, 학습 후 생성됨)를 로드. 아직 학습 전이라 이 파일은 존재하지 않음 — 스모크 테스트 때는 임시로 `yolov8n.pt`(COCO)로 바꿔서 검증.

```python
def predict_image(image_bytes: bytes) -> list[dict]:
    config = get_config()
    threshold = config["confidence_threshold"]
    max_detections = config["max_detections"]

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    result = get_model().predict(image, verbose=False)[0]
```

- 업로드된 원본 바이트를 `PIL.Image`로 디코딩. `.convert("RGB")`로 통일하는 이유: PNG 등 알파 채널(RGBA)이나 흑백(L) 이미지가 들어와도 YOLO 입력 형식을 일정하게 유지하기 위함.
- `model.predict(...)`는 리스트를 반환하는데(배치 추론 지원 때문), 이미지 1장만 넣었으므로 `[0]`으로 그 결과 하나만 꺼냄.

```python
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
```

- `result.boxes`: 탐지된 객체마다 클래스ID(`cls`), 신뢰도(`conf`), 좌표(`xyxy`)를 담고 있는 텐서 모음.
- `result.names[int(box.cls[0])]`: 클래스ID(정수)를 실제 이름(문자열)으로 변환. `yolov8n.pt`는 COCO 클래스명("bus", "person" 등), 나중에 파인튜닝된 `food_yolov8.pt`는 음식 클래스명을 반환하게 됨 — **코드 수정 없이 모델 교체만으로 전환 가능**.
- `box.xyxy[0]`: `[x1, y1, x2, y2]` 형식의 bbox 좌표(좌상단, 우하단). README 스펙과 동일한 형식.
- `confidence < threshold`일 때만 `needs_review: true` 키를 추가 — `docs/info.md`의 응답 예시(70% 미만만 태그 붙음)와 동일하게 맞춤.

```python
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:max_detections]
```

- 신뢰도 높은 순으로 정렬 후 `config.yaml`의 `max_detections`(기본 5개)만큼만 잘라서 반환. 한 이미지에 탐지된 게 5개보다 많아도 상위 5개만 응답.

## 4. `src/api/main.py`

```python
app = FastAPI(title="Food Recognition API")

@app.get("/health")
def health():
    return {"status": "ok"}
```

- 헬스체크용. 배포 환경(Modal 등)에서 서버가 살아있는지 확인하는 용도로 씀.

```python
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    return {"detections": predict_image(image_bytes)}
```

- `UploadFile = File(...)`: `multipart/form-data`로 온 `image` 필드를 파일로 받겠다는 선언. `...`은 필수값이라는 뜻(없으면 자동으로 422 응답).
- `await image.read()`로 파일 전체를 bytes로 읽고, `predict_image()`에 그대로 전달.
- 반환한 dict는 FastAPI가 자동으로 JSON 직렬화해서 응답.

## 5. 스모크 테스트 (2026-08-08)

아직 파인튜닝된 음식 모델이 없어서, `config.yaml`의 `model_path`를 일시적으로 `"yolov8n.pt"`(COCO pretrained, 자동 다운로드됨)로 바꿔서 서버 전체 흐름만 검증했다. 테스트 후 `model_path`는 다시 `"models/food_yolov8.pt"`로 원복.

```bash
uvicorn src.api.main:app --port 8000
curl -X POST http://localhost:8000/predict -F "image=@bus.jpg"
```

```json
{
  "detections": [
    {"food_name": "bus", "confidence": 0.87, "bbox": [23, 231, 805, 757]},
    {"food_name": "person", "confidence": 0.87, "bbox": [49, 399, 245, 903]},
    {"food_name": "person", "confidence": 0.85, "bbox": [669, 392, 810, 877]},
    {"food_name": "person", "confidence": 0.83, "bbox": [222, 406, 345, 858]},
    {"food_name": "person", "confidence": 0.26, "bbox": [0, 551, 63, 873], "needs_review": true}
  ]
}
```

`/health` 200 OK, 신뢰도 순 정렬, `max_detections` 제한, `needs_review` 태깅까지 README/`docs/info.md` 스펙대로 동작 확인.

## 6. 알려진 한계 (다음에 다룰 것)

- `/predict`에 이미지가 아닌 파일이나 손상된 파일이 들어오면 예외 처리 없이 500 에러가 그대로 노출됨 — 배포 전 `try/except` 필요.
- `models/food_yolov8.pt`가 아직 없어서 실제 운영 상태로는 실행 불가 (AIHub 데이터 승인 → `json_to_yolo.py` → `train.py` 순서로 학습 후 생성 예정).
- `tests/` 폴더에 자동화된 테스트 없음.
