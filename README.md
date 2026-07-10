# Food Recognition Model

음식 사진 한 장으로 음식 종류를 탐지(detection)하고, 각 음식에 대한 신뢰도(confidence)를 반환하는 딥러닝 모델 및 추론 API. 
음식 영양소 트래킹 앱의 이미지 처리(딥러닝) 파트를 담당하는 레포입니다.

## 개요

- **모델**: YOLOv8 (Ultralytics), COCO pretrained 가중치를 AIHub 음식 이미지 데이터로 파인튜닝
- **역할**: 사진 → `{음식명, 신뢰도, bbox}` 리스트 반환까지가 이 레포의 책임 범위. 양(포션) 추정, 영양소 계산, DB 저장은 백엔드에서 처리
- **배포**: FastAPI 기반 추론 서버, ONNX Runtime으로 최적화하여 서버리스 GPU(Modal 등)에 배포

## 데이터셋

[AIHub 음식 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=242)

- 500여종 음식 카테고리, 카테고리당 약 60,000장 (전체 약 300만 장)
- 이미지 + 바운딩박스 JSON 어노테이션 페어
- 카테고리별 100g/1인분 기준 영양성분 메타데이터 포함

**다운로드 방법**
1. AIHub 회원가입 후 위 링크에서 데이터 다운로드 신청 (승인 필요, 내국인만 가능)
2. 신청 시 안심존 대상 여부 먼저 확인 (일반 승인 대상인 경우가 대부분이나 헬스케어 카테고리 공통 안내가 붙어있어 확인 권장)
3. 분할 압축 파일(zip.part*) 다운로드 후 병합

```bash
cat food_data.zip.part* > food_data.zip
unzip food_data.zip -d data/raw/
```

**서브셋 구성**: 500클래스 전체 대신 자주 먹는 음식 위주 50~100개 클래스, 클래스당 300~500장으로 구성하여 1차 파인튜닝 진행

## 프로젝트 구조

```
food-recognition-model/
├── data/
│   ├── raw/                  # AIHub 원본 다운로드 (git 추적 제외)
│   ├── processed/            # YOLO 포맷 변환 후 (images/, labels/)
│   └── food_dataset.yaml     # 클래스 목록 + 경로 설정
├── src/
│   ├── preprocessing/
│   │   └── json_to_yolo.py   # AIHub JSON → YOLO txt 변환
│   ├── training/
│   │   └── train.py          # YOLOv8 파인튜닝 스크립트
│   ├── inference/
│   │   ├── predict.py        # 모델 로드 + 추론 로직
│   │   └── export_onnx.py    # .pt → ONNX 변환
│   └── api/
│       └── main.py           # FastAPI 앱 (/predict, /health)
├── models/                   # 학습된 가중치 (.pt, .onnx, git 추적 제외)
├── notebooks/                # 실험/EDA
├── tests/
│   └── test_api.py
├── config.yaml                # threshold, 최대 인식 개수 등 설정값
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

## 로컬 실행 방법

### 1. 환경 설정
```bash
git clone <repo-url>
cd food-recognition-model
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 데이터 전처리
```bash
python src/preprocessing/json_to_yolo.py --input data/raw/ --output data/processed/
```

### 3. 학습 (파인튜닝)
```bash
python src/training/train.py --data data/food_dataset.yaml --epochs 100 --imgsz 640
```
학습된 가중치는 `models/food_yolov8.pt`에 저장됩니다.

### 4. ONNX 변환 (배포용 최적화)
```bash
python src/inference/export_onnx.py --weights models/food_yolov8.pt
```

### 5. 추론 서버 실행
```bash
uvicorn src.api.main:app --reload --port 8000
```

## API 사용 예시

### `POST /predict`

**요청**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@sample.jpg"
```

**응답**
```json
{
  "detections": [
    {"food_name": "흰쌀밥", "confidence": 0.96, "bbox": [12, 34, 210, 300]},
    {"food_name": "제육볶음", "confidence": 0.91, "bbox": [220, 40, 480, 320]},
    {"food_name": "계란찜", "confidence": 0.74, "bbox": [10, 320, 180, 480], "needs_review": true}
  ]
}
```

### `GET /health`
서버 상태 확인용 헬스체크 엔드포인트

## 설정값 (`config.yaml`)

```yaml
confidence_threshold: 0.7   # 이 값 미만은 needs_review: true 태그
max_detections: 5           # 한 이미지에서 최대로 인식할 음식 개수
model_path: "models/food_yolov8.onnx"
```

## 역할 경계 (중요)

이 레포는 아래 항목을 **다루지 않습니다**. 관련 로직은 백엔드/DB에서 처리됩니다.

- 음식 양(포션) 추정 — 현재는 사용자가 앱에서 기본 1.0인분 기준 수동 조절
- 영양소(칼로리 등) 계산 — 음식명 확정 후 Supabase 영양성분 테이블에서 조회
- 음식명 검색/자동완성 — 별도 텍스트 검색 로직

> **음식 부피(양) 추정**은 depth estimation 등을 활용한 별도 연구 트랙으로 분리되어 진행 예정이며, 이 레포의 스코프에 포함되지 않습니다.

## 배포

- **추론 서버**: FastAPI + ONNX Runtime, 서버리스 GPU 환경(Modal 등) 우선 고려 — 초기 트래픽이 적어 비용 효율적이며 콜드스타트 시간도 ONNX 덕분에 단축됨
- **트래픽 증가 시**: 동일 코드로 자체 서버(EC2/GCP VM)로 전환 가능
- 이미지 저장/DB/인증은 Supabase(Storage, Postgres, Auth)에서 처리, 이 레포는 추론 API만 제공

## 향후 과제

- [ ] 서브셋 확장 (사용자 로그 기반 미인식 클래스 추가)
- [ ] bbox 면적 기반 양 추정 근사치 (2차 개선)
- [ ] (별도 트랙) Depth estimation 기반 정밀 부피 추정