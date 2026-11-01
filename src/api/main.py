from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from src.inference.predict import get_model, predict_image

# 공개 데모에서 과도하게 큰 업로드로 메모리가 터지지 않도록 제한
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_PAGE = STATIC_DIR / "index.html"
# 데모 페이지의 "Try a sample" 사진. 파일명이 라벨이 된다 (예: macaron.jpg -> "Macaron")
SAMPLES_DIR = STATIC_DIR / "samples"
SAMPLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Food Recognition API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 로컬 프론트(react-scripts dev server) 연동 테스트용. 배포 시엔 실제 프론트 도메인으로 좁혀야 함.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def demo():
    return FileResponse(DEMO_PAGE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/classes")
def classes():
    names = get_model().names
    return {"classes": [names[i] for i in sorted(names)]}


@app.get("/samples", include_in_schema=False)
def samples():
    files = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in SAMPLE_SUFFIXES) if SAMPLES_DIR.is_dir() else []
    return {
        "samples": [
            {"url": f"/static/samples/{p.name}", "label": p.stem.replace("_", " ").replace("-", " ").title()}
            for p in files
        ]
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지는 10MB 이하만 업로드할 수 있습니다.")
    try:
        detections = predict_image(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
    return {"detections": detections}
