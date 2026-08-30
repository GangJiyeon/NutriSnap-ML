from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from src.inference.predict import predict_image

app = FastAPI(title="Food Recognition API")

# 로컬 프론트(react-scripts dev server) 연동 테스트용. 배포 시엔 실제 프론트 도메인으로 좁혀야 함.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    try:
        detections = predict_image(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
    return {"detections": detections}
