from fastapi import FastAPI, File, UploadFile

from src.inference.predict import predict_image

app = FastAPI(title="Food Recognition API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    return {"detections": predict_image(image_bytes)}
