FROM python:3.11-slim

WORKDIR /app

# 서버는 ONNX Runtime만 쓰므로 PyTorch/ultralytics 없이 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config.yaml .
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
