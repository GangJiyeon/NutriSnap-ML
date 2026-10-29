# NutriSnap ML

Food detection model and inference API for **NutriSnap**, a nutrition-tracking app. Given a meal photo, it returns the detected foods with confidence scores and bounding boxes; the backend then looks up nutrition facts for each food.

This repository is the ML component of a team project (the full app lives in a private repository).

## Highlights

- **Fine-tuned YOLOv8n on 93k AIHub food images (38 classes)**: mAP50 **0.970**, mAP50-95 **0.887** on 11.7k held-out validation images.
- **Found and explained a real-world failure mode**: the model misses foods in multi-dish table photos. Traced it to the training data, where every one of the 93k images contains a single food type.
- **PyTorch-free serving**: reimplemented YOLOv8 pre/post-processing on ONNX Runtime + NumPy/OpenCV. Docker image went from **10.2 GB to 732 MB**, memory from **478 MB to 133 MB**, and predictions are identical to Ultralytics on 200 validation images.

## How it works

```
photo ──► FastAPI /predict ──► letterbox (OpenCV) ──► YOLOv8n (ONNX Runtime) ──► NMS + box rescale ──► [{food_name, confidence, bbox}]
                                                                                                              │
                                                                                         NutriSnap backend ◄──┘ (nutrition lookup, portion, storage)
```

## Dataset and preprocessing

**Source**: [AIHub food image dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=242), first bundle (`음식001`). Images come with one bounding-box JSON file each. Download requires AIHub approval and is limited to Korean residents.

| Split | Images | Boxes | Classes |
|---|---|---|---|
| Train | 93,573 | 120,405 | 38 |
| Validation | 11,679 | 14,966 | 38 |

`src/preprocessing/json_to_yolo.py` converts the AIHub JSON labels into YOLO format. Several issues in the raw data showed up during conversion:

- **Validation labels cover 87 classes, but training covers only 38.** Keeping all of them would create 49 classes with zero training images and skew evaluation. The class list is therefore built from the training split only, and validation-only classes are skipped with a warning.
- **A typo in a label folder name**: `닭파프라카볶음` in the labels vs. `닭파프리카볶음` in the images. Because of the mismatch, the whole class silently dropped out until the label folder was renamed to match.
- **Minor noise**: 33 validation labels point to image files that do not exist, and 2 boxes fall slightly outside the image. These are skipped.

## Training

| Setting | Value |
|---|---|
| Base model | YOLOv8n (COCO pretrained), 3.1M parameters |
| Image size / epochs | 640 / 100 (early-stopping patience 20) |
| Batch size | 27 (auto-selected for 8 GB VRAM) |
| Hardware / time | RTX 3050 8 GB, ~22 hours |

## Results

Validation set (11,679 images):

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.945 | 0.951 | **0.970** | **0.887** |

Out of 38 classes, 22 reach mAP50 ≥ 0.99 and 32 reach ≥ 0.95. The weakest classes:

| Class | Val images | mAP50 | mAP50-95 | Note |
|---|---|---|---|---|
| 돼지감자 (Jerusalem artichoke) | 29 | 0.787 | 0.635 | very few validation images, so the score is noisy |
| 모짜렐라치즈 (mozzarella) | 34 | 0.790 | 0.581 | recall 0.69; often appears as small pieces on top of other dishes |
| 모듬초밥 (assorted sushi) | 92 | 0.863 | 0.748 | |

Inference on the trained model (RTX 3050) takes about 2 ms per image.

## Limitations: multi-dish photos

The validation score is measured on images from the same AIHub distribution as the training data. Testing on real-world photos showed a clear gap:

- On a restaurant photo with six or seven dishes on the table, the model found nothing. The steak on the table scored **0.01** confidence.
- Cropping only the steak plate and running the model again gave **0.80**, so the model does recognize steak.

Analyzing the training labels explains the gap:

| Training label statistic | Value |
|---|---|
| Food types per image | **1 in 100% of images** |
| Boxes per image | 1 box: 81%, 2: 9%, 3: 10% (same food repeated, e.g. several macarons) |
| Median box area | 27% of the image |

The model has never seen several different foods in one photo, and the foods it has seen are usually large and centered. For now, the app should ask users to photograph one dish at a time. Possible fixes are listed in the [Roadmap](#roadmap).

## Serving with ONNX Runtime

The original server loaded the `.pt` model through Ultralytics, which pulls in PyTorch, including CUDA libraries even on CPU-only servers. The server now runs the exported ONNX model directly:

- `src/inference/predict.py` reimplements letterbox resizing, class-wise NMS, and rescaling of boxes back to the original image with NumPy/OpenCV. Class names are read from the ONNX model metadata.
- `requirements.txt` (server) and `requirements-train.txt` (training/export) are split, so the Docker image installs only the server dependencies.

| | PyTorch + Ultralytics (before) | **ONNX Runtime (after)** |
|---|---|---|
| Docker image size | 10.2 GB | **732 MB** |
| Image build time | 8 min 6 s | **43 s** |
| Container memory after first request | 478 MB | **133 MB** |
| First request latency (single run) | 944 ms | **181 ms** |

**Parity check**: on 200 validation images, the new pipeline and Ultralytics (running the same ONNX model) return the same number of boxes on every image and the same class for all 266 boxes, with 0.0 confidence difference and a mean box IoU of 0.999.

One lesson from this step: the first version resized images with PIL instead of OpenCV, and confidence scores drifted by up to 0.25. Matching the training-time preprocessing (OpenCV `INTER_LINEAR`) made the outputs identical.

## API

### `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" -F "image=@sample.jpg"
```

```json
{
  "detections": [
    {"food_name": "마카롱", "confidence": 0.94, "bbox": [140, 2, 526, 774]},
    {"food_name": "마카롱", "confidence": 0.91, "bbox": [584, 532, 964, 777]}
  ]
}
```

- `bbox` is `[x1, y1, x2, y2]` in original image pixels.
- Detections below `confidence_threshold` are returned with `"needs_review": true`, so the app can ask the user to confirm them.
- Non-image uploads return `400`.

### `GET /health`

Returns `{"status": "ok"}`.

### Configuration (`config.yaml`)

```yaml
confidence_threshold: 0.7   # below this, detections are flagged needs_review
max_detections: 5           # maximum foods returned per image
model_path: "models/food_yolov8.onnx"
```

## Getting started

### Run the inference server (Docker)

Place the trained `models/food_yolov8.onnx` first (model weights are not tracked in git).

```bash
docker build -t nutrisnap-ml .
docker run -p 8000:8000 nutrisnap-ml
```

### Local setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# inference server only
pip install -r requirements.txt

# training + ONNX export (install the CUDA build of torch first for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-train.txt
```

### Train and export

```bash
# 1. Convert AIHub labels to YOLO format.
#    Expects data/raw/{training,validation}/ with one folder whose name contains "라벨" (labels) and one containing "원천" (images).
#    Use --copy on Windows (symlinks need admin rights).
python src/preprocessing/json_to_yolo.py --input data/raw/ --output data/processed/ --copy

# 2. Fine-tune (best weights are copied to models/food_yolov8.pt)
python src/training/train.py --epochs 100
python src/training/train.py --resume        # continue an interrupted run

# 3. Export to ONNX for serving
python src/inference/export_onnx.py

# 4. Run the server and tests
uvicorn src.api.main:app --port 8000
pytest
```

AIHub zip files store Korean file names in CP949. Some unzip tools garble them, so extract with a tool that supports CP949, for example Python's `zipfile.ZipFile(path, metadata_encoding="cp949")`.

## Project structure

```
├── src/
│   ├── preprocessing/json_to_yolo.py   # AIHub JSON -> YOLO labels
│   ├── training/train.py               # YOLOv8 fine-tuning (+ --resume)
│   ├── inference/
│   │   ├── predict.py                  # ONNX Runtime inference, pre/post-processing
│   │   └── export_onnx.py              # .pt -> .onnx
│   └── api/main.py                     # FastAPI app (/predict, /health)
├── tests/
│   ├── test_api.py                     # API schema and error handling
│   └── test_postprocess.py             # letterbox, NMS, box rescaling
├── config.yaml
├── Dockerfile
├── requirements.txt                    # server (ONNX Runtime, no PyTorch)
└── requirements-train.txt              # training / export (Ultralytics, PyTorch)
```

`data/`, `models/`, and `runs/` are not tracked in git.

## Scope

This service only answers "which foods are in this photo, and where". The NutriSnap backend handles the rest:

- **Nutrition lookup**: after the food name is confirmed, it is looked up in the nutrition table (Supabase).
- **Portion size**: currently set manually in the app (default 1 serving). Volume estimation with depth models is a separate research track.
- **Storage and auth**: Supabase.

## Roadmap

- [ ] Real-world test set: 30 to 50 phone and multi-dish photos with labels, to measure the gap shown above
- [ ] Multi-dish robustness: synthetic multi-food composites and/or a multi-dish dataset, evaluated on the real-world test set
- [ ] More classes from additional AIHub bundles (`음식002` and later)
- [ ] CI that runs `pytest` on every push
- [ ] Deploy the ONNX server (serverless option such as Modal, or a small VM)

## My contributions

- Built the ML component end to end: data preprocessing, fine-tuning, evaluation and error analysis, ONNX serving, the FastAPI server, tests, and Docker.
- Contributed to selected backend features of the main app using Spring Boot.

## Timeline

- **Jul 10 – Aug 30, 2026**: inference API skeleton, preprocessing, training and export scripts, API tests and Docker
- **Oct 26 – Oct 31, 2026**: full training run on 93k images, error analysis, ONNX Runtime serving, documentation
