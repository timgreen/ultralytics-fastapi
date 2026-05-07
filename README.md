# Ultralytics FastAPI

A simple FastAPI wrapper for Ultralytics YOLO models, designed to be easily deployable via Docker and integrated with GitHub Container Registry.

## Features

- **FastAPI Endpoints**: 
  - `/predict`: Object detection.
  - `/classify`: Image or ROI classification.
- **Ultralytics YOLO**: Uses YOLOv8 nano models (`yolov8n.pt` and `yolov8n-cls.pt`).
- **Dockerized**: Based on the official Ultralytics Docker image.
- **CI/CD**: Automatically builds and pushes images to GHCR using Node.js 24.

## Environment Variables

The following environment variables can be used to configure the application:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST`   | The address to listen on. | `localhost` (Local) / `0.0.0.0` (Docker) |
| `PORT`   | The port to listen on. | `8080` |
| `DATA_DIR`| Directory for storing models and persistent data. | `/data` |

## Getting Started

### Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python main.py
   ```

### Running with Docker

```bash
docker pull ghcr.io/timgreen/ultralytics-fastapi:main
docker run -p 8080:8080 -v $(pwd)/data:/data ghcr.io/timgreen/ultralytics-fastapi:main
```

## API Usage

### 1. Object Detection (`/predict`)

**Endpoint**: `POST /predict`

**Parameters**:
- `format` (optional): `json` (default), `image`, or `image+metadata`.
- `threshold` (optional): Confidence threshold (0.0 to 1.0, default `0.5`).

**Example (JSON)**:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8080/predict
```

**Example (Annotated Image)**:
```bash
curl -X POST -F "file=@image.jpg" "http://localhost:8080/predict?format=image" --output results.jpg
```

---

### 2. Image Classification (`/classify`)

**Endpoint**: `POST /classify`

**Parameters**:
- `x1`, `y1`, `x2`, `y2` (optional): Coordinates for a Region of Interest (ROI). If all 4 are provided, the image is cropped before classification.

**Example (Whole image)**:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8080/classify
```

**Example (With ROI)**:
```bash
curl -X POST -F "file=@image.jpg" "http://localhost:8080/classify?x1=100&y1=100&x2=400&y2=400"
```

**Example Response**:
```json
{
  "top1": {
    "class": "tabby",
    "confidence": 0.85
  },
  "top5": [ ... ],
  "roi_used": { "x1": 100, "y1": 100, "x2": 400, "y2": 400 }
}
```

## Development

- The project uses `flake8` for linting in CI.
- Models are automatically downloaded to `DATA_DIR`.
