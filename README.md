# Ultralytics FastAPI

A simple FastAPI wrapper for Ultralytics YOLO models, designed to be easily deployable via Docker and integrated with GitHub Container Registry.

## Features

- **FastAPI Endpoints**: 
  - `/predict`: Object detection.
  - `/classify`: Image or ROI classification.
- **Ultralytics YOLO**: Uses YOLO26 nano models by default (`yolo26n.pt` and `yolo26n-cls.pt`).
- **Dynamic Model Loading**: Specify any Ultralytics model name in the request.
- **Dockerized**: Based on the official Ultralytics Docker image.
- **CI/CD**: Automatically builds and pushes images to GHCR using Node.js 24.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST`   | The address to listen on. | `localhost` (Local) / `0.0.0.0` (Docker) |
| `PORT`   | The port to listen on. | `8080` |
| `DATA_DIR`| Directory for storing models and persistent data. | `/data` |

## API Usage

### 1. Object Detection (`/predict`)

**Endpoint**: `POST /predict`

**Parameters**:
- `model_name` (optional): Ultralytics model identifier (default `yolo26n.pt`).
- `format` (optional): `json` (default), `image`, or `image+metadata`.
- `threshold` (optional): Confidence threshold (0.0 to 1.0, default `0.5`).

**Example (Custom Model)**:
```bash
curl -X POST -F "file=@image.jpg" "http://localhost:8080/predict?model_name=yolov10s.pt"
```

---

### 2. Image Classification (`/classify`)

**Endpoint**: `POST /classify`

**Parameters**:
- `model_name` (optional): Ultralytics model identifier (default `yolo26n-cls.pt`).
- `x1`, `y1`, `x2`, `y2` (optional): Coordinates for a Region of Interest (ROI).

**Example (Custom Model + ROI)**:
```bash
curl -X POST -F "file=@image.jpg" "http://localhost:8080/classify?model_name=yolo26s-cls.pt&x1=0&y1=0&x2=200&y2=200"
```

## Development

- The project uses `flake8` for linting in CI.
- Models are automatically downloaded to `DATA_DIR/models/` and cached in memory.
