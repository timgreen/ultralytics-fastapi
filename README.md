# Ultralytics FastAPI

A simple FastAPI wrapper for Ultralytics YOLO models, designed to be easily deployable via Docker and integrated with GitHub Container Registry.

## Features

- **FastAPI Endpoints**: 
  - `/predict`: Object detection.
  - `/classify`: Image or ROI classification.
- **Ultralytics YOLO**: Uses YOLO26 nano models by default (`yolo26n.pt` and `yolo26n-cls.pt`).
- **Dynamic Model Loading**: Specify any Ultralytics model name in the request.
- **Usecase Support**: Organize custom models and save request images by usecase directory.
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
- `model_name` (optional): Ultralytics model identifier.
- `usecase` (optional): Load model from `<DATA_DIR>/<usecase>/predict.pt`.
- `store_image` (optional): Boolean (default `false`). If `true`, saves image under `<DATA_DIR>/<usecase>/saved/`. Only works if `usecase` is set.
- `format` (optional): `json` (default), `image`, or `image+metadata`.
- `threshold` (optional): Confidence threshold (0.0 to 1.0, default `0.5`).

*Note: `model_name` and `usecase` are mutually exclusive.*

**Example (Usecase + Store Image)**:
```bash
curl -X POST -F "file=@image.jpg" "http://localhost:8080/predict?usecase=garage&store_image=true"
```

---

### 2. Image Classification (`/classify`)

**Endpoint**: `POST /classify`

**Parameters**:
- `model_name` (optional): Ultralytics model identifier.
- `usecase` (optional): Load model from `<DATA_DIR>/<usecase>/classify.pt`.
- `store_image` (optional): Boolean (default `false`). If `true`, saves image under `<DATA_DIR>/<usecase>/saved/`. Only works if `usecase` is set.
- `x1`, `y1`, `x2`, `y2` (optional): Coordinates for a Region of Interest (ROI).

**Example (Usecase + ROI + Store Image)**:
```bash
curl -X POST -F "file=@image.jpg" "http://localhost:8080/classify?usecase=garage&store_image=true&x1=100&y1=100&x2=400&y2=400"
```

## Development

- The project uses `flake8` for linting in CI.
- Models are automatically downloaded to `DATA_DIR/models/` or loaded from custom usecase paths.
- Saved images are stored under `<DATA_DIR>/<usecase>/saved/` with a timestamp filename.
- **Tip**: Use `store_image=true` to collect real-world data from your specific environment. This data is invaluable for fine-tuning your custom models to handle local lighting, shadows, and edge cases.
- All models are cached in memory after the first load.
