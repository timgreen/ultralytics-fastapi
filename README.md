# Ultralytics FastAPI

A simple FastAPI wrapper for Ultralytics YOLO models, designed to be easily deployable via Docker and integrated with GitHub Container Registry.

## Features

- **FastAPI Endpoint**: `/predict` for image inference.
- **Ultralytics YOLO**: Uses the latest YOLOv8 (defaulting to `yolov8n.pt`).
- **Dockerized**: Based on the official Ultralytics Docker image.
- **CI/CD**: Automatically builds and pushes images to GHCR.

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
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host localhost --port 8080
   ```

### Running with Docker

You can pull the image from GHCR (once published):
```bash
docker pull ghcr.io/timgreen/ultralytics-fastapi:main
```

Or build it locally:
```bash
docker build -t ultralytics-fastapi .
docker run -p 8080:8080 ultralytics-fastapi
```

## API Usage

### Predict

**Endpoint**: `POST /predict`

**Parameters**:
- `format` (query string, optional): `json` (default), `image`, or `image+metadata`.
- `threshold` (query string, optional): Confidence threshold between 0.0 and 1.0 (default `0.5`).

**Request**: Multipart form-data with an `file` field containing the image.

**Example using `curl` (JSON response)**:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8080/predict
```

**Example using `curl` (Annotated image response)**:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" "http://localhost:8080/predict?format=image" --output results.jpg
```

**Example using `curl` (Image + Metadata response)**:
Returns the image in the body and JSON in the `X-Inference-Results` header.
```bash
curl -v -X POST -F "file=@path/to/your/image.jpg" "http://localhost:8080/predict?format=image+metadata" --output results.jpg
```

**Example Response**:

```json
{
  "predictions": [
    {
      "box": {
        "x1": 100.0,
        "y1": 150.0,
        "x2": 200.0,
        "y2": 250.0
      },
      "class": "person",
      "confidence": 0.95
    }
  ]
}
```

## Development

- The project uses `black` for formatting and `flake8` for linting.
- Models are automatically downloaded by the Ultralytics library on first use.
