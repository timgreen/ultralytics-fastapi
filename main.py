import os
import io
import json
import shutil
from datetime import datetime
from typing import Literal, Dict, Tuple

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query, Depends
from PIL import Image
from ultralytics import YOLO

# Configuration from environment variables
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "8080"))
DATA_DIR = os.getenv("DATA_DIR", "/data")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Default models
DEFAULT_DET_MODEL = "yolo26n.pt"
DEFAULT_CLS_MODEL = "yolo26n-cls.pt"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set working directory to DATA_DIR
os.chdir(DATA_DIR)

app = FastAPI(
    title="Ultralytics YOLO FastAPI",
    description="A simple FastAPI wrapper for Ultralytics YOLO models.",
    version="0.6.3"
)

# Model Cache to avoid repeated loading
model_cache: Dict[str, YOLO] = {}


class InferenceParam:
    """Grouped parameters for model selection and image storage."""
    def __init__(
        self,
        model_name: str | None = Query(None, description="Ultralytics model name (e.g., yolo26n.pt)."),
        usecase: str | None = Query(None, description="Load model from <DATA_DIR>/<usecase>/<task>.pt."),
        store_image: bool = Query(False, description="Save uploaded image under <usecase_dir>/saved/. Only works if 'usecase' is set."),
        x1: float | None = Query(None, description="ROI left coordinate"),
        y1: float | None = Query(None, description="ROI top coordinate"),
        x2: float | None = Query(None, description="ROI right coordinate"),
        y2: float | None = Query(None, description="ROI bottom coordinate")
    ):
        self.model_name = model_name
        self.usecase = usecase
        self.store_image = store_image
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # Validation logic
        if self.usecase and self.model_name:
            raise HTTPException(status_code=400, detail="Cannot use 'usecase' and 'model_name' together.")
        if self.store_image and not self.usecase:
            raise HTTPException(status_code=400, detail="'store_image' can only be used together with 'usecase'.")
        
        # ROI validation: either all ROI parameters must be present or none
        roi_params = [self.x1, self.y1, self.x2, self.y2]
        if any(v is not None for v in roi_params) and not all(v is not None for v in roi_params):
            raise HTTPException(status_code=400, detail="All ROI coordinates (x1, y1, x2, y2) must be provided together.")


def get_model(model_name: str) -> YOLO:
    """Load and cache YOLO model from name or path."""
    if model_name not in model_cache:
        try:
            # If the model_name is a standard filename, store/load it from the models directory
            if not os.path.isabs(model_name) and not model_name.startswith((".", "/")):
                model_path = os.path.join(MODELS_DIR, model_name)
            else:
                model_path = model_name

            model_cache[model_name] = YOLO(model_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model '{model_name}': {str(e)}")
    return model_cache[model_name]


def save_request_image(image: Image.Image, usecase: str):
    """Save the request image under <DATA_DIR>/<usecase>/saved/ with a timestamp."""
    saved_dir = os.path.join(DATA_DIR, usecase, "saved")
    os.makedirs(saved_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(saved_dir, f"{timestamp}.jpg")

    image.convert("RGB").save(filepath, "JPEG")


async def load_image_and_model(
    file: UploadFile,
    params: InferenceParam,
    task: str
) -> Tuple[Image.Image, YOLO, str]:
    """Shared logic for loading image, model, and handling storage."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Resolve model path
    if params.usecase:
        resolved_name = os.path.join(DATA_DIR, params.usecase, f"{task}.pt")
        if not os.path.exists(resolved_name):
            raise HTTPException(
                status_code=500,
                detail=f"Usecase model not found at {resolved_name}."
            )
    else:
        resolved_name = params.model_name or (DEFAULT_DET_MODEL if task == "predict" else DEFAULT_CLS_MODEL)

    model = get_model(resolved_name)

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Optional storage
    if params.store_image and params.usecase:
        save_request_image(image, params.usecase)

    # ROI Cropping
    if params.x1 is not None:
        try:
            image = image.crop((params.x1, params.y1, params.x2, params.y2))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to crop image with provided ROI: {str(e)}")

    return image, model, resolved_name


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Ultralytics YOLO FastAPI is running"}


@app.post(
    "/init",
    summary="Initialize a new usecase",
    description="Creates a new usecase directory and copies the default YOLO26 model as the starting point."
)
async def init_usecase(
    usecase: str = Query(..., description="Name of the usecase to initialize."),
    task: Literal["predict", "classify"] = Query("classify", description="The task type (predict or classify).")
):
    """
    Initializes a usecase folder and populates it with a default model.
    """
    usecase_dir = os.path.join(DATA_DIR, usecase)
    target_path = os.path.join(usecase_dir, f"{task}.pt")

    if os.path.exists(target_path):
        raise HTTPException(status_code=400, detail=f"Model already exists for usecase '{usecase}' at {target_path}")

    # Ensure usecase directory exists
    os.makedirs(usecase_dir, exist_ok=True)

    # Determine source model
    default_model_name = DEFAULT_CLS_MODEL if task == "classify" else DEFAULT_DET_MODEL
    
    # Ensure source model is downloaded by triggering get_model
    get_model(default_model_name)
    source_path = os.path.join(MODELS_DIR, default_model_name)

    try:
        shutil.copy2(source_path, target_path)
        return {
            "message": f"Usecase '{usecase}' initialized successfully.",
            "usecase": usecase,
            "task": task,
            "model_path": target_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize usecase: {str(e)}")


@app.post(
    "/predict",
    summary="Run YOLO detection on an image",
    description="Inference for object detection.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {
                                "box": {"x1": 742.5, "y1": 49.3, "x2": 880.1, "y2": 312.4},
                                "class": "person",
                                "confidence": 0.92
                            }
                        ],
                        "model": "yolo26n.pt",
                        "roi_used": {"x1": 100.0, "y1": 100.0, "x2": 400.0, "y2": 400.0}
                    }
                },
                "image/jpeg": {}
            }
        }
    }
)
async def predict(
    file: UploadFile = File(..., description="The image file to run inference on."),
    params: InferenceParam = Depends(),
    format: Literal["json", "image", "image+metadata"] = Query(
        "json",
        description="Response format. 'json', 'image', or 'image+metadata' (image with JSON in headers)."
    ),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold.")
):
    try:
        image, model, model_id = await load_image_and_model(file, params, "predict")

        # Run inference
        results = model(image, conf=threshold)

        # Prepare JSON results
        predictions = []
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].tolist()
                predictions.append({
                    "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                    "class": model.names[int(box.cls.item())],
                    "confidence": box.conf.item()
                })

        roi_info = {"x1": params.x1, "y1": params.y1, "x2": params.x2, "y2": params.y2} if params.x1 is not None else None

        if format == "json":
            return {"predictions": predictions, "model": model_id, "roi_used": roi_info}

        # Handle image response
        annotated_frame = results[0].plot()
        annotated_frame_rgb = annotated_frame[..., ::-1]
        img = Image.fromarray(annotated_frame_rgb)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        headers = {}
        if format == "image+metadata":
            headers["X-Inference-Results"] = json.dumps({
                "predictions": predictions, 
                "model": model_id,
                "roi_used": roi_info
            })

        return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/classify",
    summary="Run YOLO classification on an image or ROI",
    description="Inference for image classification.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "top1": {"class": "tabby", "confidence": 0.85},
                        "top5": [
                            {"class": "tabby", "confidence": 0.85},
                            {"class": "tiger cat", "confidence": 0.10},
                            {"class": "Egyptian cat", "confidence": 0.02},
                            {"class": "lynx", "confidence": 0.01},
                            {"class": "leopard", "confidence": 0.01}
                        ],
                        "roi_used": {"x1": 100.0, "y1": 100.0, "x2": 400.0, "y2": 400.0},
                        "model": "yolo26n-cls.pt"
                    }
                }
            }
        }
    }
)
async def classify(
    file: UploadFile = File(..., description="The image file to classify."),
    params: InferenceParam = Depends()
):
    try:
        image, model, model_id = await load_image_and_model(file, params, "classify")

        # Run inference
        results = model(image)
        probs = results[0].probs

        top5 = [
            {"class": model.names[idx], "confidence": conf}
            for idx, conf in zip(probs.top5, probs.top5conf.tolist())
        ]

        return {
            "top1": {"class": model.names[probs.top1], "confidence": probs.top1conf.item()},
            "top5": top5,
            "roi_used": {"x1": params.x1, "y1": params.y1, "x2": params.x2, "y2": params.y2} if params.x1 is not None else None,
            "model": model_id
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
