import os
import io
import json
from datetime import datetime
from typing import Literal, Optional, Dict, Tuple

import numpy as np
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
    version="0.5.0"
)

# Model Cache to avoid repeated loading
model_cache: Dict[str, YOLO] = {}


class InferenceParams:
    """Grouped parameters for model selection and image storage."""
    def __init__(
        self,
        model_name: Optional[str] = Query(None, description="Ultralytics model name (e.g., yolo26n.pt)."),
        usecase: Optional[str] = Query(None, description="Load model from <DATA_DIR>/<usecase>/<task>.pt."),
        store_image: bool = Query(False, description="Save uploaded image under <usecase_dir>/saved/. Only works if 'usecase' is set.")
    ):
        self.model_name = model_name
        self.usecase = usecase
        self.store_image = store_image
        
        # Validation logic
        if self.usecase and self.model_name:
            raise HTTPException(status_code=400, detail="Cannot use 'usecase' and 'model_name' together.")
        if self.store_image and not self.usecase:
            raise HTTPException(status_code=400, detail="'store_image' can only be used together with 'usecase'.")


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
    params: InferenceParams, 
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
        
    return image, model, resolved_name


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Ultralytics YOLO FastAPI is running"}


@app.post(
    "/predict",
    summary="Run YOLO detection on an image",
    description="Inference for object detection."
)
async def predict(
    file: UploadFile = File(..., description="The image file to run inference on."),
    params: InferenceParams = Depends(),
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

        if format == "json":
            return {"predictions": predictions, "model": model_id}

        # Handle image response
        annotated_frame = results[0].plot()
        annotated_frame_rgb = annotated_frame[..., ::-1]
        img = Image.fromarray(annotated_frame_rgb)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        
        headers = {}
        if format == "image+metadata":
            headers["X-Inference-Results"] = json.dumps({"predictions": predictions, "model": model_id})
            
        return Response(content=buf.getvalue(), media_type="image/jpeg", headers=headers)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/classify",
    summary="Run YOLO classification on an image or ROI",
    description="Inference for image classification."
)
async def classify(
    file: UploadFile = File(..., description="The image file to classify."),
    params: InferenceParams = Depends(),
    x1: Optional[float] = Query(None, description="ROI left coordinate"),
    y1: Optional[float] = Query(None, description="ROI top coordinate"),
    x2: Optional[float] = Query(None, description="ROI right coordinate"),
    y2: Optional[float] = Query(None, description="ROI bottom coordinate")
):
    try:
        image, model, model_id = await load_image_and_model(file, params, "classify")

        # Crop if ROI provided
        if all(v is not None for v in [x1, y1, x2, y2]):
            image = image.crop((x1, y1, x2, y2))

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
            "roi_used": {"x1": x1, "y1": y1, "x2": x2, "y2": y2} if all(v is not None for v in [x1, y1, x2, y2]) else None,
            "model": model_id
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
