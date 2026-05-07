import os
import io
import json
from datetime import datetime
import numpy as np
from typing import Literal, Optional, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from ultralytics import YOLO
from PIL import Image
import uvicorn

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
    version="0.4.0"
)

# Model Cache to avoid repeated loading
model_cache: Dict[str, YOLO] = {}

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

def resolve_model_path(usecase: Optional[str], model_name: Optional[str], task: str) -> str:
    """Resolve model path based on usecase or model_name."""
    if usecase and model_name:
        raise HTTPException(status_code=400, detail="Cannot use 'usecase' and 'model_name' together.")
    
    if usecase:
        # Use case logic: <DATA_DIR>/<usecase>/<task>.pt
        model_path = os.path.join(DATA_DIR, usecase, f"{task}.pt")
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=500, 
                detail=f"Usecase model not found at {model_path}. Please ensure the file exists."
            )
        return model_path
    
    # If no usecase, return model_name or the task-specific default
    if model_name:
        return model_name
    
    return DEFAULT_DET_MODEL if task == "predict" else DEFAULT_CLS_MODEL

def save_request_image(image: Image.Image, usecase: str):
    """Save the request image under <DATA_DIR>/<usecase>/saved/ with a timestamp."""
    saved_dir = os.path.join(DATA_DIR, usecase, "saved")
    os.makedirs(saved_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join(saved_dir, filename)
    
    image.convert("RGB").save(filepath, "JPEG")
    return filepath

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Ultralytics YOLO FastAPI is running"}

@app.post(
    "/predict",
    summary="Run YOLO detection on an image",
    description="Upload an image and receive JSON predictions, an annotated image, or both."
)
async def predict(
    file: UploadFile = File(..., description="The image file to run inference on."),
    model_name: Optional[str] = Query(None, description="Ultralytics model name to use for detection."),
    usecase: Optional[str] = Query(None, description="Load model from <DATA_DIR>/<usecase>/predict.pt"),
    store_image: bool = Query(False, description="Save the uploaded image under <usecase_dir>/saved/. Only works if 'usecase' is set."),
    format: Literal["json", "image", "image+metadata"] = Query(
        "json",
        description="Response format. 'json', 'image', or 'image+metadata'."
    ),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold.")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    if store_image and not usecase:
        raise HTTPException(status_code=400, detail="'store_image' can only be used together with 'usecase'.")

    try:
        resolved_name = resolve_model_path(usecase, model_name, "predict")
        model = get_model(resolved_name)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if store_image and usecase:
            save_request_image(image, usecase)

        results = model(image, conf=threshold)

        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist() 
                c = box.cls.item()
                conf = box.conf.item()
                predictions.append({
                    "box": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]},
                    "class": model.names[int(c)],
                    "confidence": conf
                })

        if format == "json":
            return {"predictions": predictions, "model": resolved_name}

        annotated_frame = results[0].plot()
        annotated_frame_rgb = annotated_frame[..., ::-1]
        img = Image.fromarray(annotated_frame_rgb)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        
        headers = {}
        if format == "image+metadata":
            headers["X-Inference-Results"] = json.dumps({"predictions": predictions, "model": resolved_name})
            
        return Response(
            content=buf.getvalue(), 
            media_type="image/jpeg",
            headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/classify",
    summary="Run YOLO classification on an image or ROI",
    description="Upload an image and optionally specify a Region of Interest (ROI) to classify."
)
async def classify(
    file: UploadFile = File(..., description="The image file to classify."),
    model_name: Optional[str] = Query(None, description="Ultralytics model name to use for classification."),
    usecase: Optional[str] = Query(None, description="Load model from <DATA_DIR>/<usecase>/classify.pt"),
    store_image: bool = Query(False, description="Save the uploaded image under <usecase_dir>/saved/. Only works if 'usecase' is set."),
    x1: Optional[float] = Query(None, description="ROI left coordinate"),
    y1: Optional[float] = Query(None, description="ROI top coordinate"),
    x2: Optional[float] = Query(None, description="ROI right coordinate"),
    y2: Optional[float] = Query(None, description="ROI bottom coordinate")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    if store_image and not usecase:
        raise HTTPException(status_code=400, detail="'store_image' can only be used together with 'usecase'.")

    try:
        resolved_name = resolve_model_path(usecase, model_name, "classify")
        model = get_model(resolved_name)
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if store_image and usecase:
            save_request_image(image, usecase)

        if all(v is not None for v in [x1, y1, x2, y2]):
            image = image.crop((x1, y1, x2, y2))

        results = model(image)
        
        result = results[0]
        probs = result.probs
        
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        
        top5 = []
        for idx, conf in zip(probs.top5, probs.top5conf.tolist()):
            top5.append({
                "class": model.names[idx],
                "confidence": conf
            })

        return {
            "top1": {"class": model.names[top1_idx], "confidence": top1_conf},
            "top5": top5,
            "roi_used": {"x1": x1, "y1": y1, "x2": x2, "y2": y2} if all(v is not None for v in [x1, y1, x2, y2]) else None,
            "model": resolved_name
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
