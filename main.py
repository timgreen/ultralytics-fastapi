import os
import io
import json
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

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set working directory to DATA_DIR
os.chdir(DATA_DIR)

app = FastAPI(
    title="Ultralytics YOLO FastAPI",
    description="A simple FastAPI wrapper for Ultralytics YOLO models.",
    version="0.2.1"
)

# Model Cache to avoid repeated loading
model_cache: Dict[str, YOLO] = {}

def get_model(model_name: str) -> YOLO:
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

# Default models
DEFAULT_DET_MODEL = "yolo26n.pt"
DEFAULT_CLS_MODEL = "yolo26n-cls.pt"

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
    model_name: str = Query(DEFAULT_DET_MODEL, description="Ultralytics model name to use for detection."),
    format: Literal["json", "image", "image+metadata"] = Query(
        "json",
        description="Response format. 'json', 'image', or 'image+metadata'."
    ),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold.")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Load model
        model = get_model(model_name)
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        results = model(image, conf=threshold)

        # Prepare JSON predictions
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
            return {"predictions": predictions, "model": model_name}

        # Handle image formats
        annotated_frame = results[0].plot()
        annotated_frame_rgb = annotated_frame[..., ::-1]
        img = Image.fromarray(annotated_frame_rgb)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        
        headers = {}
        if format == "image+metadata":
            headers["X-Inference-Results"] = json.dumps({"predictions": predictions, "model": model_name})
            
        return Response(
            content=buf.getvalue(), 
            media_type="image/jpeg",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/classify",
    summary="Run YOLO classification on an image or ROI",
    description="Upload an image and optionally specify a Region of Interest (ROI) to classify."
)
async def classify(
    file: UploadFile = File(..., description="The image file to classify."),
    model_name: str = Query(DEFAULT_CLS_MODEL, description="Ultralytics model name to use for classification."),
    x1: Optional[float] = Query(None, description="ROI left coordinate"),
    y1: Optional[float] = Query(None, description="ROI top coordinate"),
    x2: Optional[float] = Query(None, description="ROI right coordinate"),
    y2: Optional[float] = Query(None, description="ROI bottom coordinate")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Load model
        model = get_model(model_name)
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Crop if ROI is provided
        if all(v is not None for v in [x1, y1, x2, y2]):
            image = image.crop((x1, y1, x2, y2))

        # Run inference
        results = model(image)
        
        # Parse results
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
            "model": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
