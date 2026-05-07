import os
import io
import json
import numpy as np
from typing import Literal, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from ultralytics import YOLO
from PIL import Image
import uvicorn

# Configuration from environment variables
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "8080"))
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Ensure data directory exists and set as working directory
os.makedirs(DATA_DIR, exist_ok=True)
os.chdir(DATA_DIR)

app = FastAPI(
    title="Ultralytics YOLO FastAPI",
    description="A simple FastAPI wrapper for Ultralytics YOLO models.",
    version="0.1.0"
)

# Load YOLO models
# By default, these download the weights if not present
det_model = YOLO("yolov8n.pt")
cls_model = YOLO("yolov8n-cls.pt")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Ultralytics YOLO FastAPI is running"}

@app.post(
    "/predict",
    summary="Run YOLO detection on an image",
    description="Upload an image and receive JSON predictions, an annotated image, or both (image with JSON in headers)."
)
async def predict(
    file: UploadFile = File(..., description="The image file to run inference on."),
    format: Literal["json", "image", "image+metadata"] = Query(
        "json",
        description="The desired response format. 'json' returns prediction metadata, 'image' returns the annotated image, 'image+metadata' returns the image with JSON metadata in the 'X-Inference-Results' header."
    ),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold for filtering detections.")
):
    """
    Runs YOLOv8 detection on the uploaded image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        results = det_model(image, conf=threshold)

        # Prepare JSON predictions
        predictions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get box coordinates in (top, left, bottom, right) format
                b = box.xyxy[0].tolist() 
                c = box.cls.item()
                conf = box.conf.item()
                predictions.append({
                    "box": {
                        "x1": b[0],
                        "y1": b[1],
                        "x2": b[2],
                        "y2": b[3]
                    },
                    "class": det_model.names[int(c)],
                    "confidence": conf
                })

        if format == "json":
            return {"predictions": predictions}

        # Handle image formats
        annotated_frame = results[0].plot()
        annotated_frame_rgb = annotated_frame[..., ::-1]
        img = Image.fromarray(annotated_frame_rgb)

        # Save to buffer
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        
        headers = {}
        if format == "image+metadata":
            headers["X-Inference-Results"] = json.dumps({"predictions": predictions})
            
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
    x1: Optional[float] = Query(None, description="ROI left coordinate"),
    y1: Optional[float] = Query(None, description="ROI top coordinate"),
    x2: Optional[float] = Query(None, description="ROI right coordinate"),
    y2: Optional[float] = Query(None, description="ROI bottom coordinate")
):
    """
    Runs YOLOv8 classification. If ROI coordinates are provided, the image is cropped before classification.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Crop if ROI is provided
        if all(v is not None for v in [x1, y1, x2, y2]):
            image = image.crop((x1, y1, x2, y2))

        # Run inference
        results = cls_model(image)
        
        # Parse results
        result = results[0]
        probs = result.probs
        
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        
        top5_indices = probs.top5
        top5_confs = probs.top5conf.tolist()
        
        top5 = []
        for idx, conf in zip(top5_indices, top5_confs):
            top5.append({
                "class": cls_model.names[idx],
                "confidence": conf
            })

        return {
            "top1": {
                "class": cls_model.names[top1_idx],
                "confidence": top1_conf
            },
            "top5": top5,
            "roi_used": {"x1": x1, "y1": y1, "x2": x2, "y2": y2} if all(v is not None for v in [x1, y1, x2, y2]) else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
