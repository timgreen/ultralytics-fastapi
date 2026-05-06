import os
import io
import numpy as np
from typing import Literal
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

# Load YOLO model
# By default, this downloads yolov8n.pt if not present
model = YOLO("yolov8n.pt")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Ultralytics YOLO FastAPI is running"}

@app.post(
    "/predict",
    summary="Run YOLO inference on an image",
    description="Upload an image and receive either a JSON list of predictions or an annotated image."
)
async def predict(
    file: UploadFile = File(..., description="The image file to run inference on."),
    format: Literal["json", "image"] = Query(
        "json",
        description="The desired response format. 'json' returns prediction metadata, 'image' returns the annotated image."
    )
):
    """
    Runs YOLOv8 inference on the uploaded image.

    - **file**: Image file (JPG, PNG, etc.)
    - **format**: 'json' (default) or 'image'
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        results = model(image)

        if format == "image":
            # Plot results on the image
            # annotated_frame is BGR numpy array
            annotated_frame = results[0].plot()
            # Convert BGR to RGB
            annotated_frame_rgb = annotated_frame[..., ::-1]
            img = Image.fromarray(annotated_frame_rgb)

            # Save to buffer
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return Response(content=buf.getvalue(), media_type="image/jpeg")

        # Default to JSON results
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
                    "class": model.names[int(c)],
                    "confidence": conf
                })

        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
