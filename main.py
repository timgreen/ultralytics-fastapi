import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import uvicorn

# Configuration from environment variables
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", "8080"))
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Ultralytics YOLO FastAPI")

# Load YOLO model
# By default, this downloads yolov8n.pt if not present
model = YOLO("yolov8n.pt")

@app.get("/")
async def root():
    return {"message": "Ultralytics YOLO FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run inference
        results = model(image)
        
        # Parse results
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
