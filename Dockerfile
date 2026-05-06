# Use official Ultralytics image as base
FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /app

# Install FastAPI and Uvicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=8080
ENV DATA_DIR=/data

# Create data directory
RUN mkdir -p $DATA_DIR

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "main.py"]
