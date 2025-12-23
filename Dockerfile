FROM python:3.10-slim

# Install system dependencies required by OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 - <<'PY'
from paddleocr import PaddleOCR
PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False,
          det_model_dir='det_lite', rec_model_dir='rec_lite', cls_model_dir='cls_lite')
PY

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
EXPOSE 8080

# Start command with reduced workers and threads
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "app:app"]
