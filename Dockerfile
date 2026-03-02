FROM python:3.11-slim

# Avoid Python writing .pyc files + ensure logs flush
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (LightGBM can require these on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default to demo mode so the container runs after clone
ENV DEMO_MODE=1
ENV DEMO_MODEL_PATH=models/demo/fraudshield_demo.pkl
ENV DEMO_DATA_PATH=data/demo/transactions_demo.parquet
ENV DEMO_TRAIN_CFG=config/training_demo.yaml

# Expose port (optional but nice)
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]