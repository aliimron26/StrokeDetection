FROM python:3.10-slim

WORKDIR /app

# Install system dependencies untuk OpenCV dan MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode
COPY . .

# Jalankan aplikasi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]