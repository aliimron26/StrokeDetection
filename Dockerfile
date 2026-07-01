# Gunakan base image Python yang lebih ringan
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb-xfixes0 \
    libxcb-shm0 \
    libxcb-present0 \
    libxcb-sync1 \
    libxcb-randr0 \
    libxcb-glx0 \
    libxcb-xv0 \
    libx11-6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (untuk caching)
COPY requirements.txt .

# Install Python packages dengan retry mechanism
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt || \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt || \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy aplikasi
COPY . .

# Expose port
EXPOSE 8000

# Run dengan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]