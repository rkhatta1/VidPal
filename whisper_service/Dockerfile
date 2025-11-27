FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + explicit cuDNN 9 libraries
RUN apt-get update && apt-get install -y --allow-change-held-packages\
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    libcudnn9-cuda-12 \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Ensure cuDNN libraries are in the library path
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install WhisperX and dependencies
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Copy app code
COPY app.py .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
