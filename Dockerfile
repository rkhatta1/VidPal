FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY db/ ./db/
COPY edl/ ./edl/
COPY fcpxml/ ./fcpxml/
COPY processing/ ./processing/
COPY rag/ ./rag/
COPY utils/ ./utils/

COPY config.py pipeline.py ./

COPY models/ ./models/

RUN mkdir -p /tmp/vidpal_processing

EXPOSE 5000
