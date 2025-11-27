import logging
import os
import shutil
import gc
from pathlib import Path
from typing import Optional, List
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-service")

app = FastAPI()

# Global Model Cache to avoid reloading heavy models per request
MODEL_CACHE = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 for GPU, int8 for CPU
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

class DiarizationRequest(BaseModel):
    audio_url: str
    episode_id: str
    min_speakers: Optional[int] = 1
    max_speakers: Optional[int] = 6
    hf_token: str

def download_file(uri: str, dest_path: Path):
    """Downloads media from GCS or copies from local path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if uri.startswith("gs://"):
        try:
            storage_client = storage.Client()
            parts = uri[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(dest_path))
            logger.info(f"Downloaded GCS file: {uri}")
        except Exception as e:
            logger.error(f"GCS Download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download GCS file: {e}")
    elif os.path.exists(uri):
        shutil.copy(uri, dest_path)
        logger.info(f"Copied local file: {uri}")
    else:
        raise HTTPException(status_code=404, detail=f"File not found: {uri}")

@app.post("/diarize")
async def diarize_audio(request: DiarizationRequest):
    # Create a temp directory for this specific request
    temp_dir = Path(f"/tmp/{request.episode_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_audio_path = temp_dir / "input_audio.wav"

    try:
        logger.info(f"Processing episode {request.episode_id} on device {DEVICE}")
        
        # 1. Acquire Audio
        download_file(request.audio_url, local_audio_path)

        # 2. Transcribe (Whisper)
        model_name = "medium"
        if "transcription" not in MODEL_CACHE:
            logger.info(f"Loading Whisper model: {model_name}")
            MODEL_CACHE["transcription"] = whisperx.load_model(
                model_name, DEVICE, compute_type=COMPUTE_TYPE
            )
        
        model = MODEL_CACHE["transcription"]
        audio = whisperx.load_audio(str(local_audio_path))
        
        logger.info("Transcribing audio...")
        result = model.transcribe(audio, batch_size=8)
        
        # 3. Align
        logger.info("Aligning transcription...")
        # Alignment models are small, we can load/unload or cache them
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=DEVICE
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
        )
        
        # Clean up alignment model to free VRAM immediately
        del model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # 4. Diarize (Speaker Identification)
        logger.info("Performing diarization...")
        if "diarization" not in MODEL_CACHE:
             MODEL_CACHE["diarization"] = DiarizationPipeline(
                 use_auth_token=request.hf_token,
                 device=DEVICE
             )
        
        diarize_model = MODEL_CACHE["diarization"]
        
        diarize_segments = diarize_model(
            audio,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers
        )
        
        # 5. Assign Speakers to Words
        logger.info("Assigning speakers to segments...")
        final_result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # WhisperX structure varies slightly by version, ensuring robustness
        segments = final_result.get("segments", [])
        word_segments = final_result.get("word_segments", [])
        
        return {
            "status": "success",
            "segments": segments,
            "word_segments": word_segments
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        }
    
    return {
        "status": "ok", 
        "device": DEVICE, 
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_memory": gpu_info
    }
