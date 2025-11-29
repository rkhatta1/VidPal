# config.py
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path
from typing import Literal, Optional, Dict
import os

class Settings(BaseSettings):
    # ========== Infrastructure ==========
    # Default to 'redis' hostname (docker service name)
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/0")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/0")
    
    # ========== Database ==========
    POSTGRES_HOST: str = Field(default="postgres") # Docker service name
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="vidpalai")
    POSTGRES_USER: str = Field(default="vidpalai")
    POSTGRES_PASSWORD: str = Field(default="vidpalai_secure_2025")
    
    # ========== Google Cloud ==========
    # We expect credentials to be mounted or set via GOOGLE_APPLICATION_CREDENTIALS
    GOOGLE_CLOUD_PROJECT: str = Field(...)
    GOOGLE_CLOUD_LOCATION: str = Field(default="us-central1")
    GCS_BUCKET_NAME: str = Field(...) # Required for processing
    GOOGLE_GENAI_USE_VERTEXAI: bool = Field(default=True)
    
    # ========== API Keys (Fallback) ==========
    # If using Vertex AI, these might not be needed if ADC is set up
    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    
    # ========== Processing Parameters ==========
    # Local temp dir for downloading files from GCS during processing
    TEMP_DIR: Path = Field(default=Path("/tmp/vidpal_processing"))
    BASE_DIR: Path = Field(default=Path(__file__).resolve().parent)
    FACE_LANDMARKER_PATH: str = "models/face_landmarker.task"
    CACHE_DIR: str = ".cache"
    
    FRAME_RATE: float = Field(default=30.0)
    VIDEO_WIDTH: int = Field(default=1920)
    VIDEO_HEIGHT: int = Field(default=1080)
    
    # ========== Audio/Diarization ==========
    SPEAKER_IDENTIFICATION_PROVIDER: str = Field(default="gcs", description="Provider for speaker identification. Either 'local' (Whisper) or 'gcs' (Google Cloud Speech).")
    SPEECH_LANGUAGE_CODE: str = Field(default="en-US")
    # "latest_long" is standard for long-form audio in GCloud
    SPEECH_MODEL: str = Field(default="latest_long") 
    MIN_SPEAKERS: int = Field(default=2)
    MAX_SPEAKERS: int = Field(default=6)
    EXPECTED_SPEAKERS: int = Field(default=3)

    @field_validator('SPEAKER_IDENTIFICATION_PROVIDER')
    def validate_speaker_provider(cls, v):
        if v not in ['local', 'gcs']:
            raise ValueError("SPEAKER_IDENTIFICATION_PROVIDER must be 'local' or 'gcs'")
        return v


    # ========== EDL Rules ==========
    MIN_SHOT_DURATION: float = Field(default=2.0)
    WIDE_OPENING_DURATION: float = Field(default=3.0)
    
    REACTION_KEYWORDS: list[str] = Field(default=[
        "laugh", "haha", "wow", "no way", "amazing", "oh my god"
    ])
    
    # ========== WhisperX Service ==========
    WHISPER_SERVICE_URL: str = Field(default="http://whisper-service:8000")
    HUGGINGFACE_TOKEN: str = Field(...) # Required for Pyannote inside WhisperX
    # ========== LLM Configuration ==========
    REFINE_WITH_LLM: bool = Field(default=True)
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
    
    # ========== RAG ==========
    USE_RAG: bool = Field(default=True)
    EMBEDDING_MODEL: str = Field(default="text-embedding-004")
    EMBEDDING_DIM: int = Field(default=768)

    # ========== MISC ===========
    WINDOWS_PROJECT_ROOT: str = Field(default="E:/Random/VidPal/LatestTest/Updated")
    ENABLE_CACHING: bool = Field(default=True)

    class Config:
        env_file = ".env"
        extra = 'ignore'

def get_settings() -> Settings:
    return Settings()
