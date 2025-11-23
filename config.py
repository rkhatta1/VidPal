# config.py
from pydantic_settings import BaseSettings
from pydantic import Field
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
    FACE_LANDMARKER_PATH: str = "/app/models/face_landmarker.task"
    
    FRAME_RATE: float = Field(default=30.0)
    VIDEO_WIDTH: int = Field(default=1920)
    VIDEO_HEIGHT: int = Field(default=1080)
    
    # ========== Audio/Diarization ==========
    SPEECH_LANGUAGE_CODE: str = Field(default="en-US")
    # "latest_long" is standard for long-form audio in GCloud
    SPEECH_MODEL: str = Field(default="latest_long") 
    MIN_SPEAKERS: int = Field(default=2)
    MAX_SPEAKERS: int = Field(default=6)
    EXPECTED_SPEAKERS: int = Field(default=3)

    # ========== EDL Rules ==========
    MIN_SHOT_DURATION: float = Field(default=2.0)
    WIDE_OPENING_DURATION: float = Field(default=3.0)
    
    REACTION_KEYWORDS: list[str] = Field(default=[
        "laugh", "haha", "wow", "no way", "amazing", "oh my god"
    ])
    
    # ========== LLM Configuration ==========
    REFINE_WITH_LLM: bool = Field(default=True)
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash") # Faster/Cheaper for editing
    
    # ========== RAG ==========
    USE_RAG: bool = Field(default=True)
    EMBEDDING_MODEL: str = Field(default="text-embedding-004")
    EMBEDDING_DIM: int = Field(default=768)

    class Config:
        env_file = ".env"
        extra = 'ignore'

def get_settings() -> Settings:
    return Settings()
