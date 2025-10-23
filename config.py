# config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Literal, Optional
import os


class Settings(BaseSettings):
    """Centralized configuration for VidPalAI pipeline."""
    
    # ========== I/O Paths ==========
    INPUT_DIR: Path = Field(default=Path("input"))
    OUTPUT_DIR: Path = Field(default=Path("output"))
    CACHE_DIR: Path = Field(default=Path(".cache"))
    
    MASTER_AUDIO_FILE: Path = Field(default=Path("input/audio.mp3"))
    VIDEO_FILES: dict[str, Path] = Field(default={
        "cam_host": Path("input/cam_host.mp4"),
        "cam_guest": Path("input/cam_guest.mp4"),
        "cam_wide": Path("input/cam_wide.mp4"),
    })
    
    # ========== Processing Parameters ==========
    PROCESS_DURATION_MINUTES: int = Field(default=5, ge=1)
    VIDEO_INTERVAL_SECONDS: float = Field(default=8.0, ge=0.5)  # Sparse sampling
    GCS_BUCKET_NAME: str = Field(default="vidpalai-temp-audio")

    
    # ========== Audio/Transcription ==========
    USE_GPU: bool = Field(default=True)
    WHISPER_MODEL: str = Field(default="base")
    WHISPERX_BATCH_SIZE: int = Field(default=16)  # Higher for GPU
    COMPUTE_TYPE: Literal["float16", "int8", "float32"] = Field(default="float16")
    SPEECH_LANGUAGE_CODE: str = Field(default="en-US")
    SPEECH_MODEL: str = Field(default="latest_long")
    EXPECTED_SPEAKERS: Optional[int] = Field(default=None)  # Set if you know exact count
    MIN_SPEAKERS: int = Field(default=2)
    MAX_SPEAKERS: int = Field(default=6)
    # ========== EDL Generation ==========
    MIN_SHOT_DURATION: float = Field(default=2.0, ge=0.5)  # Minimum shot length
    WIDE_OPENING_DURATION: float = Field(default=3.0)  # Opening wide shot duration
    RAPID_WINDOW_SECONDS: float = Field(default=8.0)  # Window to detect rapid changes
    RAPID_CHANGES_THRESHOLD: int = Field(default=3)  # Switches to trigger wide shot
    
    # Reaction keywords to detect in transcript
    REACTION_KEYWORDS: list[str] = Field(default=[
        "laugh", "haha", "wow", "that's crazy", "[laughter]",
        "oh my god", "really", "no way", "amazing"
    ])
    REACTION_SHOT_DURATION: tuple[float, float] = Field(default=(1.5, 3.0))  # Min, max
    
    # ========== LLM Configuration ==========
    REFINE_WITH_LLM: bool = Field(default=False)  # Optional LLM refinement
    GEMINI_MODEL: str = Field(default="gemini-2.5-pro")
    LLM_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    HUGGINGFACE_TOKEN: str = Field(default="")
    # Vertex AI Configuration
    GOOGLE_GENAI_USE_VERTEXAI: bool = Field(default=True)
    GOOGLE_CLOUD_PROJECT: str = Field(default="")
    GOOGLE_CLOUD_LOCATION: str = Field(default="us-central1")
    
    # ========== Database Configuration ==========
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5435)
    POSTGRES_DB: str = Field(default="vidpalai")
    POSTGRES_USER: str = Field(default="vidpalai")
    POSTGRES_PASSWORD: str = Field(default="vidpalai_secure_2025")
    
    # ========== RAG Configuration (updated for Gemini) ==========
    USE_RAG: bool = Field(default=True)
    RAG_CHUNK_SIZE: int = Field(default=30)  # Seconds per chunk
    RAG_TOP_K: int = Field(default=3)
    
    # Gemini Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")
    EMBEDDING_DIM: int = Field(default=768)
    
    # ========== Video/FCPXML ==========
    FRAME_RATE: float = Field(default=30.0)
    VIDEO_WIDTH: int = Field(default=1920)
    VIDEO_HEIGHT: int = Field(default=1080)
    WINDOWS_PROJECT_ROOT: str = Field(default="E:/Random/VidPal")
    
    # ========== Feature Flags ==========
    ENABLE_CACHING: bool = Field(default=True)
    VLM_SPARSE_FPS: float = Field(default=0.2)  # Very sparse if enabled
    
    # ========== VLM Configuration ==========
    ENABLE_VLM_PROCESSING: bool = Field(default=True)  # Disabled by default
    VLM_MODEL: str = Field(default="apple/FastVLM-0.5B")
    VLM_TRANSITION_WINDOW: float = Field(default=1.0)  # Seconds before/after transition
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
