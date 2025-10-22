
# processing/audio_transcription.py
import logging
from typing import List, Dict, Any, Optional
import whisperx
import torch
from config import get_settings
from utils.caching import Cache, compute_file_hash

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Optimized audio transcription with WhisperX."""
    
    def __init__(self, cache: Optional[Cache] = None):
        self.settings = get_settings()
        self.cache = cache
        self.device = "cuda" if (torch.cuda.is_available() and self.settings.USE_GPU) else "cpu"
        self.compute_type = self.settings.COMPUTE_TYPE if self.device == "cuda" else "int8"
        
        logger.info(f"Audio transcriber initialized (device: {self.device})")
    
    def transcribe(
        self,
        audio_path: str,
        duration_limit_seconds: Optional[int] = None,
        speaker_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            duration_limit_seconds: Optional duration limit
            speaker_segments: Optional speaker diarization to enrich transcript
        
        Returns:
            List of word-level transcript entries with speaker labels
        """
        # Check cache
        if self.cache and self.settings.ENABLE_CACHING:
            file_hash = compute_file_hash(audio_path)
            cache_key = f"{file_hash}_{duration_limit_seconds}"
            
            cached = self.cache.get(cache_key, "transcription")
            if cached:
                logger.info("✅ Using cached transcription")
                return self._enrich_with_speakers(cached, speaker_segments)
        
        logger.info("Starting audio transcription...")
        
        # Load model
        model = whisperx.load_model(
            self.settings.WHISPER_MODEL,
            self.device,
            compute_type=self.compute_type
        )
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Truncate if needed
        if duration_limit_seconds:
            sample_rate = 16000
            audio = audio[:int(duration_limit_seconds * sample_rate)]
        
        # Transcribe
        result = model.transcribe(
            audio,
            batch_size=self.settings.WHISPERX_BATCH_SIZE
        )
        
        logger.info(f"Detected language: {result['language']}")
        
        # Align for word-level timestamps
        logger.info("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        
        # Extract word-level data
        transcript = []
        for seg in result["segments"]:
            for word_data in seg.get("words", []):
                transcript.append({
                    "word": word_data.get("word", ""),
                    "start": word_data.get("start", 0.0),
                    "end": word_data.get("end", 0.0),
                    "score": word_data.get("score", None),
                })
        
        logger.info(f"✅ Transcribed {len(transcript)} words")
        
        # Cache results
        if self.cache and self.settings.ENABLE_CACHING:
            self.cache.set(cache_key, "transcription", transcript)
        
        # Enrich with speaker labels
        return self._enrich_with_speakers(transcript, speaker_segments)
    
    def _enrich_with_speakers(
        self,
        transcript: List[Dict[str, Any]],
        speaker_segments: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Add speaker labels to transcript words."""
        if not speaker_segments:
            return transcript
        
        logger.info("Enriching transcript with speaker labels...")
        
        enriched = []
        for word_data in transcript:
            word_time = word_data["start"]
            
            # Find matching speaker segment
            speaker = "unknown"
            for seg in speaker_segments:
                if seg["start"] <= word_time <= seg["end"]:
                    speaker = seg["speaker_id"]
                    break
            
            enriched_word = word_data.copy()
            enriched_word["speaker"] = speaker
            enriched.append(enriched_word)
        
        return enriched
