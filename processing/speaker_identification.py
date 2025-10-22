
# processing/speaker_identification.py
import logging
from typing import List, Dict, Any, Tuple, Optional
import whisperx
import torch
from collections import defaultdict
from config import get_settings
from db.models import EpisodeRepository
from utils.caching import Cache, compute_file_hash

logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    """Optimized speaker identification with WhisperX."""
    
    def __init__(self, cache: Optional[Cache] = None):
        self.settings = get_settings()
        self.cache = cache
        self.device = "cuda" if (torch.cuda.is_available() and self.settings.USE_GPU) else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"Speaker identifier initialized (device: {self.device})")
    
    def identify_speakers(
        self,
        audio_path: str,
        episode_id: str,
        duration_limit_seconds: Optional[int] = None,
        expected_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Perform speaker diarization and role mapping.
        
        Returns:
            Tuple of (speaker_segments, role_mapping)
        """
        # Check cache
        if self.cache and self.settings.ENABLE_CACHING:
            file_hash = compute_file_hash(audio_path)
            cache_key = f"{file_hash}_{duration_limit_seconds}_{expected_speakers}"
            
            cached = self.cache.get(cache_key, "speaker_diarization")
            if cached:
                logger.info("✅ Using cached speaker diarization")
                return cached['segments'], cached['role_mapping']
        
        logger.info(f"Starting speaker identification (device: {self.device})")
        
        # Load models
        logger.info("Loading WhisperX model...")
        model = whisperx.load_model(
            self.settings.WHISPER_MODEL,
            self.device,
            compute_type=self.compute_type
        )
        
        # Load and transcribe audio
        logger.info("Transcribing audio...")
        audio = whisperx.load_audio(audio_path)
        
        # Truncate if needed
        if duration_limit_seconds:
            sample_rate = 16000
            audio = audio[:int(duration_limit_seconds * sample_rate)]
        
        result = model.transcribe(
            audio,
            batch_size=self.settings.WHISPERX_BATCH_SIZE
        )
        
        logger.info(f"Detected language: {result['language']}")
        
        # Align
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
        
        # Diarization
        logger.info("Performing speaker diarization...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.settings.HUGGINGFACE_TOKEN,
                device=self.device
            )
            
            # Configure diarization parameters
            diarize_kwargs = {}
            if expected_speakers is not None:
                diarize_kwargs['num_speakers'] = expected_speakers
            else:
                diarize_kwargs['min_speakers'] = min_speakers
                diarize_kwargs['max_speakers'] = max_speakers
            
            diarize_segments = diarize_model(audio, **diarize_kwargs)
            assigned = whisperx.assign_word_speakers(diarize_segments, result)
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}, using transcription only")
            assigned = result
        
        # Extract speaker segments
        speaker_segments = []
        for seg in assigned.get("segments", []):
            speaker_id = seg.get("speaker", "SPEAKER_00")
            speaker_segments.append({
                "speaker_id": speaker_id,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
                "confidence": seg.get("confidence", None),
            })
        
        # Create role mapping
        role_mapping = self._create_role_mapping(speaker_segments)
        
        unique_speakers = len(set(s['speaker_id'] for s in speaker_segments))
        logger.info(f"✅ Identified {unique_speakers} speakers in {len(speaker_segments)} segments")
        
        # Cache results
        if self.cache and self.settings.ENABLE_CACHING:
            self.cache.set(cache_key, "speaker_diarization", {
                'segments': speaker_segments,
                'role_mapping': role_mapping,
            })
        
        # Save to database
        EpisodeRepository.save_speaker_segments(episode_id, speaker_segments)
        
        # Calculate stats for speakers
        speaker_stats = self._calculate_speaker_stats(speaker_segments)
        EpisodeRepository.save_speakers(episode_id, role_mapping, speaker_stats)
        
        return speaker_segments, role_mapping
    
    def _create_role_mapping(
        self,
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Map speaker IDs to roles based on talk time."""
        speaker_durations = defaultdict(float)
        
        for seg in speaker_segments:
            duration = seg["end"] - seg["start"]
            speaker_durations[seg["speaker_id"]] += duration
        
        # Sort by talk time
        sorted_speakers = sorted(
            speaker_durations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        role_mapping = {}
        if len(sorted_speakers) >= 1:
            role_mapping[sorted_speakers[0][0]] = "host"
            logger.info(f"Host: {sorted_speakers[0][0]} ({sorted_speakers[0][1]:.1f}s)")
        
        if len(sorted_speakers) >= 2:
            role_mapping[sorted_speakers[1][0]] = "guest"
            logger.info(f"Guest: {sorted_speakers[1][0]} ({sorted_speakers[1][1]:.1f}s)")
        
        # Additional speakers
        for i, (speaker_id, duration) in enumerate(sorted_speakers[2:], start=3):
            role_mapping[speaker_id] = f"speaker_{i}"
            logger.info(f"Speaker {i}: {speaker_id} ({duration:.1f}s)")
        
        return role_mapping
    
    def _calculate_speaker_stats(
        self,
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each speaker."""
        stats = defaultdict(lambda: {'total_duration': 0.0, 'segment_count': 0})
        
        for seg in speaker_segments:
            speaker_id = seg['speaker_id']
            duration = seg['end'] - seg['start']
            stats[speaker_id]['total_duration'] += duration
            stats[speaker_id]['segment_count'] += 1
        
        return dict(stats)
