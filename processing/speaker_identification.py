import logging
import requests
import uuid
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from google.cloud import storage
from config import get_settings
from db.models import EpisodeRepository
from utils.caching import Cache, compute_file_hash

logger = logging.getLogger(__name__)

class LocalSpeakerIdentifier:
    """Speaker identification using local WhisperX Microservice."""
    
    def __init__(self, cache: Optional[Cache] = None):
        self.settings = get_settings()
        self.cache = cache
        self.storage_client = storage.Client(project=self.settings.GOOGLE_CLOUD_PROJECT)
        logger.info(f"SpeakerIdentifier initialized. Service URL: {self.settings.WHISPER_SERVICE_URL}")
    
    def identify_speakers(
        self,
        audio_path: str,
        episode_id: str,
        duration_limit_seconds: Optional[int] = None, 
        expected_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], str]:
        
        # --- 1. RESOLVE SPEAKER COUNT ---
        # Logic: Use argument if provided -> else use Settings -> else use min/max defaults
        target_speakers = expected_speakers or self.settings.EXPECTED_SPEAKERS
        
        if target_speakers:
            final_min = target_speakers
            final_max = target_speakers
            logger.info(f"Configuring diarization for exactly {target_speakers} speakers.")
        else:
            final_min = min_speakers
            final_max = max_speakers
            logger.info(f"Configuring diarization for {final_min}-{final_max} speakers.")

        # --- 2. CHECK CACHE ---
        cache_key = None
        if self.cache and self.settings.ENABLE_CACHING:
            file_hash = compute_file_hash(audio_path)
            # Include speaker counts in key so changing settings invalidates cache
            cache_key = f"whisperx_{file_hash}_{final_min}_{final_max}"
            
            cached = self.cache.get(cache_key, "speaker_diarization")
            if cached:
                logger.info("✅ Using cached speaker diarization")

                segments = cached['segments']
                role_mapping = cached['role_mapping']
                transcript = cached['transcript']

                # Persist diarization to DB for this episode
                EpisodeRepository.save_speaker_segments(episode_id, segments)
                speaker_stats = self._calculate_speaker_stats(segments)
                EpisodeRepository.save_speakers(episode_id, role_mapping, speaker_stats)

                return segments, role_mapping, transcript, cache_key

        # --- 3. PREPARE AUDIO ---
        # If local, upload to GCS for the microservice to access
        audio_url = audio_path
        temp_gcs_blob = None

        if not audio_path.startswith("gs://"):
            bucket = self.storage_client.bucket(self.settings.GCS_BUCKET_NAME)
            blob_name = f"vidpalai/temp_whisper/{episode_id}/{uuid.uuid4().hex}.mp3"
            blob = bucket.blob(blob_name)
            logger.info(f"Uploading local audio to GCS: {blob_name}")
            blob.upload_from_filename(audio_path)
            audio_url = f"gs://{self.settings.GCS_BUCKET_NAME}/{blob_name}"
            temp_gcs_blob = blob

        # --- 4. CALL MICROSERVICE ---
        payload = {
            "audio_url": audio_url,
            "episode_id": episode_id,
            "min_speakers": final_min,
            "max_speakers": final_max,
            "hf_token": self.settings.HUGGINGFACE_TOKEN
        }

        try:
            logger.info("⏳ Calling WhisperX microservice...")
            response = requests.post(
                f"{self.settings.WHISPER_SERVICE_URL}/diarize",
                json=payload,
                timeout=3600 # 1 hour timeout
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            logger.error(f"WhisperX Service failed: {e}")
            if temp_gcs_blob:
                try: temp_gcs_blob.delete() 
                except: pass
            raise e

        # --- 5. PROCESS RESULTS ---
        raw_segments = result.get("segments", [])
        speaker_segments = []
        full_transcript = []
        
        for seg in raw_segments:
            speaker = seg.get("speaker", "SPEAKER_00")
            
            speaker_segments.append({
                'speaker_id': speaker,
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip(),
                'words': [w['word'] for w in seg.get('words', [])]
            })
            
            if 'words' in seg:
                for w in seg['words']:
                    full_transcript.append({
                        "word": w['word'],
                        "start": w['start'],
                        "end": w['end'],
                        "speaker": w.get("speaker", speaker),
                        "score": w.get('score', 0.0)
                    })

        role_mapping = self._create_role_mapping(speaker_segments)

        # Cache & Save
        if self.cache and self.settings.ENABLE_CACHING and cache_key:
            self.cache.set(cache_key, "speaker_diarization", {
                'segments': speaker_segments,
                'role_mapping': role_mapping,
                'transcript': full_transcript,
            })

        EpisodeRepository.save_speaker_segments(episode_id, speaker_segments)
        speaker_stats = self._calculate_speaker_stats(speaker_segments)
        EpisodeRepository.save_speakers(episode_id, role_mapping, speaker_stats)

        if temp_gcs_blob:
            try: temp_gcs_blob.delete()
            except: pass

        return speaker_segments, role_mapping, full_transcript, cache_key

    def _create_role_mapping(self, speaker_segments: List[Dict[str, Any]]) -> Dict[str, str]:
        """Map speaker IDs to abstract numerical roles by duration."""
        speaker_durations = defaultdict(float)
        for seg in speaker_segments:
            speaker_durations[seg["speaker_id"]] += seg["end"] - seg["start"]
        
        sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
        
        role_mapping = {}
        for i, (speaker_id, duration) in enumerate(sorted_speakers):
            role_name = f"speaker_{i:02d}"
            role_mapping[speaker_id] = role_name
        
        return role_mapping
    
    def _calculate_speaker_stats(self, speaker_segments):
        stats = defaultdict(lambda: {'total_duration': 0.0, 'segment_count': 0})
        for seg in speaker_segments:
            speaker_id = seg['speaker_id']
            duration = seg['end'] - seg['start']
            stats[speaker_id]['total_duration'] += duration
            stats[speaker_id]['segment_count'] += 1
        return dict(stats)
