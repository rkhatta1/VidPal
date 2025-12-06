# processing/gcs_speaker_identifier.py
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import tempfile
import subprocess
import os
import uuid
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.api_core import client_options
from config import get_settings
from db.models import EpisodeRepository
from utils.caching import Cache, compute_file_hash

logger = logging.getLogger(__name__)


class GcsSpeakerIdentifier:
    """Speaker identification using Google Cloud Speech-to-Text for diarization."""
    
    def __init__(self, cache: Optional[Cache] = None):
        self.settings = get_settings()
        self.cache = cache
        self.device = "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Initialize Google Cloud Speech client with explicit quota project
        client_opts = client_options.ClientOptions(
            quota_project_id=self.settings.GOOGLE_CLOUD_PROJECT
        )
        
        self.speech_client = speech.SpeechClient(
            client_options=client_opts
        )
        
        # Initialize Google Cloud Storage client
        self.storage_client = storage.Client(
            project=self.settings.GOOGLE_CLOUD_PROJECT
        )
        
        logger.info(f"GcsSpeakerIdentifier initialized (device: {self.device})")
        logger.info(f"Using Google Cloud Speech-to-Text for diarization")
        logger.info(f"Project: {self.settings.GOOGLE_CLOUD_PROJECT}")
    
    def identify_speakers(
        self,
        audio_path: str,
        episode_id: str,
        duration_limit_seconds: Optional[int] = None,
        expected_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[Dict[str, Any]], str]:
        """
        Perform speaker diarization and role mapping.
        """
        # Initialize cache_key to None to prevent UnboundLocalError
        cache_key = None

        # Check cache
        if self.cache and self.settings.ENABLE_CACHING:
            file_hash = compute_file_hash(audio_path)
            expected_speakers_val = expected_speakers or self.settings.EXPECTED_SPEAKERS
            cache_key = f"{file_hash}_{duration_limit_seconds}_{expected_speakers_val}"
            
            cached = self.cache.get(cache_key, "speaker_diarization")
            if cached:
                logger.info("✅ Using cached speaker diarization")

                segments = cached['segments']
                role_mapping = cached['role_mapping']
                transcript = cached.get('transcript', [])

                # IMPORTANT: still persist diarization to DB for THIS episode
                EpisodeRepository.save_speaker_segments(episode_id, segments)
                speaker_stats = self._calculate_speaker_stats(segments)
                EpisodeRepository.save_speakers(episode_id, role_mapping, speaker_stats)

                return segments, role_mapping, transcript, cache_key
        
        logger.info(f"Starting speaker identification (device: {self.device})")
        
        # Convert audio to proper format for Google Cloud Speech
        processed_audio_path = self._prepare_audio_for_gcs(audio_path, duration_limit_seconds)
        
        # Upload to GCS and get URI
        gcs_uri = self._upload_to_gcs(processed_audio_path, episode_id)

        # Determine speaker counts
        final_expected_speakers = expected_speakers or self.settings.EXPECTED_SPEAKERS
        final_min_speakers = final_expected_speakers or min_speakers
        final_max_speakers = final_expected_speakers or max_speakers

        # Perform diarization with Google Cloud Speech-to-Text
        logger.info("Performing speaker diarization with Google Cloud Speech-to-Text...")
        diarization_result = self._diarize_with_google_speech(
            gcs_uri,
            final_expected_speakers,
            final_min_speakers,
            final_max_speakers,
        )
        
        # Extract speaker segments from diarization
        speaker_segments = self._extract_speaker_segments(diarization_result)
        
        full_transcript = self._extract_full_transcript(diarization_result)
        
        # Create role mapping (NOW USES NUMERICAL LABELS)
        role_mapping = self._create_role_mapping(speaker_segments)
        
        unique_speakers = len(set(s['speaker_id'] for s in speaker_segments))
        logger.info(f"✅ Identified {unique_speakers} speakers in {len(speaker_segments)} segments")
        
        # Cache results (only if cache is enabled and key was generated)
        if self.cache and self.settings.ENABLE_CACHING:
            # If we skipped the first block, we might need to generate the key now
            if not cache_key:
                file_hash = compute_file_hash(audio_path)
                expected_speakers_val = expected_speakers or self.settings.EXPECTED_SPEAKERS
                cache_key = f"{file_hash}_{duration_limit_seconds}_{expected_speakers_val}"

            self.cache.set(cache_key, "speaker_diarization", {
                'segments': speaker_segments,
                'role_mapping': role_mapping,
                'transcript': full_transcript,
            })
        
        # Save to database
        EpisodeRepository.save_speaker_segments(episode_id, speaker_segments)
        
        # Calculate stats for speakers
        speaker_stats = self._calculate_speaker_stats(speaker_segments)
        EpisodeRepository.save_speakers(episode_id, role_mapping, speaker_stats)
        
        # Cleanup temp file and GCS file
        if processed_audio_path != audio_path:
            Path(processed_audio_path).unlink(missing_ok=True)
        
        self._cleanup_gcs_file(gcs_uri)
        
        return speaker_segments, role_mapping, full_transcript, cache_key

    def _extract_full_transcript(
        self,
        response: speech.LongRunningRecognizeResponse,
    ) -> List[Dict[str, Any]]:
        """
        Extract word-level transcript from Google Cloud Speech response.
        """
        if not response.results or not response.results[-1].alternatives:
            logger.warning("No transcript alternatives found")
            return []
        
        alternative = response.results[-1].alternatives[0]
        words_info = alternative.words
        
        if not words_info:
            logger.warning("No words found in transcript")
            return []
        
        transcript = []
        for word_info in words_info:
            transcript.append({
                "word": word_info.word,
                "start": word_info.start_time.total_seconds(),
                "end": word_info.end_time.total_seconds(),
                "speaker": f"SPEAKER_{word_info.speaker_tag:02d}",
                # Google Speech API doesn't provide a word-level 'score' like Whisper
                "score": None, 
            })
        
        logger.info(f"Extracted {len(transcript)} words from Google Speech transcript")
        return transcript
    
    def _prepare_audio_for_gcs(
        self,
        audio_path: str,
        duration_limit_seconds: Optional[int] = None,
    ) -> str:
        """
        Prepare audio file for Google Cloud Speech-to-Text.
        Converts to LINEAR16 WAV format (mono, 16kHz).
        """
        logger.info("Preparing audio for Google Cloud Speech-to-Text...")
        
        # Create temp file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-acodec', 'pcm_s16le',  # LINEAR16 encoding
        ]
        
        # Add duration limit if specified
        if duration_limit_seconds:
            cmd.extend(['-t', str(duration_limit_seconds)])
        
        cmd.extend(['-y', temp_wav_path])
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Audio converted to LINEAR16 WAV: {temp_wav_path}")
            return temp_wav_path
        
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr}")
            raise
    
    def _upload_to_gcs(self, audio_path: str, episode_id: str) -> str:
        """
        Upload audio file to Google Cloud Storage.
        
        Returns:
            GCS URI (gs://bucket-name/path/to/file.wav)
        """
        bucket_name = self.settings.GCS_BUCKET_NAME
        
        # Create bucket if it doesn't exist
        try:
            bucket = self.storage_client.bucket(bucket_name)
            if not bucket.exists():
                logger.info(f"Creating GCS bucket: {bucket_name}")
                bucket = self.storage_client.create_bucket(
                    bucket_name,
                    location=self.settings.GOOGLE_CLOUD_LOCATION
                )
        except Exception as e:
            logger.error(f"Failed to access/create bucket: {e}")
            raise
        
        # Generate unique blob name
        blob_name = f"vidpalai/diarization/{episode_id}/{uuid.uuid4().hex}.wav"
        blob = bucket.blob(blob_name)
        
        # Upload file
        logger.info(f"Uploading audio to GCS: gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(audio_path)
        
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"✅ Audio uploaded to: {gcs_uri}")
        
        return gcs_uri
    
    def _cleanup_gcs_file(self, gcs_uri: str) -> None:
        """Delete temporary file from GCS."""
        try:
            # Parse GCS URI
            if not gcs_uri.startswith("gs://"):
                return
            
            path_parts = gcs_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1]
            
            # Delete blob
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            
            logger.info(f"Cleaned up GCS file: {gcs_uri}")
        except Exception as e:
            logger.warning(f"Failed to cleanup GCS file: {e}")
    
    def _diarize_with_google_speech(
        self,
        gcs_uri: str,
        expected_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int,
    ) -> speech.LongRunningRecognizeResponse:
        """
        Perform speaker diarization using Google Cloud Speech-to-Text.
        """
        # Configure diarization
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=expected_speakers or min_speakers,
            max_speaker_count=expected_speakers or max_speakers,
        )
        
        # Configure recognition
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=self.settings.SPEECH_LANGUAGE_CODE,
            diarization_config=diarization_config,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
            model=self.settings.SPEECH_MODEL,
        )
        
        # Create audio object with GCS URI
        audio = speech.RecognitionAudio(uri=gcs_uri)
        
        # Use long-running recognize
        logger.info("Starting Google Cloud Speech-to-Text recognition...")
        logger.info("This may take a few minutes for longer audio files...")
        
        operation = self.speech_client.long_running_recognize(
            config=config,
            audio=audio
        )
        
        logger.info("Waiting for operation to complete...")
        
        # FIXED: Increased timeout from 600 (10 mins) to 3600 (1 hour)
        response = operation.result(timeout=3600) 
        
        logger.info("✅ Google Cloud Speech-to-Text diarization complete")
        return response
    
    def _extract_speaker_segments(
        self,
        response: speech.LongRunningRecognizeResponse,
    ) -> List[Dict[str, Any]]:
        """
        Extract speaker segments from Google Cloud Speech response.
        """
        if not response.results:
            logger.warning("No results from Speech-to-Text")
            return []
        
        # The last result contains all words with speaker tags
        result = response.results[-1]
        
        if not result.alternatives:
            logger.warning("No alternatives in final result")
            return []
        
        alternative = result.alternatives[0]
        words_info = alternative.words
        
        if not words_info:
            logger.warning("No words with speaker tags")
            return []
        
        # Build speaker segments by grouping consecutive words from same speaker
        segments = []
        current_segment = None
        
        for word_info in words_info:
            speaker_tag = word_info.speaker_tag
            speaker_id = f"SPEAKER_{speaker_tag:02d}"
            
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()
            
            # Check if we should start a new segment
            if current_segment is None or current_segment['speaker_id'] != speaker_id:
                # Save previous segment
                if current_segment is not None:
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'speaker_id': speaker_id,
                    'start': start_time,
                    'end': end_time,
                    'text': word,
                    'words': [word],
                }
            else:
                # Continue current segment
                current_segment['end'] = end_time
                current_segment['text'] += ' ' + word
                current_segment['words'].append(word)
        
        # Add final segment
        if current_segment is not None:
            segments.append(current_segment)
        
        logger.info(f"Extracted {len(segments)} speaker segments")
        return segments
    
    def _create_role_mapping(
        self,
        speaker_segments: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        MODIFIED: Map speaker IDs to abstract numerical roles
        (e.g., 'speaker_00', 'speaker_01').
        """
        speaker_durations = defaultdict(float)
        for seg in speaker_segments:
            speaker_durations[seg["speaker_id"]] += seg["end"] - seg["start"]
        
        # Sort by talk time, so 'speaker_00' is consistently the most talkative
        sorted_speakers = sorted(
            speaker_durations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        role_mapping = {}
        for i, (speaker_id, duration) in enumerate(sorted_speakers):
            role_name = f"speaker_{i:02d}" # e.g., speaker_00, speaker_01
            role_mapping[speaker_id] = role_name
            logger.info(f"Mapped: {speaker_id} -> {role_name} ({duration:.1f}s)")
        
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

