import logging
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy import signal
from scipy.io import wavfile

from .models import FileOffset, SyncResult
from .exceptions import (
    SyncError,
    NoOverlapError,
    ExcessiveGapError,
    LowConfidenceError,
    AudioExtractionError,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLE_RATE = 8000  # Downsample for faster correlation
DEFAULT_CHUNK_SECONDS = 60  # Use first N seconds for correlation
MIN_CONFIDENCE_THRESHOLD = 0.15  # Below this, sync is unreliable
MAX_GAP_SECONDS = 120.0  # 2 minutes max gap between files
MIN_OVERLAP_SECONDS = 5.0  # Minimum required overlap


class AudioSyncModule:
    """
    Synchronizes multiple video/audio files using audio cross-correlation.
    
    MVP Implementation:
    - No drift compensation
    - Processes common_range only
    - Throws error if gap > 2 minutes
    """
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
        max_gap_seconds: float = MAX_GAP_SECONDS,
        min_overlap_seconds: float = MIN_OVERLAP_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.confidence_threshold = confidence_threshold
        self.max_gap_seconds = max_gap_seconds
        self.min_overlap_seconds = min_overlap_seconds
    
    def sync_files(
        self,
        master_path: Path,
        master_id: str,
        slave_files: Dict[str, Path],
        work_dir: Path,
    ) -> SyncResult:
        """
        Synchronize all slave files to the master using audio cross-correlation.
        
        Args:
            master_path: Path to master video/audio file
            master_id: Identifier for master file (e.g., "cam1" or "master_audio")
            slave_files: Dict mapping file_id to file path
            work_dir: Temporary directory for extracted audio
        
        Returns:
            SyncResult with all timing information
        
        Raises:
            ExcessiveGapError: If any file has offset > max_gap_seconds
            NoOverlapError: If files have no common overlap
            LowConfidenceError: If sync confidence is below threshold
        """
        logger.info(f"🎵 Starting audio sync with master: {master_id}")
        logger.info(f"   Slave files: {list(slave_files.keys())}")
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract audio from master
        master_audio_path = work_dir / "master_audio.wav"
        master_audio = self._extract_audio(master_path, master_audio_path)
        master_duration = len(master_audio) / self.sample_rate
        
        logger.info(f"   Master duration: {master_duration:.2f}s")
        
        # Step 2: Process each slave file
        file_offsets: Dict[str, FileOffset] = {}
        
        # Add master (always at offset 0)
        file_offsets[master_id] = FileOffset(
            file_id=master_id,
            file_path=str(master_path),
            original_duration=master_duration,
            offset_seconds=0.0,
            global_in_point=0.0,
            global_out_point=master_duration,
            sync_confidence=1.0,
        )
        
        # Process slaves
        for file_id, file_path in slave_files.items():
            logger.info(f"   Processing: {file_id}")
            
            slave_audio_path = work_dir / f"{file_id}_audio.wav"
            slave_audio = self._extract_audio(file_path, slave_audio_path)
            slave_duration = len(slave_audio) / self.sample_rate
            
            # Compute offset
            offset_seconds, confidence = self._compute_offset(
                master_audio, slave_audio
            )
            
            # Validate confidence
            if confidence < self.confidence_threshold:
                raise LowConfidenceError(file_id, confidence, self.confidence_threshold)
            
            # Validate gap
            if abs(offset_seconds) > self.max_gap_seconds:
                raise ExcessiveGapError(abs(offset_seconds), self.max_gap_seconds)
            
            # Calculate global positions
            global_in = offset_seconds
            global_out = offset_seconds + slave_duration
            
            file_offsets[file_id] = FileOffset(
                file_id=file_id,
                file_path=str(file_path),
                original_duration=slave_duration,
                offset_seconds=offset_seconds,
                global_in_point=global_in,
                global_out_point=global_out,
                sync_confidence=confidence,
            )
            
            logger.info(
                f"   📍 {file_id}: offset={offset_seconds:+.3f}s, "
                f"duration={slave_duration:.2f}s, confidence={confidence:.2f}"
            )
        
        # Step 3: Normalize timeline and calculate ranges
        sync_result = self._build_sync_result(master_id, file_offsets)
        
        # Step 4: Validate common range
        if sync_result.common_duration < self.min_overlap_seconds:
            raise NoOverlapError(
                f"Common overlap of {sync_result.common_duration:.2f}s is less than "
                f"minimum required {self.min_overlap_seconds:.2f}s"
            )
        
        logger.info(
            f"✅ Sync complete:\n"
            f"   Global range: [{sync_result.global_start:.2f}s, {sync_result.global_end:.2f}s]\n"
            f"   Common range: [{sync_result.common_start:.2f}s, {sync_result.common_end:.2f}s]\n"
            f"   Common duration: {sync_result.common_duration:.2f}s"
        )
        
        return sync_result
    
    def _extract_audio(self, input_path: Path, output_path: Path) -> np.ndarray:
        """
        Extract audio from video/audio file as mono WAV at target sample rate.
        
        Raises:
            AudioExtractionError: If ffmpeg fails
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",                      # No video
            "-ac", "1",                 # Mono
            "-ar", str(self.sample_rate),
            "-acodec", "pcm_s16le",     # 16-bit PCM
            "-loglevel", "error",
            str(output_path),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise AudioExtractionError(str(input_path), e.stderr)
        
        # Load WAV
        try:
            _, audio_data = wavfile.read(output_path)
        except Exception as e:
            raise AudioExtractionError(str(input_path), str(e))
        
        # Normalize to float32 [-1, 1]
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        return audio_float
    
    def _compute_offset(
        self,
        master_audio: np.ndarray,
        slave_audio: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute time offset of slave relative to master using cross-correlation.
        
        Returns:
            (offset_seconds, confidence)
            - Positive offset: slave starts AFTER master began
            - Negative offset: slave starts BEFORE master began
        """
        # Use chunk for correlation (performance + avoids drift issues)
        chunk_samples = int(self.chunk_seconds * self.sample_rate)
        
        master_chunk = master_audio[:chunk_samples]
        slave_chunk = slave_audio[:chunk_samples]
        
        # Handle case where files are shorter than chunk
        min_len = min(len(master_chunk), len(slave_chunk))
        if min_len < self.sample_rate:  # Less than 1 second
            raise SyncError("Audio too short for reliable sync (< 1 second)")
        
        master_chunk = master_chunk[:min_len]
        slave_chunk = slave_chunk[:min_len]
        
        # Normalize (zero mean, unit variance)
        master_norm = self._normalize_audio(master_chunk)
        slave_norm = self._normalize_audio(slave_chunk)
        
        # Cross-correlation using FFT (faster for large arrays)
        correlation = signal.correlate(master_norm, slave_norm, mode='full', method='fft')
        
        # Find peak
        peak_index = np.argmax(np.abs(correlation))
        peak_value = np.abs(correlation[peak_index])
        
        # Normalize confidence (0 to 1)
        max_possible = len(slave_chunk)
        confidence = min(peak_value / max_possible, 1.0)
        
        # Convert peak index to time offset
        # In 'full' mode, zero lag is at index (len(slave) - 1)
        zero_lag_index = len(slave_chunk) - 1
        lag_samples = peak_index - zero_lag_index
        
        # Positive lag = slave is delayed relative to master
        offset_seconds = lag_samples / self.sample_rate
        
        return offset_seconds, confidence
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to zero mean and unit variance."""
        mean = np.mean(audio)
        std = np.std(audio)
        if std < 1e-10:
            return audio - mean
        return (audio - mean) / std
    
    def _build_sync_result(
        self,
        master_id: str,
        file_offsets: Dict[str, FileOffset],
    ) -> SyncResult:
        """
        Build final SyncResult with normalized timeline.
        Shifts all timestamps so global_start is always 0.
        """
        # Find bounds
        all_in_points = [fo.global_in_point for fo in file_offsets.values()]
        all_out_points = [fo.global_out_point for fo in file_offsets.values()]
        
        raw_global_start = min(all_in_points)
        raw_global_end = max(all_out_points)
        
        # Common range: intersection of all files
        common_start = max(all_in_points)
        common_end = min(all_out_points)
        
        # Normalize timeline so earliest point is 0
        if raw_global_start < 0:
            shift = abs(raw_global_start)
            logger.info(f"   🔄 Normalizing timeline by +{shift:.2f}s")
            
            for fo in file_offsets.values():
                fo.global_in_point += shift
                fo.global_out_point += shift
            
            common_start += shift
            common_end += shift
            raw_global_end += shift
            raw_global_start = 0.0
        
        # Check if master is fully covered
        master_offset = file_offsets[master_id]
        has_full_overlap = (
            common_start <= master_offset.global_in_point and
            common_end >= master_offset.global_out_point
        )
        
        return SyncResult(
            master_file_id=master_id,
            file_offsets=file_offsets,
            global_start=raw_global_start,
            global_end=raw_global_end,
            common_start=common_start,
            common_end=common_end,
            has_full_overlap=has_full_overlap,
        )
