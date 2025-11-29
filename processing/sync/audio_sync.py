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
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = 300
MIN_CONFIDENCE_THRESHOLD = 0.5
MAX_GAP_SECONDS = 300.0


class AudioSyncModule:
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
        max_gap_seconds: float = MAX_GAP_SECONDS,
        min_overlap_seconds: float = 5.0,
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
        logger.info(f"🎵 Starting sync. Master: {master_id}. Slaves: {list(slave_files.keys())}")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare Master
        master_audio_path = work_dir / "master_audio.wav"
        master_audio = self._extract_audio(master_path, master_audio_path)
        master_duration = float(len(master_audio) / self.sample_rate)
        
        file_offsets: Dict[str, FileOffset] = {}
        
        # Master is always 0
        file_offsets[master_id] = FileOffset(
            file_id=master_id,
            file_path=str(master_path),
            original_duration=master_duration,
            offset_seconds=0.0,
            global_in_point=0.0,
            global_out_point=master_duration,
            sync_confidence=1.0,
        )
        
        # 2. Process Slaves
        for file_id, file_path in slave_files.items():
            slave_audio_path = work_dir / f"{file_id}_audio.wav"
            slave_audio = self._extract_audio(file_path, slave_audio_path)
            slave_duration = float(len(slave_audio) / self.sample_rate)
            
            # Compute Offset
            offset_seconds, confidence = self._compute_offset(master_audio, slave_audio)
            
            logger.info(f"   🔎 {file_id}: offset={offset_seconds:.3f}s, confidence={confidence:.3f}")

            if confidence < self.confidence_threshold:
                raise LowConfidenceError(file_id, confidence, self.confidence_threshold)
                
            if abs(offset_seconds) > self.max_gap_seconds:
                raise ExcessiveGapError(abs(offset_seconds), self.max_gap_seconds)
            
            # 3. Calculate Global Timeline
            global_in = offset_seconds
            global_out = offset_seconds + slave_duration
            
            # Explicitly cast to Python native floats for DB compatibility
            file_offsets[file_id] = FileOffset(
                file_id=file_id,
                file_path=str(file_path),
                original_duration=float(slave_duration),
                offset_seconds=float(offset_seconds),
                global_in_point=float(global_in),
                global_out_point=float(global_out),
                sync_confidence=float(confidence),
            )

        return self._build_sync_result(master_id, file_offsets)

    def _extract_audio(self, input_path: Path, output_path: Path) -> np.ndarray:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn", "-ac", "1",
            "-ar", str(self.sample_rate),
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            str(output_path),
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            rate, data = wavfile.read(output_path)
            
            if len(data) == 0:
                raise AudioExtractionError(str(input_path), "Empty audio file")
                
            # Convert to float [-1, 1]
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            
            # Robust Normalization (max-abs)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
            
            return data
            
        except Exception as e:
            raise AudioExtractionError(str(input_path), str(e))

    def _compute_offset(
        self, 
        master_audio: np.ndarray, 
        slave_audio: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates time lag.
        Returns (offset_seconds, confidence)
        """
        chunk_len = int(self.chunk_seconds * self.sample_rate)
        n1 = len(master_audio)
        n2 = len(slave_audio)
        
        process_len = min(n1, n2, chunk_len)
        if process_len < self.sample_rate:
             raise SyncError("Audio too short for correlation")
             
        data1 = master_audio[:process_len]
        data2 = slave_audio[:process_len]
        
        # Correlation
        correlation = signal.correlate(data1, data2, mode='full', method='fft')
        
        lag_index = np.argmax(correlation)
        peak_val = correlation[lag_index]
        
        # PMR Confidence
        abs_corr = np.abs(correlation)
        median_val = np.median(abs_corr)
        
        if median_val == 0:
            confidence = 1.0 if peak_val > 0 else 0.0
        else:
            pmr = peak_val / median_val
            confidence = min(pmr / 6.0, 1.0)
            
        # Lag Calculation
        zero_lag_idx = len(data2) - 1
        lag_samples = lag_index - zero_lag_idx
        
        offset_seconds = lag_samples / self.sample_rate
        
        return float(offset_seconds), float(confidence)

    def _build_sync_result(self, master_id, file_offsets):
        all_in = [f.global_in_point for f in file_offsets.values()]
        all_out = [f.global_out_point for f in file_offsets.values()]
        
        g_start = min(all_in)
        g_end = max(all_out)
        
        # Normalize so min start is 0
        if g_start < 0:
            shift = abs(g_start)
            logger.info(f"   🔄 Shifting global timeline by +{shift:.2f}s")
            for f in file_offsets.values():
                f.global_in_point += shift
                f.global_out_point += shift
            g_start = 0.0
            g_end += shift

        c_start = max([f.global_in_point for f in file_offsets.values()])
        c_end = min([f.global_out_point for f in file_offsets.values()])
        
        return SyncResult(
            master_file_id=master_id,
            file_offsets=file_offsets,
            # Explicit float conversion
            global_start=float(g_start),
            global_end=float(g_end),
            common_start=float(c_start),
            common_end=float(c_end),
            # Explicit boolean conversion for DB
            has_full_overlap=bool(c_end > c_start)
        )
