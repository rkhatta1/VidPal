import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio_from_video(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 48000,
    channels: int = 2,
) -> Path:
    """
    Extract audio track from video file as high-quality audio.
    
    Args:
        video_path: Source video file
        output_path: Destination audio file (extension determines format)
        sample_rate: Output sample rate (default 48kHz for broadcast quality)
        channels: Number of audio channels (default stereo)
    
    Returns:
        Path to extracted audio file
    
    Raises:
        RuntimeError: If extraction fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine codec based on output extension
    ext = output_path.suffix.lower()
    codec_map = {
        '.mp3': ['libmp3lame', '-q:a', '2'],
        '.wav': ['pcm_s16le'],
        '.aac': ['aac', '-b:a', '192k'],
        '.m4a': ['aac', '-b:a', '192k'],
    }
    
    codec_args = codec_map.get(ext, ['libmp3lame', '-q:a', '2'])
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # No video
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-acodec", codec_args[0],
        *codec_args[1:],
        "-loglevel", "error",
        str(output_path),
    ]
    
    logger.info(f"🎵 Extracting audio: {video_path.name} → {output_path.name}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Audio extraction failed for {video_path}: {e.stderr}"
        )
    
    if not output_path.exists():
        raise RuntimeError(f"Audio extraction produced no output: {output_path}")
    
    logger.info(f"✅ Audio extracted: {output_path.name}")
    return output_path


def get_media_duration(file_path: Path) -> float:
    """Get duration of media file in seconds using ffprobe."""
    import json
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(file_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    return float(data["format"]["duration"])


def has_audio_stream(file_path: Path) -> bool:
    """Check if a media file contains an audio stream."""
    import json
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",
        str(file_path),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    return len(data.get("streams", [])) > 0
