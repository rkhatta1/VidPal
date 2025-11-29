# processing/emotion.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Generator, List
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import get_settings
from db.models import EpisodeRepository
from utils.caching import Cache, compute_file_hash

logger = logging.getLogger(__name__)

# --- Blendshape Category Names ---
JAW_OPEN_NAME = "jawOpen"
MOUTH_SMILE_L_NAME = "mouthSmileLeft"
MOUTH_SMILE_R_NAME = "mouthSmileRight"

# --- TUNED THRESHOLDS ---
SMILE_THRESHOLD = 0.5
JAW_OPEN_THRESHOLD = 0.05
VERBOSE_LOG_THRESHOLD = 0.05

DEFAULT_SAMPLE_RATE = 2.0
FACE_DETECTION_CONFIDENCE = 0.5


class EmotionDetector:
    """
    Analyzes video files using MediaPipe Face Landmarker to detect
    emotional reactions (e.g., laughter) based on blendshapes.
    """
    
    def __init__(
        self, 
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        cache: Optional[Cache] = None
    ):
        self.settings = get_settings()
        self.sample_rate = sample_rate
        self.frame_interval = 1.0 / self.sample_rate
        self.cache = cache
        
        self.model_path = Path(self.settings.FACE_LANDMARKER_PATH).resolve()
        
        self.detector_options = None
        
        if not self.model_path.exists():
            logger.warning(f"MediaPipe model not found at {self.model_path}. Emotion detection will fail.")
            return

        try:
            delegate = python.BaseOptions.Delegate.CPU
            if self.settings.USE_GPU:
                try:
                    delegate = python.BaseOptions.Delegate.GPU
                except Exception as e:
                    logger.warning(f"Failed to set GPU delegate: {e}")

            base_options = python.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=delegate
            )

            self.detector_options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                num_faces=self.settings.MAX_SPEAKERS,
                min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            )
            logger.info(f"✅ MediaPipe EmotionDetector initialized")
            
        except Exception as e:
            logger.error(f"Failed to create MediaPipe options: {e}")

    def detect_emotions(
        self,
        video_path: Path,
        camera_id: str,
        episode_id: Optional[str] = None,
        duration_limit_seconds: Optional[int] = None, # Legacy, kept for compatibility
        start_time: float = 0.0,  # NEW
        end_time: Optional[float] = None, # NEW
        verbose_log: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Analyzes a video file and yields ReactionEvent dictionaries.
        Supports start/end time for sync-aware processing.
        """
        
        # 1. Check Database (Return everything, filtering happens in generator if needed, 
        # but usually DB store is "truth")
        if episode_id:
            db_events = EpisodeRepository.get_reaction_events(episode_id, camera_id)
            if db_events:
                logger.info(f"✅ Found {len(db_events)} emotion events in DB for {camera_id}")
                for event in db_events:
                    # Filter retrieved events by time window
                    ts = event['timestamp']
                    if ts >= start_time and (end_time is None or ts <= end_time):
                        yield event
                return

        # 2. Check Cache
        cache_key = None
        if self.cache and self.settings.ENABLE_CACHING:
            file_hash = compute_file_hash(video_path)
            # Include start/end in cache key to avoid returning partial results for full run
            # or use a broader strategy. For now, specific keys are safer.
            cache_key = f"emotion_{file_hash}_{start_time}_{end_time}_{self.sample_rate}"
            cached_events = self.cache.get(cache_key, "emotion_detection")
            
            if cached_events is not None:
                logger.info(f"✅ Found {len(cached_events)} emotion events in Cache")
                for event in cached_events:
                    yield event
                return

        # 3. Process Video
        if not self.detector_options:
            logger.error("Detector not initialized. Skipping.")
            return

        logger.info(f"Starting emotion detection for {camera_id} ({start_time}-{end_time})")

        try:
            detector = vision.FaceLandmarker.create_from_options(self.detector_options)
        except Exception as e:
            logger.error(f"Failed to create detector instance: {e}")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return

        last_processed_time = -self.frame_interval
        collected_events = [] 
        
        # Fast forward to start_time if possible
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            last_processed_time = start_time - self.frame_interval

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Stop if we exceeded end_time
            if end_time and current_time > end_time:
                break
            
            # Stop if legacy duration limit is hit
            if duration_limit_seconds and current_time > duration_limit_seconds:
                break

            # Skip frames based on sample rate
            if (current_time - last_processed_time) < self.frame_interval:
                continue
                
            # Skip frames before start_time (double check in case seek failed)
            if current_time < start_time:
                continue

            last_processed_time = current_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            try:
                timestamp_ms = int(current_time * 1000)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                
                event = self._parse_blendshapes_for_laughter(
                    detection_result,
                    current_time,
                    camera_id,
                    verbose_log
                )
                
                if event:
                    collected_events.append(event)
                    yield event
                    
            except Exception as e:
                # Ignore timestamp monotonic errors from mediapipe (happens with seeking)
                if "monotonically increasing" not in str(e):
                    logger.warning(f"MediaPipe detection error at {current_time:.1f}s: {e}")

        detector.close()
        cap.release()
        
        # 4. Save to Cache
        if self.cache and cache_key:
            self.cache.set(cache_key, "emotion_detection", collected_events)
            
        logger.info(f"Finished emotion detection for: {camera_id}")

    def _parse_blendshapes_for_laughter(
        self,
        detection_result: vision.FaceLandmarkerResult,
        timestamp: float,
        camera_id: str,
        verbose_log: bool = False
    ) -> Optional[Dict[str, Any]]:
        if not detection_result.face_blendshapes:
            return None

        for face_blendshapes in detection_result.face_blendshapes:
            categories = {b.category_name: b.score for b in face_blendshapes}
            
            jaw_open = categories.get(JAW_OPEN_NAME, 0.0)
            smile_l = categories.get(MOUTH_SMILE_L_NAME, 0.0)
            smile_r = categories.get(MOUTH_SMILE_R_NAME, 0.0)
            
            if verbose_log and (jaw_open > VERBOSE_LOG_THRESHOLD or smile_l > VERBOSE_LOG_THRESHOLD):
                logger.debug(f"{timestamp:.2f}s {camera_id} | Jaw: {jaw_open:.3f} | Smile: {max(smile_l, smile_r):.3f}")

            is_laughing = (
                (smile_l > SMILE_THRESHOLD or smile_r > SMILE_THRESHOLD) and
                (jaw_open > JAW_OPEN_THRESHOLD)
            )

            if is_laughing:
                return {
                    "timestamp": round(timestamp, 2),
                    "event": "laughter",
                    "camera": camera_id,
                    "score": round(max(smile_l, smile_r), 3)
                }
        return None

    @staticmethod
    def cluster_emotion_events(
        events: List[Dict[str, Any]],
        window_seconds: float = 30.0,
        min_events: int = 4
    ) -> List[Dict[str, float]]:
        """Identifies clusters of emotion events."""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        clusters = []
        
        for i in range(len(sorted_events)):
            window_start = sorted_events[i]['timestamp']
            window_end = window_start + window_seconds
            
            current_window_events = [
                e for e in sorted_events[i:] 
                if e['timestamp'] <= window_end
            ]
            
            if len(current_window_events) >= min_events:
                cluster_range = {
                    'start': current_window_events[0]['timestamp'],
                    'end': current_window_events[-1]['timestamp']
                }
                clusters.append(cluster_range)

        if not clusters:
            return []

        # Merge
        merged_clusters = []
        clusters.sort(key=lambda x: x['start'])
        
        current_start = clusters[0]['start']
        current_end = clusters[0]['end']
        
        for i in range(1, len(clusters)):
            next_start = clusters[i]['start']
            next_end = clusters[i]['end']
            
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged_clusters.append({'start': current_start, 'end': current_end})
                current_start = next_start
                current_end = next_end
                
        merged_clusters.append({'start': current_start, 'end': current_end})
        return merged_clusters
