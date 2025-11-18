# processing/emotion.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Generator
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import get_settings

logger = logging.getLogger(__name__)

# --- Blendshape Category Names ---
JAW_OPEN_NAME = "jawOpen"
MOUTH_SMILE_L_NAME = "mouthSmileLeft"
MOUTH_SMILE_R_NAME = "mouthSmileRight"

# --- MODIFICATION FOR TUNING ---
# Based on logs, smile scores are high and jaw scores are very low.
SMILE_THRESHOLD = 0.5
JAW_OPEN_THRESHOLD = 0.05  # Lowered from 0.3
# --- END MODIFICATION ---

# How many frames per second to analyze.
DEFAULT_SAMPLE_RATE = 2.0
FACE_DETECTION_CONFIDENCE = 0.5
# Low threshold to log potential values for tuning
VERBOSE_LOG_THRESHOLD = 0.05


class EmotionDetector:
    """
    Analyzes video files using MediaPipe Face Landmarker to detect
    emotional reactions (e.g., laughter) based on blendshapes.
    """
    
    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE):
        self.settings = get_settings()
        self.sample_rate = sample_rate
        self.frame_interval = 1.0 / self.sample_rate
        
        self.model_path = Path("models/face_landmarker.task")
        
        if not self.model_path.exists():
            logger.error(f"MediaPipe model not found at {self.model_path}")
            logger.error("Please download 'face_landmarker_v2_with_blendshapes.task' and place it in the 'models/' directory.")
            raise FileNotFoundError(str(self.model_path))

        try:
            # --- MODIFICATION FOR GPU ---
            delegate = python.BaseOptions.Delegate.CPU
            if self.settings.USE_GPU:
                logger.info("Attempting to set MediaPipe delegate to GPU...")
                try:
                    delegate = python.BaseOptions.Delegate.GPU
                    logger.info("MediaPipe delegate set to GPU.")
                except Exception as e:
                    logger.warning(f"Failed to set MediaPipe GPU delegate, falling back to CPU: {e}")
                    delegate = python.BaseOptions.Delegate.CPU
            else:
                 logger.info("MediaPipe delegate set to CPU (as per settings).")
                 
            base_options = python.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=delegate # <-- APPLIED DELEGATE
            )
            # --- END MODIFICATION ---

            self.detector_options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                num_faces=self.settings.MAX_SPEAKERS,
                min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            )
            
            logger.info(f"✅ MediaPipe EmotionDetector class initialized (options ready, model: {self.model_path.name})")
            
        except Exception as e:
            logger.error(f"Failed to create MediaPipe options: {e}")
            raise

    def detect_emotions(
        self,
        video_path: Path,
        camera_id: str,
        duration_limit_seconds: Optional[int] = None,
        verbose_log: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Analyzes a single video file and yields ReactionEvent dictionaries
        for each detected emotional event.
        """
        logger.info(f"Starting emotion detection for: {camera_id} (Sample rate: {self.sample_rate} FPS)")

        try:
            detector = vision.FaceLandmarker.create_from_options(self.detector_options)
            logger.info(f"Created new detector instance for {camera_id}")
        except Exception as e:
            logger.error(f"Failed to create detector instance for {camera_id}: {e}")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            detector.close()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            logger.warning(f"Could not get FPS for {camera_id}, using 30.0")
            fps = 30.0
            
        last_processed_time = -self.frame_interval
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if duration_limit_seconds and current_time > duration_limit_seconds:
                logger.info(f"Reached duration limit of {duration_limit_seconds}s for {camera_id}")
                break

            if (current_time - last_processed_time) < self.frame_interval:
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
                    verbose_log=verbose_log
                )
                
                if event:
                    yield event
                    
            except Exception as e:
                logger.warning(f"MediaPipe detection failed at {current_time:.2f}s for {camera_id}: {e}")

        detector.close()
        cap.release()
        logger.info(f"Finished emotion detection for: {camera_id}")

    def _parse_blendshapes_for_laughter(
        self,
        detection_result: vision.FaceLandmarkerResult,
        timestamp: float,
        camera_id: str,
        verbose_log: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Detects laughter using separate thresholds for smile and jaw.
        """
        if not detection_result.face_blendshapes:
            return None

        for face_blendshapes in detection_result.face_blendshapes:
            categories = {b.category_name: b.score for b in face_blendshapes}
            
            jaw_open = categories.get(JAW_OPEN_NAME, 0.0)
            smile_l = categories.get(MOUTH_SMILE_L_NAME, 0.0)
            smile_r = categories.get(MOUTH_SMILE_R_NAME, 0.0)
            
            if verbose_log and (jaw_open > VERBOSE_LOG_THRESHOLD or smile_l > VERBOSE_LOG_THRESHOLD or smile_r > VERBOSE_LOG_THRESHOLD):
                logger.info(
                    f"  [VERBOSE] ts: {timestamp:.2f}s | "
                    f"cam: {camera_id} | "
                    f"jaw: {jaw_open:.3f} | "
                    f"smile_L: {smile_l:.3f} | "
                    f"smile_R: {smile_r:.3f}"
                )

            # --- NEW LOGIC ---
            is_smiling = (smile_l > SMILE_THRESHOLD or smile_r > SMILE_THRESHOLD)
            is_jaw_open = (jaw_open > JAW_OPEN_THRESHOLD)
            
            is_laughing = is_smiling and is_jaw_open
            # --- END NEW LOGIC ---

            if is_laughing:
                return {
                    "timestamp": round(timestamp, 2),
                    "event": "laughter",
                    "camera": camera_id,
                    "score": round(max(smile_l, smile_r), 3),
                    "jaw_score": round(jaw_open, 3) # Added for more context
                }
                
        return None

def cluster_emotion_events(
events: List[Dict[str, Any]],
window_seconds: float = 30.0,
min_events: int = 4
) -> List[Dict[str, float]]:
"""
Clusters emotion events based on density (e.g., 4 events within 30s).
Returns a list of ranges: [{'start': 126.0, 'end': 142.5}, ...]
"""
    if not events:
        return []

    # Sort by timestamp
    sorted_events = sorted(events, key=lambda x: x['timestamp'])
    
    raw_clusters = []
    
    # 1. Identify dense windows
    # We iterate through every event and treat it as the potential 'start' of a window
    for i in range(len(sorted_events)):
        current_window_start = sorted_events[i]['timestamp']
        current_window_end = current_window_start + window_seconds
        
        # Find all events in this 30s window
        window_events = [
            e['timestamp'] 
            for e in sorted_events[i:] 
            if e['timestamp'] <= current_window_end
        ]
        
        # If threshold met, create a tentative cluster from First to Last event in window
        if len(window_events) >= min_events:
            raw_clusters.append({
                'start': window_events[0],
                'end': window_events[-1]
            })

    if not raw_clusters:
        return []

    # 2. Merge overlapping clusters
    merged = []
    if raw_clusters:
        # Sort by start time
        raw_clusters.sort(key=lambda x: x['start'])
        
        current_start = raw_clusters[0]['start']
        current_end = raw_clusters[0]['end']
        
        for i in range(1, len(raw_clusters)):
            next_start = raw_clusters[i]['start']
            next_end = raw_clusters[i]['end']
            
            # If next cluster starts before (or exactly when) current ends, merge them
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                # Push current and start new
                merged.append({'start': current_start, 'end': current_end})
                current_start = next_start
                current_end = next_end
        
        # Append final cluster
        merged.append({'start': current_start, 'end': current_end})
            
    return merged


# --- Standalone Test Runner (No changes needed here) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main_logger = logging.getLogger(__name__)
    main_logger.info("Running EmotionDetector in standalone test mode...")

    TEST_DURATION_LIMIT = 900  # 5 minutes
    
    # --- SET THIS TO TRUE TO SEE RAW BLENDSHAPE SCORES ---
    VERBOSE_TESTING = True 
    # ---

    try:
        detector = EmotionDetector(sample_rate=DEFAULT_SAMPLE_RATE)
        
        settings = get_settings()
        video_paths = settings.VIDEO_FILES
        
        if not video_paths:
            main_logger.error("No VIDEO_FILES defined in config.py or .env")
        
        main_logger.info(f"Using sample rate: {DEFAULT_SAMPLE_RATE} FPS")
        if TEST_DURATION_LIMIT:
            main_logger.info(f"Test duration limit: {TEST_DURATION_LIMIT} seconds per file")
        if VERBOSE_TESTING:
            main_logger.info(f"Verbose logging is ON. (Smile: {SMILE_THRESHOLD}, Jaw: {JAW_OPEN_THRESHOLD})")


        for camera_id, video_path in video_paths.items():
            if not video_path.exists():
                main_logger.warning(f"Video file not found, skipping: {video_path}")
                continue
            
            print(f"\n" + "="*60)
            print(f"🎬 PROCESSING: {camera_id} ({video_path.name})")
            print("="*60)
            
            event_count = 0
            
            try:
                for event in detector.detect_emotions(
                    video_path,
                    camera_id,
                    duration_limit_seconds=TEST_DURATION_LIMIT,
                    verbose_log=VERBOSE_TESTING
                ):
                    print(f"🎉 REACTION DETECTED: {event}")
                    event_count += 1
            except Exception as e:
                main_logger.error(f"Error during detection for {camera_id}: {e}")
            
            print(f"--- Finished {camera_id}. Found {event_count} events. ---")

        main_logger.info("✅ Standalone test complete.")

    except FileNotFoundError as e:
        main_logger.error(f"CRITICAL ERROR: Could not find model file.")
        main_logger.error("Please download 'face_landmarker_v2_with_blendshapes.task' from MediaPipe's website")
        main_logger.error("and place it in a 'models/' directory at the project root.")
    except Exception as e:
        main_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
