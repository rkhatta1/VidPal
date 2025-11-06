# processing/camera_speaker_mapper.py
import logging
from typing import Dict, List, Any
from pathlib import Path
import cv2
from PIL import Image
from google import genai
from config import get_settings

logger = logging.getLogger(__name__)

class CameraSpeakerMapper:
    """
    Automatically maps camera IDs to speaker IDs using Gemini.
    """

    def __init__(self):
        self.settings = get_settings()
        # Initialize Gemini client using ADC from llm_refiner.py
        if self.settings.GOOGLE_GENAI_USE_VERTEXAI:
            self.client = genai.Client(
                vertexai=True,
                project=self.settings.GOOGLE_CLOUD_PROJECT,
                location=self.settings.GOOGLE_CLOUD_LOCATION,
            )
        else:
            self.client = genai.Client(api_key=self.settings.GOOGLE_API_KEY)
        
        logger.info("✅ CameraSpeakerMapper initialized with Gemini")

    def map_speakers_to_cameras(
        self,
        video_paths: Dict[str, Path],
        speaker_segments: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Maps camera IDs to speaker IDs by analyzing video frames at speech moments.

        Args:
            video_paths: A dictionary mapping camera IDs to video file paths.
            speaker_segments: A list of speaker segments from diarization.

        Returns:
            A dictionary mapping camera IDs to speaker IDs (e.g., {'cam_a': 'SPEAKER_00'}).
        """
        logger.info("Starting automatic speaker-to-camera mapping...")
        camera_to_speaker_map = {}
        speaker_to_camera_map = {}

        # Find the first clear speaking segment for each speaker
        unique_speaker_ids = sorted(list(set(seg['speaker_id'] for seg in speaker_segments)))

        for speaker_id in unique_speaker_ids:
            # Find a good segment for this speaker (e.g., longer than 2 seconds)
            speech_segment = self._find_clear_speech_segment(speaker_segments, speaker_id)
            if not speech_segment:
                logger.warning(f"Could not find a clear speech segment for {speaker_id}")
                continue

            # Get the timestamp in the middle of the segment
            timestamp = speech_segment['start'] + (speech_segment['end'] - speech_segment['start']) / 2

            logger.info(f"Analyzing frames at {timestamp:.2f}s for {speaker_id}...")

            # Check each camera at this timestamp
            for camera_id, video_path in video_paths.items():
                if "wide" in camera_id.lower():
                    logger.info(f"Skipping wide-angle camera '{camera_id}' from mapping.")
                    continue

                if camera_id in camera_to_speaker_map:
                    continue # Already mapped

                frame = self._extract_frame(video_path, timestamp)
                if frame:
                    is_speaking = self._is_person_speaking(frame)
                    if is_speaking:
                        logger.info(f"  ✅ Found {speaker_id} on {camera_id}")
                        camera_to_speaker_map[camera_id] = speaker_id
                        speaker_to_camera_map[speaker_id] = camera_id
                        # Once a speaker is mapped, we don't need to check them on other cameras
                        break 
            
        logger.info(f"Finished mapping: {camera_to_speaker_map}")
        return camera_to_speaker_map

    def _find_clear_speech_segment(self, speaker_segments: List[Dict[str, Any]], speaker_id: str) -> Dict[str, Any]:
        """Finds a suitable speech segment for a given speaker."""
        for seg in speaker_segments:
            if seg['speaker_id'] == speaker_id and (seg['end'] - seg['start']) > 2.0:
                return seg
        return None

    def _extract_frame(self, video_path: Path, time_seconds: float) -> Image.Image:
        """Extracts a single frame from a video at a specific time."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video file: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(time_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.warning(f"Could not read frame at {time_seconds:.2f}s from {video_path}")
                return None

            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Error extracting frame from {video_path}: {e}")
            return None

    def _is_person_speaking(self, frame: Image.Image) -> bool:
        """
        Uses Gemini to determine if the person in the frame is speaking.
        """
        try:
            prompt = "A person is speaking at this exact moment. Based on their mouth movement and expression, is the person in this frame the one who is currently speaking? Answer with only YES or NO."
            response = self.client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, frame])
            
            # Extract the text and clean it
            answer = response.text.strip().upper()
            logger.debug(f"Gemini response: {answer}")
            
            return "YES" in answer
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return False
