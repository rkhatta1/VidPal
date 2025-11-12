# processing/speaker_camera_mapping.py
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import subprocess
import os
import time
from google import genai
from google.api_core import exceptions
from config import get_settings

logger = logging.getLogger(__name__)


class SpeakerCameraMapper:
    """
    Uses Gemini VLM to map speaker IDs to camera IDs by analyzing
    video snippets.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Gemini client
        if self.settings.GOOGLE_GENAI_USE_VERTEXAI:
            logger.info("Initializing Gemini (Video) with Vertex AI")
            self.client = genai.Client(
                vertexai=True,
                project=self.settings.GOOGLE_CLOUD_PROJECT,
                location=self.settings.GOOGLE_CLOUD_LOCATION,
            )
        else:
            logger.info("Initializing Gemini (Video) with API key")
            self.client = genai.Client(
                api_key=self.settings.GOOGLE_API_KEY,
            )
            
        # Use a model that supports video
        # self.model = self.client.models.get(self.settings.GEMINI_MODEL)
        logger.info(f"✅ Speaker Camera Mapper initialized (model: {self.settings.GEMINI_MODEL})")
    
    def _extract_snippet(
        self,
        video_path: Path,
        timestamp: float,
        duration: float = 3.0
    ) -> Optional[str]:
        """
        Extract a video snippet around a timestamp using ffmpeg.
        Returns the path to the temporary snippet file.
        """
        start_time = max(0, timestamp - (duration / 2.0))
        temp_snippet = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_snippet_path = temp_snippet.name
        temp_snippet.close()
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264', # Re-encode for compatibility
            '-c:a', 'aac',
            '-an', # No audio needed for VLM
            '-y',
            temp_snippet_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Extracted snippet: {temp_snippet_path}")
            return temp_snippet_path
        
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg snippet extraction failed: {e.stderr}")
            Path(temp_snippet_path).unlink(missing_ok=True)
            return None
    
    def _upload_to_gemini(self, file_path: str) -> Optional[genai.types.File]:
        """
        Uploads a video file to the Gemini API.
        """
        logger.info(f"Uploading {file_path} to Gemini API...")
        try:
            video_file = self.client.files.create(
                path=file_path,
                display_name=Path(file_path).name,
            )
            
            # Wait for processing
            state = video_file.state.name
            while state == "PROCESSING":
                logger.info("Waiting for Gemini file processing...")
                time.sleep(2)
                video_file = self.client.files.get(video_file.name)
                state = video_file.state.name
            
            if state == "FAILED":
                logger.error(f"Gemini file upload failed: {video_file.state.processing_failure_reason}")
                return None
            
            logger.info(f"✅ File uploaded and ready: {video_file.name}")
            return video_file
        
        except Exception as e:
            logger.error(f"Failed to upload video to Gemini: {e}")
            return None
    
    def map_roles_to_cameras(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        video_paths: Dict[str, Path]
    ) -> Dict[str, str]:
        """
        Finds the correct camera for each speaker role.
        
        Returns:
            A map of {role: camera_id}, e.g., {"host": "cam_a", "guest": "cam_b"}
        """
        logger.info("Starting speaker-to-camera mapping...")
        final_role_camera_map = {}
        
        # Get unique speaker IDs (e.g., 'SPEAKER_00', 'SPEAKER_01')
        unique_speaker_ids = set(seg['speaker_id'] for seg in speaker_segments)
        
        for speaker_id in unique_speaker_ids:
            # Find the role for this speaker (e.g., 'host')
            role = role_mapping.get(speaker_id)
            if not role:
                logger.warning(f"No role found for {speaker_id}, skipping.")
                continue
            
            # Find a good, long segment for this speaker
            segments = [s for s in speaker_segments if s['speaker_id'] == speaker_id]
            segments.sort(key=lambda s: s['end'] - s['start'], reverse=True)
            
            if not segments:
                logger.warning(f"No segments found for {speaker_id}, skipping.")
                continue
            
            longest_segment = segments[0]
            mid_timestamp = (longest_segment['start'] + longest_segment['end']) / 2.0
            
            logger.info(f"Analyzing {speaker_id} (role: {role}) at {mid_timestamp:.1f}s")
            
            # Extract snippets from all cameras at this timestamp
            snippets = {}
            gemini_files = {}
            temp_snippet_paths = []
            
            for camera_id, video_path in video_paths.items():
                snippet_path = self._extract_snippet(video_path, mid_timestamp)
                if snippet_path:
                    gemini_file = self._upload_to_gemini(snippet_path)
                    if gemini_file:
                        snippets[camera_id] = genai.Part.from_file(gemini_file)
                        gemini_files[camera_id] = gemini_file
                    temp_snippet_paths.append(Path(snippet_path))
            
            if not snippets:
                logger.error("No snippets could be extracted or uploaded.")
                continue
                
            # Build the prompt
            camera_options = ", ".join(snippets.keys())
            prompt_parts = [
                f"You are a professional video editor. The audio track confirms that the speaker '{speaker_id}' (role: '{role}') is talking at this moment.",
                "Here are video clips from all available cameras:",
            ]
            
            # Add all video parts
            for camera_id, part in snippets.items():
                prompt_parts.append(f"**{camera_id}**:")
                prompt_parts.append(part)
            
            prompt_parts.append(
                f"Analyze the lip movement and body language in all clips. Which camera ({camera_options}) is clearly focused on the person who is speaking?"
                "Respond with *only* the camera ID (e.g., 'cam_a') and nothing else."
            )
            
            # Call Gemini
            try:
                response = self.client.models.generate_content(model=self.settings.GEMINI_MODEL, contents=prompt_parts)
                chosen_camera = response.text.strip().replace("'", "").replace('"', "")
                
                # Validate response
                if chosen_camera in video_paths:
                    logger.info(f"✅ Mapped {speaker_id} ({role}) -> {chosen_camera}")
                    final_role_camera_map[role] = chosen_camera
                else:
                    logger.warning(f"Gemini returned an invalid camera ID: '{chosen_camera}'")
                    
            except Exception as e:
                logger.error(f"Gemini VLM call failed for {speaker_id}: {e}")
            
            # Cleanup uploaded files
            for file in gemini_files.values():
                try:
                    self.client.files.delete(file.name)
                except Exception:
                    pass # Don't block on cleanup failure
            
            # Cleanup local snippets
            for p in temp_snippet_paths:
                p.unlink(missing_ok=True)
                
        # Add a default 'wide' camera
        if "cam_wide" in video_paths:
            final_role_camera_map["default_wide"] = "cam_wide"
        
        logger.info(f"Final camera mapping: {final_role_camera_map}")
        return final_role_camera_map
