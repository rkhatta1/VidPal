# processing/speaker_camera_mapping.py (Corrected for Vertex AI)
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import subprocess
import os
import time
import uuid
from google import genai
from google.genai import types
from google.cloud import storage  # NEW IMPORT
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
        
        # NEW: Initialize GCS Client
        self.storage_client = storage.Client(
            project=self.settings.GOOGLE_CLOUD_PROJECT
        )
            
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
    
    # NEW GCS UPLOAD METHOD (from speaker_identification.py)
    def _upload_to_gcs(self, file_path: str) -> Optional[str]:
        """
        Upload audio file to Google Cloud Storage.
        Returns:
            GCS URI (gs://bucket-name/path/to/file.mp4)
        """
        bucket_name = self.settings.GCS_BUCKET_NAME
        
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
            return None
        
        # Generate unique blob name
        blob_name = f"vidpalai/speaker_mapping/{uuid.uuid4().hex}.mp4"
        blob = bucket.blob(blob_name)
        
        # Upload file
        logger.info(f"Uploading snippet to GCS: gs://{bucket_name}/{blob_name}")
        try:
            blob.upload_from_filename(file_path)
            gcs_uri = f"gs://{bucket_name}/{blob_name}"
            logger.info(f"✅ Snippet uploaded to: {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return None

    # NEW GCS CLEANUP METHOD (from speaker_identification.py)
    def _cleanup_gcs_file(self, gcs_uri: str) -> None:
        """Delete temporary file from GCS."""
        try:
            if not gcs_uri.startswith("gs://"):
                return
            
            path_parts = gcs_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1]
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.delete()
            logger.info(f"Cleaned up GCS file: {gcs_uri}")
        except Exception as e:
            logger.warning(f"Failed to cleanup GCS file: {e}")
    
    def map_roles_to_cameras(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        video_paths: Dict[str, Path]
    ) -> Dict[str, str]:
        """
        Finds the correct camera for each speaker role.
        """
        logger.info("Starting speaker-to-camera mapping...")
        final_role_camera_map = {}
        
        unique_speaker_ids = set(seg['speaker_id'] for seg in speaker_segments)
        
        for speaker_id in unique_speaker_ids:
            role = role_mapping.get(speaker_id)
            if not role:
                logger.warning(f"No role found for {speaker_id}, skipping.")
                continue
            
            # ... (find longest segment logic is unchanged) ...
            segments = [s for s in speaker_segments if s['speaker_id'] == speaker_id]
            segments.sort(key=lambda s: s['end'] - s['start'], reverse=True)
            if not segments:
                logger.warning(f"No segments found for {speaker_id}, skipping.")
                continue
            longest_segment = segments[0]
            mid_timestamp = (longest_segment['start'] + longest_segment['end']) / 2.0
            
            logger.info(f"Analyzing {speaker_id} (role: {role}) at {mid_timestamp:.1f}s")
            
            # MODIFIED: Upload snippets to GCS
            snippets = {}
            gcs_uris = []
            temp_snippet_paths = []
            
            for camera_id, video_path in video_paths.items():
                snippet_path = self._extract_snippet(video_path, mid_timestamp)
                if snippet_path:
                    temp_snippet_paths.append(Path(snippet_path))
                    gcs_uri = self._upload_to_gcs(snippet_path)
                    if gcs_uri:
                        # Create a Gemini Part using the GCS URI
                        snippets[camera_id] = types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4")
                        gcs_uris.append(gcs_uri)
            
            if not snippets:
                logger.error("No snippets could be extracted or uploaded to GCS.")
                # Local cleanup just in case
                for p in temp_snippet_paths:
                    p.unlink(missing_ok=True)
                continue
                
            # Build the prompt
            camera_options = ", ".join(snippets.keys())
            prompt_parts = [
                f"You are a professional video editor. The audio track confirms that the speaker '{speaker_id}' (role: '{role}') is talking at this moment.",
                "Here are video clips from all available cameras:",
            ]
            
            for camera_id, part in snippets.items():
                prompt_parts.append(f"**{camera_id}**:")
                prompt_parts.append(part)
            
            prompt_parts.append(
                f"Analyze the lip movement and body language in all clips. Which camera ({camera_options}) is clearly focused on the person who is speaking?"
                "Respond with *only* the camera ID (e.g., 'cam_a') and nothing else."
            )
            
            # Call Gemini
            try:
                response = self.client.models.generate_content(
                    model=self.settings.GEMINI_MODEL,
                    contents=prompt_parts
                )
                chosen_camera = response.text.strip().replace("'", "").replace('"', "")
                
                if chosen_camera in video_paths:
                    logger.info(f"✅ Mapped {speaker_id} ({role}) -> {chosen_camera}")
                    final_role_camera_map[role] = chosen_camera
                else:
                    logger.warning(f"Gemini returned an invalid camera ID: '{chosen_camera}'")
                    
            except Exception as e:
                logger.error(f"Gemini VLM call failed for {speaker_id}: {e}")
            
            # Cleanup GCS files
            for uri in gcs_uris:
                self._cleanup_gcs_file(uri)
            
            # Cleanup local snippets
            for p in temp_snippet_paths:
                p.unlink(missing_ok=True)
                
        if "cam_wide" in video_paths:
            final_role_camera_map["default_wide"] = "cam_wide"
        
        logger.info(f"Final camera mapping: {final_role_camera_map}")
        return final_role_camera_map
