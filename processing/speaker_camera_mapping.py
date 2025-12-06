# processing/speaker_camera_mapping.py
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import subprocess
import os
import uuid
from google import genai
from google.genai import types
from google.cloud import storage
from config import get_settings
# Import SyncResult for type hinting (needs to be available in path)
# If circular import issues arise, use 'Any' or string forward ref.
from processing.sync.models import SyncResult 

logger = logging.getLogger(__name__)


class SpeakerCameraMapper:
    """
    Uses Gemini VLM to map speaker IDs to camera IDs by analyzing
    video snippets. Sync-aware.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
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
        
        self.storage_client = storage.Client(
            project=self.settings.GOOGLE_CLOUD_PROJECT
        )
            
        logger.info(f"✅ Speaker Camera Mapper initialized")
    
    def _extract_snippet(
        self,
        video_path: Path,
        timestamp: float,
        duration: float = 3.0
    ) -> Optional[str]:
        """
        Extract a video snippet around a LOCAL timestamp using ffmpeg.
        """
        # Ensure we don't seek before 0
        start_time = max(0.0, timestamp - (duration / 2.0))
        
        temp_snippet = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_snippet_path = temp_snippet.name
        temp_snippet.close()
        
        cmd = [
            'ffmpeg',
            '-ss', f"{start_time:.3f}", # Fast seek before input
            '-i', str(video_path),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-an', # Drop audio to save bandwidth/processing
            '-y',
            '-loglevel', 'error',
            temp_snippet_path
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return temp_snippet_path
        
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg snippet extraction failed: {e.stderr}")
            Path(temp_snippet_path).unlink(missing_ok=True)
            return None


    def build_manual_snippet_set(
        self,
        episode_id: str,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        video_paths: Dict[str, Path],
        sync_result: Optional[SyncResult] = None,
        snippet_duration: float = 3.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build per-speaker, per-camera preview clips for human mapping.

        Returns a JSON-serializable structure:

        {
          "speaker_00": {
            "speaker_id": "SPEAKER_01",
            "global_timestamp": 123.45,
            "snippets": {
              "cam_wide":  { "gcs_uri": "...", "local_timestamp": 120.0 },
              "cam_host":  { "gcs_uri": "...", "local_timestamp": 121.0 }
            }
          },
          ...
        }
        """
        logger.info("🎬 Building manual speaker→camera snippet set...")
        role_to_speaker_id_map = {role: speaker_id for speaker_id, role in role_mapping.items()}
        result: Dict[str, Dict[str, Any]] = {}

        for role, speaker_id in role_to_speaker_id_map.items():
            # Pick the longest segment for this speaker
            segments = [s for s in speaker_segments if s["speaker_id"] == speaker_id]
            if not segments:
                logger.warning(f"No segments found for {speaker_id} ({role}), skipping.")
                continue

            segments.sort(key=lambda s: (s["end"] - s["start"]), reverse=True)
            chosen = segments[0]
            global_mid = (chosen["start"] + chosen["end"]) / 2.0
            logger.info(f"Preparing snippets for {role} ({speaker_id}) at global {global_mid:.2f}s")

            role_entry: Dict[str, Any] = {
                "speaker_id": speaker_id,
                "global_timestamp": float(global_mid),
                "snippets": {}
            }

            for camera_id, video_path in video_paths.items():
                # Default: assume "global time == local time"
                local_ts = global_mid

                if sync_result is not None:
                    local = sync_result.get_source_timecode(camera_id, global_mid)
                    if local is None:
                        logger.info(
                            f"  ⏭️  {camera_id}: no footage at {global_mid:.2f}s, skipping snippet."
                        )
                        continue
                    local_ts = local

                snippet_path = self._extract_snippet(video_path, local_ts, duration=snippet_duration)
                if not snippet_path:
                    logger.warning(f"  ⚠️ Failed to extract snippet for {camera_id}")
                    continue

                gcs_uri = self._upload_to_gcs(snippet_path)
                # Remove local temp file; keep the GCS blob for UI
                try:
                    Path(snippet_path).unlink(missing_ok=True)
                except Exception:
                    pass

                if not gcs_uri:
                    logger.warning(f"  ⚠️ Failed to upload snippet for {camera_id}")
                    continue

                role_entry["snippets"][camera_id] = {
                    "gcs_uri": gcs_uri,
                    "local_timestamp": float(local_ts),
                }

            if not role_entry["snippets"]:
                logger.warning(f"No valid snippets generated for {role}, skipping.")
                continue

            result[role] = role_entry

        logger.info(f"✅ Built snippet set for {len(result)} roles (episode={episode_id})")
        return result

    def _upload_to_gcs(self, file_path: str) -> Optional[str]:
        bucket_name = self.settings.GCS_BUCKET_NAME
        try:
            bucket = self.storage_client.bucket(bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(bucket_name, location=self.settings.GOOGLE_CLOUD_LOCATION)
        except Exception as e:
            logger.error(f"Failed to access/create bucket: {e}")
            return None
        
        blob_name = f"vidpalai/speaker_mapping/{uuid.uuid4().hex}.mp4"
        blob = bucket.blob(blob_name)
        
        try:
            blob.upload_from_filename(file_path)
            return f"gs://{bucket_name}/{blob_name}"
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            return None

    def _cleanup_gcs_file(self, gcs_uri: str) -> None:
        try:
            if not gcs_uri.startswith("gs://"): return
            path_parts = gcs_uri[5:].split("/", 1)
            bucket = self.storage_client.bucket(path_parts[0])
            bucket.blob(path_parts[1]).delete()
        except Exception as e:
            logger.warning(f"Failed to cleanup GCS file: {e}")
    
    def map_roles_to_cameras(
        self,
        speaker_segments: List[Dict[str, Any]],
        role_mapping: Dict[str, str],
        video_paths: Dict[str, Path],
        sync_result: Optional[SyncResult] = None # NEW ARGUMENT
    ) -> Dict[str, str]:
        """
        Finds the correct camera for each speaker role.
        If sync_result is provided, translates global speaker timestamps
        to local video timestamps for extraction.
        """
        logger.info("Starting speaker-to-camera mapping...")
        
        role_to_speaker_id_map = {v: k for k, v in role_mapping.items()}
        final_role_camera_map = {}

        for role, speaker_id in role_to_speaker_id_map.items():
            # Find a good segment to analyze (longest)
            segments = [s for s in speaker_segments if s['speaker_id'] == speaker_id]
            segments.sort(key=lambda s: s['end'] - s['start'], reverse=True)
            
            if not segments:
                continue
                
            longest_segment = segments[0]
            # This is Global Time
            global_mid_timestamp = (longest_segment['start'] + longest_segment['end']) / 2.0
            
            logger.info(f"Analyzing {role} ({speaker_id}) at Global Time {global_mid_timestamp:.1f}s")
            
            snippets = {}
            gcs_uris = []
            temp_paths = []
            
            for camera_id, video_path in video_paths.items():
                # Calculate Local Time for this camera
                local_timestamp = global_mid_timestamp
                
                if sync_result:
                    offset = sync_result.get_offset(camera_id)
                    if offset:
                        # Convert Global to Local
                        # Global In Point = The global time where the video file starts (0s local)
                        # So: Local = Global - Global_In_Point
                        local_timestamp = global_mid_timestamp - offset.global_in_point
                        
                        # Check bounds
                        if local_timestamp < 0 or local_timestamp > offset.original_duration:
                            logger.info(f"Skipping {camera_id} for mapping: timestamp out of bounds.")
                            continue
                    else:
                        # Fallback if no offset found (shouldn't happen if sync passed)
                        pass

                snippet_path = self._extract_snippet(video_path, local_timestamp)
                
                if snippet_path:
                    temp_paths.append(Path(snippet_path))
                    gcs_uri = self._upload_to_gcs(snippet_path)
                    if gcs_uri:
                        snippets[camera_id] = types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4")
                        gcs_uris.append(gcs_uri)
            
            if not snippets:
                logger.warning(f"No snippets extracted for {role}, skipping.")
                for p in temp_paths: p.unlink(missing_ok=True)
                continue
                
            # Build Prompt
            camera_options = ", ".join(snippets.keys())
            prompt_parts = [
                f"You are a professional video editor. The audio track confirms that the speaker '{speaker_id}' is talking at this moment.",
                "Here are video clips from all available cameras:",
            ]
            for camera_id, part in snippets.items():
                prompt_parts.append(f"**{camera_id}**:")
                prompt_parts.append(part)
            
            prompt_parts.append(
                f"Analyze the lip movement and body language in all clips. Which camera ({camera_options}) is clearly focused on the person who is speaking?"
                "Respond with *only* the camera ID (e.g., 'cam_1') and nothing else."
            )
            
            try:
                response = self.client.models.generate_content(
                    model=self.settings.GEMINI_MODEL,
                    contents=prompt_parts
                )
                chosen_camera = response.text.strip().replace("'", "").replace('"', "")
                
                if chosen_camera in video_paths:
                    logger.info(f"✅ Mapped {role} -> {chosen_camera}")
                    final_role_camera_map[role] = chosen_camera
                else:
                    logger.warning(f"Gemini returned invalid camera: '{chosen_camera}'")
            except Exception as e:
                logger.error(f"Gemini VLM call failed: {e}")
            
            # Cleanup
            for uri in gcs_uris: self._cleanup_gcs_file(uri)
            for p in temp_paths: p.unlink(missing_ok=True)
                
        # Handle wide default if not mapped
        # We can default to the one named "cam_wide" or just the first one if not set
        if "cam_wide" in video_paths and "default_wide" not in final_role_camera_map:
            final_role_camera_map["default_wide"] = "cam_wide"
        
        return final_role_camera_map
