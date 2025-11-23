# pipeline.py
import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import time
import uuid

from google.cloud import storage

from config import get_settings
from db.models import EpisodeRepository
from rag.pgvector_store import PGVectorRAGStore
from processing.speaker_identification import SpeakerIdentifier
from processing.emotion import EmotionDetector
from processing.speaker_camera_mapping import SpeakerCameraMapper
from processing.llm_refiner import LLMRefiner
from edl.rules import RuleBasedEDLGenerator
from fcpxml.premiere_xml_generator import PremiereXMLGenerator

logger = logging.getLogger(__name__)

class VidPalAIPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.storage_client = storage.Client(project=self.settings.GOOGLE_CLOUD_PROJECT)
        
        # Components
        self.rag_store = PGVectorRAGStore() if self.settings.USE_RAG else None
        self.speaker_identifier = SpeakerIdentifier(cache=None) # Cache handled by DB mostly now
        self.emotion_detector = EmotionDetector()
        self.speaker_camera_mapper = SpeakerCameraMapper()
        self.llm_refiner = LLMRefiner(rag_store=self.rag_store) if self.settings.REFINE_WITH_LLM else None
        
        self.edl_generator = RuleBasedEDLGenerator(
            fps=self.settings.FRAME_RATE,
            min_shot_s=self.settings.MIN_SHOT_DURATION,
            wide_open_s=self.settings.WIDE_OPENING_DURATION,
            reaction_keywords=self.settings.REACTION_KEYWORDS
        )
        self.fcpxml_generator = PremiereXMLGenerator()

    def _acquire_media(self, uri: str, dest_path: Path):
        """
        Retrieves media from GCS or a local path.
        If uri starts with 'gs://', downloads from GCS.
        Otherwise, assumes it's a local path (e.g., /app/data/...) and copies it.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if uri.startswith("gs://"):
            # GCS Download Logic
            try:
                parts = uri[5:].split("/", 1)
                bucket_name = parts[0]
                blob_name = parts[1]
                
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                logger.info(f"☁️  Downloading GCS: {uri} -> {dest_path}")
                blob.download_to_filename(str(dest_path))
            except Exception as e:
                logger.error(f"GCS Download failed: {e}")
                raise
        else:
            # Local File Logic
            source_path = Path(uri)
            if not source_path.exists():
                raise FileNotFoundError(f"Local file not found: {source_path}")
            
            logger.info(f"📂 Copying Local: {source_path} -> {dest_path}")
            shutil.copy2(source_path, dest_path)

    def _upload_blob(self, local_path: Path, destination_blob_name: str) -> str:
        """Uploads a local file to the configured GCS bucket."""
        bucket = self.storage_client.bucket(self.settings.GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        
        logger.info(f"Uploading {local_path} to gs://{self.settings.GCS_BUCKET_NAME}/{destination_blob_name}...")
        blob.upload_from_filename(str(local_path))
        
        return f"gs://{self.settings.GCS_BUCKET_NAME}/{destination_blob_name}"

    def process_episode(
        self,
        episode_id: str,
        audio_gcs_uri: str,
        video_gcs_uris: Dict[str, str], # camera_id -> gs:// uri
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        EpisodeRepository.update_status(episode_id, "processing")
        
        # 1. Setup Local Workspace
        # We perform all processing in a unique temp directory to allow concurrency
        session_dir = self.settings.TEMP_DIR / episode_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        local_audio_path = session_dir / "master_audio.mp3"
        local_video_paths = {}
        
        try:
            # 2. Download Assets
            logger.info("PHASE 1: Asset Download")
            self._acquire_media(audio_gcs_uri, local_audio_path)
            
            for cam_id, uri in video_gcs_uris.items():
                local_path = session_dir / f"{cam_id}.mp4"
                self._acquire_media(uri, local_path)
                local_video_paths[cam_id] = local_path
                
                # Save to DB for record keeping
                EpisodeRepository.add_video_file(
                    episode_id=episode_id,
                    camera_id=cam_id,
                    file_path=uri, # Store the GCS URI in DB, not local path
                )
            
            EpisodeRepository.create_episode(
                episode_id=episode_id,
                title=title,
                audio_path=audio_gcs_uri
            )

            # 3. Processing (Parallelized)
            logger.info("PHASE 2: AI Analysis")
            
            # We run Emotion Detection locally (uses MediaPipe)
            # Speaker ID uploads to Cloud Speech (uses local audio -> convert -> GCS)
            
            # Emotion Detection
            all_emotion_events = []
            for cam_id, vid_path in local_video_paths.items():
                # We scan all cameras for reactions
                events = list(self.emotion_detector.detect_emotions(
                    video_path=vid_path,
                    camera_id=cam_id,
                    episode_id=episode_id
                ))
                all_emotion_events.extend(events)
            
            EpisodeRepository.save_reaction_events(episode_id, all_emotion_events)
            emotion_clusters = EmotionDetector.cluster_emotion_events(all_emotion_events)
            
            # Speaker Diarization (Cloud Speech)
            # Note: SpeakerIdentifier needs to be updated slightly to accept explicit GCS bucket if needed,
            # but currently it handles the upload internaly.
            speaker_segments, role_mapping, transcript, _ = self.speaker_identifier.identify_speakers(
                audio_path=str(local_audio_path),
                episode_id=episode_id
            )
            
            # RAG Ingestion
            if self.rag_store:
                self.rag_store.ingest_transcript_chunks(episode_id, transcript)

            # Speaker-Camera Mapping (Gemini)
            # This needs snippet extraction from local files
            role_camera_map = self.speaker_camera_mapper.map_roles_to_cameras(
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                video_paths=local_video_paths
            )
            
            # 4. EDL Generation
            logger.info("PHASE 3: Editing Decisions")
            edl_result = self.edl_generator.generate_edl(
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                role_camera_map=role_camera_map,
                transcript=transcript
            )
            cuts = edl_result["cuts"]

            # 5. LLM Refinement
            if self.llm_refiner:
                logger.info("PHASE 4: LLM Refinement")
                cuts = self.llm_refiner.refine_edl(
                    episode_id=episode_id,
                    base_edl=cuts,
                    transcript=transcript,
                    role_mapping=role_mapping
                )
            
            EpisodeRepository.save_edl_cuts(episode_id, cuts)
            
            # 6. Generate XML
            logger.info("PHASE 5: Final XML")
            xml_filename = f"{episode_id}_VidPal_Edit.xml"
            local_xml_path = session_dir / xml_filename
            
            self.fcpxml_generator.generate(
                cuts=cuts,
                video_paths=local_video_paths, # Needs local paths to calculate durations/frames
                output_path=local_xml_path,
                episode_id=episode_id,
                master_audio_path=local_audio_path,
                emotion_clusters=emotion_clusters
            )
            
            # Upload XML to GCS
            final_gcs_uri = self._upload_blob(local_xml_path, f"outputs/{episode_id}/{xml_filename}")
            
            EpisodeRepository.update_episode_metadata(episode_id, {
                "final_xml_uri": final_gcs_uri,
                "speakers_count": len(role_mapping),
                "processing_duration": time.time() - start_time
            })

            EpisodeRepository.update_status(episode_id, "completed")
            
            # Cleanup Local
            shutil.rmtree(session_dir)
            
            return {
                "status": "completed",
                "xml_uri": final_gcs_uri,
                "processing_time": time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            EpisodeRepository.update_status(episode_id, "failed")
            # Try cleanup even on fail
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise e
