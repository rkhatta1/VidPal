import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import time

from google.cloud import storage

from config import get_settings
from db.models import EpisodeRepository
from rag.pgvector_store import PGVectorRAGStore
from processing.speaker_identification import LocalSpeakerIdentifier
from processing.gcs_speaker_identifier import GcsSpeakerIdentifier
from processing.emotion import EmotionDetector
from processing.speaker_camera_mapping import SpeakerCameraMapper
from processing.llm_refiner import LLMRefiner
from processing.sync import AudioSyncModule, SyncResult, SyncError
from processing.sync.utils import extract_audio_from_video, has_audio_stream
from edl.rules import RuleBasedEDLGenerator
from fcpxml.premiere_xml_generator import PremiereXMLGenerator
from utils.caching import Cache

logger = logging.getLogger(__name__)


class VidPalAIPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.storage_client = storage.Client(project=self.settings.GOOGLE_CLOUD_PROJECT)
        self.cache = Cache(cache_dir=Path(self.settings.CACHE_DIR))
        
        # Components
        self.rag_store = PGVectorRAGStore() if self.settings.USE_RAG else None
        
        if self.settings.SPEAKER_IDENTIFICATION_PROVIDER == 'gcs':
            logger.info("Using GCS for speaker identification.")
            self.speaker_identifier = GcsSpeakerIdentifier(cache=self.cache)
        else:
            logger.info("Using local Whisper service for speaker identification.")
            self.speaker_identifier = LocalSpeakerIdentifier(cache=self.cache)

        self.emotion_detector = EmotionDetector(cache=self.cache)
        self.speaker_camera_mapper = SpeakerCameraMapper()
        self.llm_refiner = LLMRefiner(rag_store=self.rag_store) if self.settings.REFINE_WITH_LLM else None
        self.sync_module = AudioSyncModule()
        
        self.edl_generator = RuleBasedEDLGenerator(
            fps=self.settings.FRAME_RATE,
            min_shot_s=self.settings.MIN_SHOT_DURATION,
            wide_open_s=self.settings.WIDE_OPENING_DURATION,
            reaction_keywords=self.settings.REACTION_KEYWORDS
        )
        self.fcpxml_generator = PremiereXMLGenerator()

    def _acquire_media(self, uri: str, dest_path: Path):
        """Retrieves media from GCS or a local path."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if uri.startswith("gs://"):
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

    def _determine_master_file(
        self,
        audio_gcs_uri: Optional[str],
        video_gcs_uris: Dict[str, str],
        sync_options: Optional[Dict[str, Any]],
    ) -> str:
        """
        Determine which file should be the master/reference.
        
        Priority:
        1. Explicit master_file_id in sync_options
        2. "master_audio" if audio_url provided
        3. First video file
        """
        if sync_options and sync_options.get("master_file_id"):
            master_id = sync_options["master_file_id"]
            # Validate it exists
            if master_id != "master_audio" and master_id not in video_gcs_uris:
                raise ValueError(
                    f"master_file_id '{master_id}' not found in video_urls. "
                    f"Available: {list(video_gcs_uris.keys())}"
                )
            return master_id
        
        if audio_gcs_uri:
            return "master_audio"
        
        # Default to first video
        return list(video_gcs_uris.keys())[0]

    def _run_sync_phase(
        self,
        session_dir: Path,
        local_audio_path: Path,
        local_video_paths: Dict[str, Path],
        master_file_id: str,
    ) -> SyncResult:
        """Execute the audio synchronization phase."""
        logger.info("PHASE 0: Audio Synchronization")
        
        # Prepare master and slave files
        if master_file_id == "master_audio":
            master_path = local_audio_path
            slave_files = dict(local_video_paths)
        else:
            master_path = local_video_paths[master_file_id]
            slave_files = {
                k: v for k, v in local_video_paths.items()
                if k != master_file_id
            }
        
        sync_work_dir = session_dir / "sync_audio"
        sync_result = self.sync_module.sync_files(
            master_path=master_path,
            master_id=master_file_id,
            slave_files=slave_files,
            work_dir=sync_work_dir,
        )
        
        return sync_result

    def process_episode(
        self,
        episode_id: str,
        audio_gcs_uri: Optional[str],  # NOW OPTIONAL
        video_gcs_uris: Dict[str, str],
        title: Optional[str] = None,
        sync_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        
        Args:
            episode_id: Unique identifier for this episode
            audio_gcs_uri: GCS URI for master audio file (OPTIONAL - extracted from master video if not provided)
            video_gcs_uris: Dict mapping camera_id to GCS URI
            title: Episode title
            sync_options: Optional dict with:
                - enabled: bool (default True if multiple files)
                - master_file_id: str (default first video if no audio_url)
        """
        start_time = time.time()
        EpisodeRepository.update_status(episode_id, "processing")
        
        # Parse options
        sync_options = sync_options or {}
        enable_sync = sync_options.get("enabled", len(video_gcs_uris) > 1)
        
        # Determine master file
        master_file_id = self._determine_master_file(
            audio_gcs_uri, video_gcs_uris, sync_options
        )
        logger.info(f"📌 Master file: {master_file_id}")
        
        # Setup workspace
        session_dir = self.settings.TEMP_DIR / episode_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        local_video_paths: Dict[str, Path] = {}
        local_audio_path: Optional[Path] = None
        
        try:
            # === PHASE 1: Asset Download ===
            logger.info("PHASE 1: Asset Download")
            
            # Download all videos first
            for cam_id, uri in video_gcs_uris.items():
                ext = Path(uri).suffix or ".mp4"
                local_path = session_dir / f"{cam_id}{ext}"
                self._acquire_media(uri, local_path)
                local_video_paths[cam_id] = local_path
                
                EpisodeRepository.add_video_file(
                    episode_id=episode_id,
                    camera_id=cam_id,
                    file_path=uri,
                )
            
            # Handle audio: download or extract
            if audio_gcs_uri:
                # Separate audio file provided
                local_audio_path = session_dir / "master_audio.mp3"
                self._acquire_media(audio_gcs_uri, local_audio_path)
                logger.info(f"🎵 Using provided audio: {audio_gcs_uri}")
            else:
                # Extract audio from master video
                if master_file_id not in local_video_paths:
                    raise ValueError(
                        f"Cannot extract audio: master '{master_file_id}' not in video files"
                    )
                
                master_video_path = local_video_paths[master_file_id]
                
                # Verify master video has audio
                if not has_audio_stream(master_video_path):
                    raise ValueError(
                        f"Master video '{master_file_id}' has no audio stream. "
                        "Please provide a separate audio_url or choose a different master."
                    )
                
                local_audio_path = session_dir / "master_audio.mp3"
                extract_audio_from_video(master_video_path, local_audio_path)
                
                # Update audio_gcs_uri reference for DB
                audio_gcs_uri = f"extracted_from:{video_gcs_uris[master_file_id]}"
                logger.info(f"🎵 Extracted audio from: {master_file_id}")
            
            EpisodeRepository.create_episode(
                episode_id=episode_id,
                title=title,
                audio_path=audio_gcs_uri
            )

            # === PHASE 0: Sync (if enabled) ===
            sync_result: Optional[SyncResult] = None
            processing_start = 0.0
            processing_end: Optional[float] = None
            
            if enable_sync and len(local_video_paths) > 1:
                try:
                    sync_result = self._run_sync_phase(
                        session_dir=session_dir,
                        local_audio_path=local_audio_path,
                        local_video_paths=local_video_paths,
                        master_file_id=master_file_id,
                    )
                    
                    EpisodeRepository.save_sync_result(episode_id, sync_result)
                    
                    processing_start = sync_result.common_start
                    processing_end = sync_result.common_end
                    
                    logger.info(
                        f"📐 Processing range: [{processing_start:.2f}s, {processing_end:.2f}s] "
                        f"({sync_result.common_duration:.2f}s common overlap)"
                    )
                    
                except SyncError as e:
                    logger.error(f"Sync failed: {e}")
                    EpisodeRepository.update_status(episode_id, "failed")
                    raise
            elif len(local_video_paths) == 1:
                logger.info("📐 Single video file - skipping sync")
            
            # === PHASE 2: AI Analysis ===
            logger.info("PHASE 2: AI Analysis")
            
            # Emotion Detection
            all_emotion_events = []
            for cam_id, vid_path in local_video_paths.items():
                events = list(self.emotion_detector.detect_emotions(
                    video_path=vid_path,
                    camera_id=cam_id,
                    episode_id=episode_id,
                    start_time=processing_start,
                    end_time=processing_end,
                ))
                all_emotion_events.extend(events)
            
            EpisodeRepository.save_reaction_events(episode_id, all_emotion_events)
            emotion_clusters = EmotionDetector.cluster_emotion_events(all_emotion_events)
            
            # Speaker Diarization
            speaker_segments, role_mapping, transcript, _ = self.speaker_identifier.identify_speakers(
                audio_path=str(local_audio_path),
                episode_id=episode_id,
            )
            
            # Filter segments to processing range
            if sync_result:
                speaker_segments = [
                    seg for seg in speaker_segments
                    if seg['end'] > processing_start and seg['start'] < processing_end
                ]
                for seg in speaker_segments:
                    seg['start'] = max(seg['start'], processing_start)
                    seg['end'] = min(seg['end'], processing_end)
                
                transcript = [
                    word for word in transcript
                    if processing_start <= word['start'] <= processing_end
                ]
            
            # RAG Ingestion
            if self.rag_store:
                self.rag_store.ingest_transcript_chunks(episode_id, transcript)

            # Speaker-Camera Mapping
            role_camera_map = self.speaker_camera_mapper.map_roles_to_cameras(
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                video_paths=local_video_paths
            )
            
            # === PHASE 3: EDL Generation ===
            logger.info("PHASE 3: Editing Decisions")
            edl_result = self.edl_generator.generate_edl(
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                role_camera_map=role_camera_map,
                transcript=transcript,
                start_time=processing_start,
                end_time=processing_end,
                sync_result=sync_result,
            )
            cuts = edl_result["cuts"]

            # === PHASE 4: LLM Refinement ===
            if self.llm_refiner:
                logger.info("PHASE 4: LLM Refinement")
                cuts = self.llm_refiner.refine_edl(
                    episode_id=episode_id,
                    base_edl=cuts,
                    transcript=transcript,
                    role_mapping=role_mapping
                )
            
            EpisodeRepository.save_edl_cuts(episode_id, cuts)
            
            # === PHASE 5: XML Generation ===
            logger.info("PHASE 5: Final XML")
            xml_filename = f"{episode_id}_VidPal_Edit.xml"
            local_xml_path = session_dir / xml_filename
            
            self.fcpxml_generator.generate(
                cuts=cuts,
                video_paths=local_video_paths,
                output_path=local_xml_path,
                episode_id=episode_id,
                master_audio_path=local_audio_path,
                emotion_clusters=emotion_clusters,
                sync_result=sync_result,
            )
            
            # Upload XML
            final_gcs_uri = self._upload_blob(
                local_xml_path, f"outputs/{episode_id}/{xml_filename}"
            )
            
            # Update metadata
            metadata = {
                "final_xml_uri": final_gcs_uri,
                "speakers_count": len(role_mapping),
                "processing_duration": time.time() - start_time,
                "master_file_id": master_file_id,
                "audio_extracted": audio_gcs_uri.startswith("extracted_from:") if audio_gcs_uri else False,
            }
            if sync_result:
                metadata["sync"] = {
                    "common_start": sync_result.common_start,
                    "common_end": sync_result.common_end,
                    "common_duration": sync_result.common_duration,
                }
            
            EpisodeRepository.update_episode_metadata(episode_id, metadata)
            EpisodeRepository.update_status(episode_id, "completed")
            
            # Cleanup
            shutil.rmtree(session_dir)
            
            return {
                "status": "completed",
                "xml_uri": final_gcs_uri,
                "processing_time": time.time() - start_time,
                "sync_applied": sync_result is not None,
                "master_file_id": master_file_id,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            EpisodeRepository.update_status(episode_id, "failed")
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise e
