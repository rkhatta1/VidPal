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
            reaction_keywords=self.settings.REACTION_KEYWORDS,
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
        
        logger.info(
            f"Uploading {local_path} to gs://{self.settings.GCS_BUCKET_NAME}/{destination_blob_name}..."
        )
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
        audio_gcs_uri: Optional[str],
        video_gcs_uris: Dict[str, str],
        title: Optional[str] = None,
        sync_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main pipeline entry point.
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
                # Separate audio file provided by the user
                local_audio_path = session_dir / "master_audio.mp3"
                self._acquire_media(audio_gcs_uri, local_audio_path)
                logger.info(f"🎵 Using provided audio: {audio_gcs_uri}")
            else:
                # No separate master audio provided – extract audio from master video
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
                
                # Mark that this audio was derived from the master video (for DB / later logic)
                audio_gcs_uri = f"extracted_from:{video_gcs_uris[master_file_id]}"
                logger.info(f"🎵 Extracted audio from: {master_file_id}")
            
            # Persist episode with audio reference (explicit or derived)
            EpisodeRepository.create_episode(
                episode_id=episode_id,
                title=title,
                audio_path=audio_gcs_uri
            )

            # Decide what asset the XML should use as the "master audio" source:
            #   - If the user provided a real master audio file, use that.
            #   - If we extracted audio from the master video, use the *video file* itself
            #     as the audio source in the XML (Option B semantics).
            use_separate_master_audio = bool(
                audio_gcs_uri and not str(audio_gcs_uri).startswith("extracted_from:")
            )

            if use_separate_master_audio:
                xml_audio_source = local_audio_path
                logger.info("🎵 XML will reference explicit master audio file.")
            else:
                xml_audio_source = local_video_paths[master_file_id]
                logger.info(
                    "🎵 XML will reference master video '%s' as audio source "
                    "(no separate master audio provided).",
                    master_file_id,
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

            # Speaker-Camera Mapping (now sync-aware)
            role_camera_map = self.speaker_camera_mapper.map_roles_to_cameras(
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                video_paths=local_video_paths,
                sync_result=sync_result,
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
                    role_mapping=role_mapping,
                )
            
            EpisodeRepository.save_edl_cuts(episode_id, cuts)
            
            # === PHASE 5: XML Generation ===
            logger.info("PHASE 5: Final XML")
            xml_filename = f"{episode_id}_VidPal_Edit.xml"
            local_xml_path = session_dir / xml_filename
            
            # OPTION B:
            # If audio was extracted from a video, do NOT reference a standalone
            # master audio file in the XML. Let Premiere use the video audio.
            if audio_gcs_uri and audio_gcs_uri.startswith("extracted_from:"):
                xml_master_audio_path: Optional[Path] = None
            else:
                xml_master_audio_path = local_audio_path
            
            self.fcpxml_generator.generate(
                cuts=cuts,
                video_paths=local_video_paths,
                output_path=local_xml_path,
                episode_id=episode_id,
                master_audio_path=xml_audio_source,
                emotion_clusters=emotion_clusters,
                sync_result=sync_result,
            )
            
            # Upload XML
            final_gcs_uri = self._upload_blob(
                local_xml_path, f"outputs/{episode_id}/{xml_filename}"
            )
            
            # Update metadata
            metadata: Dict[str, Any] = {
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


    def prepare_for_manual_mapping(
        self,
        episode_id: str,
        audio_gcs_uri: Optional[str],
        video_gcs_uris: Dict[str, str],
        title: Optional[str] = None,
        sync_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Phase A for human-in-the-loop mapping.

        - Downloads media
        - Runs sync (if enabled)
        - Runs emotion detection
        - Runs diarization + transcript
        - Builds 3s preview clips per (speaker_role, camera_id) and uploads to GCS
        - Stores results in episodes.metadata["mapping_snippets"]
        - Sets processing_status = 'awaiting_mapping'
        """
        start_time = time.time()
        EpisodeRepository.update_status(episode_id, "processing")

        sync_options = sync_options or {}
        enable_sync = sync_options.get("enabled", len(video_gcs_uris) > 1)

        master_file_id = self._determine_master_file(
            audio_gcs_uri, video_gcs_uris, sync_options
        )
        logger.info(f"[MANUAL] 📌 Master file: {master_file_id}")

        # Separate temp dir to avoid colliding with the auto pipeline
        session_dir = self.settings.TEMP_DIR / f"{episode_id}_prepare"
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        local_video_paths: Dict[str, Path] = {}
        local_audio_path: Optional[Path] = None

        try:
            # === Download videos ===
            logger.info("[MANUAL] PHASE 1: Asset Download")
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

            # === Audio: provided or extracted from master video ===
            if audio_gcs_uri and not str(audio_gcs_uri).startswith("extracted_from:"):
                local_audio_path = session_dir / "master_audio.mp3"
                self._acquire_media(audio_gcs_uri, local_audio_path)
                logger.info(f"[MANUAL] 🎵 Using provided audio: {audio_gcs_uri}")
            else:
                if master_file_id not in local_video_paths:
                    raise ValueError(
                        f"Cannot extract audio: master '{master_file_id}' not in video files"
                    )

                master_video_path = local_video_paths[master_file_id]

                if not has_audio_stream(master_video_path):
                    raise ValueError(
                        f"Master video '{master_file_id}' has no audio stream. "
                        "Please provide a separate audio_url or choose a different master."
                    )

                local_audio_path = session_dir / "master_audio.mp3"
                extract_audio_from_video(master_video_path, local_audio_path)
                audio_gcs_uri = f"extracted_from:{video_gcs_uris[master_file_id]}"
                logger.info(f"[MANUAL] 🎵 Extracted audio from: {master_file_id}")

            # Update episode record with audio path, title, etc.
            EpisodeRepository.create_episode(
                episode_id=episode_id,
                title=title,
                audio_path=audio_gcs_uri,
            )

            # === Sync (if enabled) ===
            logger.info("[MANUAL] PHASE 0: Audio Synchronization")
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
                        f"[MANUAL] 📐 Processing range: "
                        f"[{processing_start:.2f}s, {processing_end:.2f}s]"
                    )
                except SyncError as e:
                    logger.error(f"[MANUAL] Sync failed: {e}")
                    EpisodeRepository.update_status(episode_id, "failed")
                    raise
            elif len(local_video_paths) == 1:
                logger.info("[MANUAL] Single video file - skipping sync")

            # === Emotion Detection (precompute; re-used via DB in Phase B) ===
            logger.info("[MANUAL] PHASE 2a: Emotion Detection")
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

            # === Speaker Diarization ===
            logger.info("[MANUAL] PHASE 2b: Speaker Identification")
            speaker_segments, role_mapping, transcript, _ = self.speaker_identifier.identify_speakers(
                audio_path=str(local_audio_path),
                episode_id=episode_id,
            )

            # Clamp to processing range if sync applied
            if sync_result:
                speaker_segments = [
                    seg for seg in speaker_segments
                    if seg["end"] > processing_start and seg["start"] < processing_end
                ]
                for seg in speaker_segments:
                    seg["start"] = max(seg["start"], processing_start)
                    seg["end"] = min(seg["end"], processing_end)

                transcript = [
                    w for w in transcript
                    if processing_start <= w["start"] <= (processing_end or w["start"])
                ]

            # Persist transcript + role_mapping so Phase B can reuse them
            EpisodeRepository.update_episode_metadata(episode_id, {
                "role_mapping": role_mapping,
                "transcript_words": transcript,
            })

            # === Build snippets for manual mapping ===
            logger.info("[MANUAL] PHASE 2c: Build mapping snippets")
            mapping_snippets = self.speaker_camera_mapper.build_manual_snippet_set(
                episode_id=episode_id,
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                video_paths=local_video_paths,
                sync_result=sync_result,
            )

            # Save metadata & status
            metadata: Dict[str, Any] = {
                "mapping_mode": "manual",
                "mapping_status": "awaiting_mapping",
                "mapping_snippets": mapping_snippets,
                "speakers_count": len(role_mapping),
                "master_file_id": master_file_id,
                "audio_extracted": bool(audio_gcs_uri and str(audio_gcs_uri).startswith("extracted_from:")),
                "prepare_processing_duration": time.time() - start_time,
            }
            if sync_result:
                metadata["sync"] = {
                    "common_start": sync_result.common_start,
                    "common_end": sync_result.common_end,
                    "common_duration": sync_result.common_duration,
                }

            EpisodeRepository.update_episode_metadata(episode_id, metadata)
            EpisodeRepository.update_status(episode_id, "awaiting_mapping")

            # Cleanup temp files
            shutil.rmtree(session_dir)

            logger.info(f"[MANUAL] ✅ Episode {episode_id} ready for manual mapping")
            return {
                "status": "awaiting_mapping",
                "episode_id": episode_id,
            }

        except Exception as e:
            logger.error(f"[MANUAL] prepare_for_manual_mapping failed: {e}", exc_info=True)
            EpisodeRepository.update_status(episode_id, "failed")
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise


    def finalize_with_manual_mapping(
        self,
        episode_id: str,
        role_camera_map: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Phase B for human-in-the-loop mapping.

        - Reloads media based on DB records
        - Rebuilds SyncResult from DB (no re-sync)
        - Reuses diarization + transcript stored in DB/metadata
        - Runs emotion clustering, RAG, EDL, LLM refinement
        - Generates final Premiere XML using the provided role_camera_map
        """
        start_time = time.time()
        EpisodeRepository.update_status(episode_id, "processing")

        episode = EpisodeRepository.get_episode(episode_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")

        metadata = episode.get("metadata") or {}

        # Persist mapping into metadata immediately
        EpisodeRepository.update_episode_metadata(episode_id, {
            "manual_role_camera_map": role_camera_map,
            "mapping_status": "processing",
        })

        # === Resolve video & audio sources ===
        video_rows = EpisodeRepository.get_video_files(episode_id)
        if not video_rows:
            raise ValueError(f"No video_files records for episode {episode_id}")

        video_gcs_uris: Dict[str, str] = {
            row["camera_id"]: row["file_path"] for row in video_rows
        }

        audio_ref = episode.get("audio_path")
        master_file_id = metadata.get("master_file_id")
        if not master_file_id:
            master_file_id = self._determine_master_file(audio_ref, video_gcs_uris, sync_options=None)

        # Ensure a default wide camera exists in the mapping
        if "default_wide" not in role_camera_map:
            if "cam_wide" in video_gcs_uris:
                role_camera_map["default_wide"] = "cam_wide"
            else:
                # Fallback: pick the first camera as wide
                role_camera_map["default_wide"] = next(iter(video_gcs_uris.keys()))

        # === Working directory ===
        session_dir = self.settings.TEMP_DIR / f"{episode_id}_final"
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        local_video_paths: Dict[str, Path] = {}
        local_audio_path: Optional[Path] = None

        try:
            # Download videos
            logger.info("[MANUAL] FINALIZE: Downloading media")
            for cam_id, uri in video_gcs_uris.items():
                ext = Path(uri).suffix or ".mp4"
                local_path = session_dir / f"{cam_id}{ext}"
                self._acquire_media(uri, local_path)
                local_video_paths[cam_id] = local_path

            # Decide which asset the XML should use as the master audio source.
            # If the user originally provided a real master audio file (audio_path not "extracted_from:"),
            # we reference that file. Otherwise, we reference the master VIDEO file's audio.
            use_separate_master_audio = bool(
                audio_ref and not str(audio_ref).startswith("extracted_from:")
            )

            if use_separate_master_audio:
                # We don't need to actually download the audio just for XML – the path/filename is enough
                # because Premiere will relink against WINDOWS_PROJECT_ROOT.
                xml_audio_source = Path(str(audio_ref))
                logger.info(
                    "[MANUAL] FINALIZE: Using explicit master audio '%s' as XML audio source",
                    audio_ref,
                )
            else:
                if master_file_id not in local_video_paths:
                    raise ValueError(
                        f"Cannot use master video audio in finalize: master '{master_file_id}' not in video files"
                    )
                xml_audio_source = local_video_paths[master_file_id]
                logger.info(
                    "[MANUAL] FINALIZE: Using master video '%s' as XML audio source "
                    "(no separate master audio provided)",
                    master_file_id,
                )

            # === Rebuild SyncResult (if existed) ===
            sync_dict = EpisodeRepository.get_sync_result(episode_id)
            sync_result: Optional[SyncResult] = None
            processing_start = 0.0
            processing_end: Optional[float] = None

            if sync_dict:
                sync_result = SyncResult.from_dict(sync_dict)
                # Patch file paths for local versions
                for file_id, offset in sync_result.file_offsets.items():
                    if file_id == "master_audio" and local_audio_path:
                        offset.file_path = str(local_audio_path)
                    elif file_id in local_video_paths:
                        offset.file_path = str(local_video_paths[file_id])

                processing_start = sync_result.common_start
                processing_end = sync_result.common_end
                logger.info(
                    f"[MANUAL] FINALIZE: Using stored sync range "
                    f"[{processing_start:.2f}s, {processing_end:.2f}s]"
                )

            # === Emotion events (reused or recomputed) ===
            logger.info("[MANUAL] FINALIZE: Emotion detection / reuse")
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

            # === Load diarization + transcript from DB / metadata ===
            logger.info("[MANUAL] FINALIZE: Loading diarization from DB")
            speaker_segments = EpisodeRepository.get_speaker_segments(episode_id)
            role_mapping = metadata.get("role_mapping") or EpisodeRepository.get_role_mapping(episode_id)
            transcript = metadata.get("transcript_words") or []

            logger.info(
                f"[MANUAL] FINALIZE: Loaded {len(speaker_segments)} speaker segments "
                f"and {len(transcript)} words for episode {episode_id}"
            )
            if not speaker_segments:
                logger.warning(
                "[MANUAL] FINALIZE: No speaker segments found; "
                "EDL will fall back to a single wide shot."
                )

            if sync_result:
                speaker_segments = [
                    seg for seg in speaker_segments
                    if seg["end"] > processing_start and seg["start"] < processing_end
                ]
                for seg in speaker_segments:
                    seg["start"] = max(seg["start"], processing_start)
                    seg["end"] = min(seg["end"], processing_end)

                transcript = [
                    w for w in transcript
                    if processing_start <= w["start"] <= (processing_end or w["start"])
                ]

            # === RAG ingestion (idempotent) ===
            if self.rag_store:
                self.rag_store.ingest_transcript_chunks(episode_id, transcript)

            # === EDL Generation with manual role_camera_map ===
            logger.info("[MANUAL] FINALIZE: Generating EDL with manual mapping")
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

            # === LLM Refinement (optional) ===
            if self.llm_refiner:
                logger.info("[MANUAL] FINALIZE: LLM refinement")
                cuts = self.llm_refiner.refine_edl(
                    episode_id=episode_id,
                    base_edl=cuts,
                    transcript=transcript,
                    role_mapping=role_mapping,
                )

            EpisodeRepository.save_edl_cuts(episode_id, cuts)

            # === XML Generation ===
            logger.info("[MANUAL] FINALIZE: Generating Premiere XML")
            xml_filename = f"{episode_id}_VidPal_Edit.xml"
            local_xml_path = session_dir / xml_filename

            self.fcpxml_generator.generate(
                cuts=cuts,
                video_paths=local_video_paths,
                output_path=local_xml_path,
                episode_id=episode_id,
                master_audio_path=xml_audio_source,
                emotion_clusters=emotion_clusters,
                sync_result=sync_result,
            )

            final_gcs_uri = self._upload_blob(
                local_xml_path,
                f"outputs/{episode_id}/{xml_filename}",
            )

            new_metadata: Dict[str, Any] = {
                "final_xml_uri": final_gcs_uri,
                "manual_role_camera_map": role_camera_map,
                "mapping_status": "completed",
                "finalize_processing_duration": time.time() - start_time,
                "master_file_id": master_file_id,
                "audio_extracted": bool(audio_ref and str(audio_ref).startswith("extracted_from:")),
            }
            if sync_result:
                new_metadata["sync"] = {
                    "common_start": sync_result.common_start,
                    "common_end": sync_result.common_end,
                    "common_duration": sync_result.common_duration,
                }

            EpisodeRepository.update_episode_metadata(episode_id, new_metadata)
            EpisodeRepository.update_status(episode_id, "completed")

            shutil.rmtree(session_dir)

            logger.info(f"[MANUAL] ✅ Finalized episode {episode_id} with manual mapping")
            return {
                "status": "completed",
                "xml_uri": final_gcs_uri,
                "processing_time_finalize": time.time() - start_time,
                "sync_applied": sync_result is not None,
                "master_file_id": master_file_id,
            }

        except Exception as e:
            logger.error(f"[MANUAL] finalize_with_manual_mapping failed: {e}", exc_info=True)
            EpisodeRepository.update_status(episode_id, "failed")
            if session_dir.exists():
                shutil.rmtree(session_dir)
            raise
