# pipeline.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import uuid

from config import get_settings
from db.connection import db
from db.models import EpisodeRepository
from utils.caching import Cache
from rag.pgvector_store import PGVectorRAGStore
from processing.speaker_identification import SpeakerIdentifier
from edl.rules import RuleBasedEDLGenerator
from processing.llm_refiner import LLMRefiner
from fcpxml.generator import FCPXMLGenerator
from processing.vlm_processor import VLMProcessor
from processing.speaker_camera_mapping import SpeakerCameraMapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VidPalAIPipeline:
    """Optimized VidPalAI multicam editing pipeline."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize components
        self.cache = Cache(self.settings.CACHE_DIR) if self.settings.ENABLE_CACHING else None
        self.rag_store = PGVectorRAGStore() if self.settings.USE_RAG else None
        
        self.speaker_identifier = SpeakerIdentifier(cache=self.cache)
        self.speaker_camera_mapper = SpeakerCameraMapper()

        self.edl_generator = RuleBasedEDLGenerator(
            fps=self.settings.FRAME_RATE,
            min_shot_s=self.settings.MIN_SHOT_DURATION,
            wide_open_s=self.settings.WIDE_OPENING_DURATION,
            rapid_window_s=self.settings.RAPID_WINDOW_SECONDS,
            rapid_changes=self.settings.RAPID_CHANGES_THRESHOLD,
            reaction_keywords=self.settings.REACTION_KEYWORDS,
        )
        
        if self.settings.REFINE_WITH_LLM:
            self.llm_refiner = LLMRefiner(rag_store=self.rag_store)
        else:
            self.llm_refiner = None
            
        if self.settings.ENABLE_VLM_PROCESSING:
            self.vlm_processor = VLMProcessor()
        else:
            self.vlm_processor = None
        
        self.fcpxml_generator = FCPXMLGenerator()
        
        logger.info("✅ VidPalAI pipeline initialized")
    
    def process_episode(
        self,
        episode_id: Optional[str] = None,
        title: Optional[str] = None,
        audio_path: Optional[Path] = None,
        video_paths: Optional[Dict[str, Path]] = None,
        duration_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete episode through the pipeline.
        
        Args:
            episode_id: Unique episode identifier (auto-generated if None)
            title: Episode title
            audio_path: Path to master audio file
            video_paths: Dictionary of camera_id -> video file path
            duration_minutes: Optional processing duration limit
        
        Returns:
            Dictionary with processing results and output paths
        """
        start_time = time.time()
        
        # Setup
        if episode_id is None:
            episode_id = f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        audio_path = audio_path or self.settings.MASTER_AUDIO_FILE
        video_paths = video_paths or self.settings.VIDEO_FILES
        duration_seconds = (duration_minutes or self.settings.PROCESS_DURATION_MINUTES) * 60
        
        logger.info(f"🎬 Processing episode: {episode_id}")
        logger.info(f"   Duration: {duration_seconds/60:.1f} minutes")
        logger.info(f"   Audio: {audio_path}")
        logger.info(f"   Cameras: {list(video_paths.keys())}")
        
        # Create episode record
        EpisodeRepository.create_episode(
            episode_id=episode_id,
            title=title,
            duration_seconds=duration_seconds,
            audio_path=str(audio_path),
        )
        EpisodeRepository.update_status(episode_id, "processing")
        
        # Add video file records
        for camera_id, video_path in video_paths.items():
            EpisodeRepository.add_video_file(
                episode_id=episode_id,
                camera_id=camera_id,
                file_path=str(video_path),
            )
        
        try:
            # ===== PHASE 1: Audio Processing =====
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: Audio Processing")
            logger.info("="*60)
            
            phase_start = time.time()
            
            # Speaker identification (includes diarization)
            speaker_segments, role_mapping, transcript, cache_key = self.speaker_identifier.identify_speakers(
                audio_path=str(audio_path),
                episode_id=episode_id,
                duration_limit_seconds=duration_seconds,
            )
            
            # Audio transcription
            # transcript = self.audio_transcriber.transcribe(
            #     audio_path=str(audio_path),
            #     duration_limit_seconds=duration_seconds,
            #     speaker_segments=speaker_segments,
            # )
            
            logger.info(f"✅ Phase 1 completed in {time.time() - phase_start:.1f}s")

            # PHASE 1.5 - Speaker to Camera Mapping
            logger.info("\n" + "="*60)
            logger.info("PHASE 1.5: Speaker-Camera Mapping")
            logger.info("="*60)
            
            phase_start = time.time()
            role_camera_map = self.cache.get(cache_key, "speaker_camera_map") if self.cache and cache_key else None
            
            if not role_camera_map:
                logger.info("Cache miss. Running Speaker-Camera mapping...")
                role_camera_map = self.speaker_camera_mapper.map_roles_to_cameras(
                    speaker_segments=speaker_segments,
                    role_mapping=role_mapping,
                    video_paths=video_paths
                )
                if self.cache and cache_key:
                    self.cache.set(cache_key, "speaker_camera_map", role_camera_map)
            else:
                logger.info("✅ Using cached speaker-camera map")
            
            logger.info(f"✅ Phase 1.5 completed in {time.time() - phase_start:.1f}s")
            
            # ===== PHASE 2: RAG Ingestion (Transcript) =====
            if self.rag_store:
                logger.info("\n" + "="*60)
                logger.info("PHASE 2: RAG Ingestion (Transcript)")
                logger.info("="*60)
                
                phase_start = time.time()
                
                self.rag_store.ingest_transcript_chunks(
                    episode_id=episode_id,
                    transcript=transcript,
                    chunk_duration=self.settings.RAG_CHUNK_SIZE,
                )
                
                logger.info(f"✅ Phase 2 completed in {time.time() - phase_start:.1f}s")
            
            # ===== PHASE 3: EDL Generation (Rule-Based) =====
            logger.info("\n" + "="*60)
            logger.info("PHASE 3: EDL Generation (Rule-Based)")
            logger.info("="*60)
            
            phase_start = time.time()
            
            edl_result = self.edl_generator.generate_edl(
                speaker_segments=speaker_segments,
                role_mapping=role_mapping,
                role_camera_map=role_camera_map,
                transcript=transcript,
                start_time=0.0,
                end_time=duration_seconds,
            )
            
            cuts = edl_result["cuts"]
            logger.info(f"Generated {len(cuts)} rule-based cuts")
            
            logger.info(f"✅ Phase 3 completed in {time.time() - phase_start:.1f}s")
            
            # ===== PHASE 4: VLM Processing (Optional) =====
            vlm_descriptions = []
            if self.vlm_processor:
                logger.info("\n" + "="*60)
                logger.info("PHASE 4: VLM Scene Analysis")
                logger.info("="*60)
                
                phase_start = time.time()
                vlm_descriptions = self.cache.get(cache_key, "vlm_descriptions") if self.cache and cache_key else None
                
                if not vlm_descriptions:
                    logger.info("Cache miss. Running VLM processing...")
                    vlm_descriptions = self.vlm_processor.process_scene_transitions(
                        video_paths=video_paths,
                        cuts=cuts,
                        episode_id=episode_id,
                    )
                    
                    # Store VLM descriptions in RAG
                    if self.rag_store and vlm_descriptions:
                        self.rag_store.ingest_vlm_descriptions(
                            episode_id=episode_id,
                            vlm_descriptions=vlm_descriptions,
                        )
                    
                    # Store in cache
                    if self.cache and cache_key:
                        self.cache.set(cache_key, "vlm_descriptions", vlm_descriptions)
                else:
                    logger.info("✅ Using cached VLM descriptions")
                
                logger.info(f"✅ Phase 4 completed in {time.time() - phase_start:.1f}s")
            
            # ===== PHASE 5: LLM Refinement (Optional) =====
            if self.llm_refiner:
                logger.info("\n" + "="*60)
                logger.info("PHASE 5: LLM EDL Refinement")
                logger.info("="*60)
                
                phase_start = time.time()
                
                cuts = self.llm_refiner.refine_edl(
                    episode_id=episode_id,
                    base_edl=cuts,
                    transcript=transcript,
                    role_mapping=role_mapping,
                    vlm_descriptions=vlm_descriptions,  # Pass VLM context
                )
                
                logger.info(f"✅ Phase 5 completed in {time.time() - phase_start:.1f}s")
            
            # Save final EDL to database
            EpisodeRepository.save_edl_cuts(episode_id, cuts)
            
            # ===== PHASE 6: FCPXML Generation =====
            logger.info("\n" + "="*60)
            logger.info("PHASE 6: FCPXML Generation")
            logger.info("="*60)
            
            phase_start = time.time()
            
            output_path = self.settings.OUTPUT_DIR / f"{episode_id}.fcpxml"
            self.fcpxml_generator.generate(
                cuts=cuts,
                video_paths=video_paths,
                output_path=output_path,
                episode_id=episode_id,
                master_audio_path=audio_path,
            )
            
            logger.info(f"✅ Phase 6 completed in {time.time() - phase_start:.1f}s")
            
            # Update status
            EpisodeRepository.update_status(episode_id, "completed")
            
            # Summary
            total_time = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("🎉 PROCESSING COMPLETE")
            logger.info("="*60)
            logger.info(f"Episode ID: {episode_id}")
            logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            logger.info(f"Cuts generated: {len(cuts)}")
            logger.info(f"Output: {output_path}")
            logger.info("="*60)
            
            return {
                "episode_id": episode_id,
                "processing_time": total_time,
                "cuts_count": len(cuts),
                "output_path": str(output_path),
                "speakers": len(role_mapping),
                "transcript_words": len(transcript),
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            EpisodeRepository.update_status(episode_id, "failed")
            raise


def main():
    """Main entry point."""
    pipeline = VidPalAIPipeline()
    
    result = pipeline.process_episode(
        title="Sample Episode",
    )
    
    logger.info(f"\n✅ Results: {result}")


if __name__ == "__main__":
    main()
