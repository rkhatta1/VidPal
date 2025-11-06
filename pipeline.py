# pipeline.py

import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import time
from datetime import datetime

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
from processing.camera_speaker_mapper import CameraSpeakerMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class VidPalAIPipeline:
    """Optimized VidPalAI multicam editing pipeline."""
    
    def __init__(self):
        """Initialize VidPalAI pipeline with all components."""
        self.settings = get_settings()
        
        # Initialize cache (only once!)
        self.cache = Cache(self.settings.CACHE_DIR) if self.settings.ENABLE_CACHING else None
        logger.info(f"Cache: {'enabled' if self.cache else 'disabled'}")
        
        # Initialize RAG store for embeddings and VLM retrieval
        self.rag_store = PGVectorRAGStore() if self.settings.USE_RAG else None
        logger.info(f"RAG store: {'enabled' if self.rag_store else 'disabled'}")
        
        # Initialize audio processing
        self.speaker_identifier = SpeakerIdentifier(cache=self.cache)
        logger.info("✅ Speaker identifier initialized")
        
        # Initialize EDL generation (always used)
        self.edl_generator = RuleBasedEDLGenerator(
            fps=self.settings.FRAME_RATE,
            min_shot_s=self.settings.MIN_SHOT_DURATION,
            wide_open_s=self.settings.WIDE_OPENING_DURATION,
            rapid_window_s=self.settings.RAPID_WINDOW_SECONDS,
            rapid_changes=self.settings.RAPID_CHANGES_THRESHOLD,
            reaction_keywords=self.settings.REACTION_KEYWORDS,
        )
        logger.info("✅ EDL generator initialized")
        
        # Initialize LLM refiner (optional)
        if self.settings.REFINE_WITH_LLM:
            self.llm_refiner = LLMRefiner(rag_store=self.rag_store)
            logger.info("✅ LLM refiner initialized")
        else:
            self.llm_refiner = None
            logger.info("LLM refiner: disabled")
        
        # Initialize VLM processor (optional)
        if self.settings.ENABLE_VLM_PROCESSING:
            self.vlm_processor = VLMProcessor(cache=self.cache)
            logger.info("✅ VLM processor initialized")
        else:
            self.vlm_processor = None
            logger.info("VLM processor: disabled")
        
        # Initialize FCPXML generator (always used)
        self.fcpxml_generator = FCPXMLGenerator()
        logger.info("✅ FCPXML generator initialized")
        
        logger.info("="*60)
        logger.info("✅ VidPalAI pipeline fully initialized")
        logger.info("="*60)
    
    def _generate_deterministic_episode_id(
        self,
        video_paths: Dict[str, Path],
        audio_path: Path,
        duration_minutes: int,
    ) -> str:
        """
        Generate deterministic episode_id based on input files.
        Same files and duration = same episode_id, enabling proper caching.
        
        Args:
            video_paths: Dictionary of camera_id -> video path
            audio_path: Master audio file path
            duration_minutes: Processing duration in minutes
        
        Returns:
            Deterministic episode_id
        """
        # Create a stable hash from absolute paths and duration
        # This ensures same inputs always generate the same ID
        hash_input = "|".join([
            str(audio_path.resolve()),  # Absolute path to audio
            str(sorted([(k, str(v.resolve())) for k, v in video_paths.items()])),  # Sorted video paths
            str(duration_minutes),  # Duration component
        ])
        
        # Generate SHA256 hash (first 12 chars for readability)
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        
        # Create readable episode ID
        episode_id = f"ep_{hash_digest}"
        
        logger.debug(f"Generated deterministic episode ID: {episode_id}")
        logger.debug(f"Hash input: {hash_input}")
        
        return episode_id
    
    def process_episode(
        self,
        title: Optional[str] = None,
        audio_path: Optional[Path] = None,
        video_paths: Optional[Dict[str, Path]] = None,
        duration_minutes: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete episode through the pipeline.
        
        Args:
            title: Episode title
            audio_path: Path to master audio file
            video_paths: Dictionary of camera_id -> video file path
            duration_minutes: Optional processing duration limit
            output_dir: Output directory
        
        Returns:
            Dictionary with processing results and output paths
        """
        start_time = time.time()
        
        # Setup defaults
        audio_path = audio_path or self.settings.MASTER_AUDIO_FILE
        video_paths = video_paths or self.settings.VIDEO_FILES
        duration_minutes = duration_minutes or self.settings.PROCESS_DURATION_MINUTES
        output_dir = output_dir or self.settings.OUTPUT_DIR
        duration_seconds = duration_minutes * 60
        
        # ✅ GENERATE DETERMINISTIC EPISODE_ID (NOT random hash with timestamp)
        episode_id = self._generate_deterministic_episode_id(
            video_paths=video_paths,
            audio_path=Path(audio_path),
            duration_minutes=duration_minutes,
        )
        
        logger.info("\n" + "="*60)
        logger.info(f"Episode ID (deterministic): {episode_id}")
        if title:
            logger.info(f"Title: {title}")
        logger.info(f"Duration: {duration_minutes} minutes ({duration_seconds:.0f}s)")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Cameras: {list(video_paths.keys())}")
        logger.info("="*60 + "\n")
        
        # Create episode record
        EpisodeRepository.create_episode(
            episode_id=episode_id,
            title=title or f"Episode_{episode_id}",
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
            
            # Speaker identification now also returns the transcript
            speaker_segments, role_mapping, transcript = self.speaker_identifier.identify_speakers(
                audio_path=str(audio_path),
                episode_id=episode_id,
                duration_limit_seconds=duration_seconds,
            )
            
            logger.info(f"✅ Phase 1 completed in {time.time() - phase_start:.1f}s")

            # ===== PHASE 1.5: Automatic Speaker-Camera Mapping =====
            logger.info("\n" + "="*60)
            logger.info("PHASE 1.5: Automatic Speaker-Camera Mapping")
            logger.info("="*60)
            phase_start = time.time()

            camera_speaker_mapper = CameraSpeakerMapper()
            camera_to_speaker_map = camera_speaker_mapper.map_speakers_to_cameras(
                video_paths=video_paths,
                speaker_segments=speaker_segments,
            )

            # Update role_mapping with the new camera-to-speaker mapping
            # This assumes the speaker_id from the mapper is the ground truth
            for camera_id, speaker_id in camera_to_speaker_map.items():
                if speaker_id in role_mapping:
                    # We can now create a more descriptive role, e.g., "host (cam_a)"
                    role_mapping[speaker_id] = f"{role_mapping[speaker_id]} ({camera_id})"

            # Build the dynamic CAMERA_SETUP
            dynamic_camera_setup = {
                cam: f"Speaker: {spk}" for cam, spk in camera_to_speaker_map.items()
            }
            
            # Find the wide camera and add it with a generic description
            wide_camera_id = next((cam_id for cam_id in video_paths if "wide" in cam_id.lower()), None)
            if wide_camera_id:
                all_speakers = sorted(list(set(role_mapping.values())))
                dynamic_camera_setup[wide_camera_id] = f"All participants ({', '.join(all_speakers)})"

            self.settings.CAMERA_SETUP = dynamic_camera_setup
            logger.info(f"Dynamic CAMERA_SETUP: {self.settings.CAMERA_SETUP}")

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
                    vlm_descriptions=vlm_descriptions,
                )
                
                logger.info(f"✅ Phase 5 completed in {time.time() - phase_start:.1f}s")
            
            # Save final EDL to database
            EpisodeRepository.save_edl_cuts(episode_id, cuts)
            
            # ===== PHASE 6: FCPXML Generation =====
            logger.info("\n" + "="*60)
            logger.info("PHASE 6: FCPXML Generation")
            logger.info("="*60)
            phase_start = time.time()
            
            output_path = Path(output_dir) / f"{episode_id}.fcpxml"
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
            logger.info("="*60 + "\n")
            
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
