
# db/models.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from db.connection import db

logger = logging.getLogger(__name__)


class EpisodeRepository:
    """Repository for episode data management."""
    
    @staticmethod
    def create_episode(
        episode_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        audio_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a new episode record."""
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO episodes 
                (episode_id, title, description, duration_seconds, audio_path, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (episode_id) DO UPDATE 
                SET title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    duration_seconds = EXCLUDED.duration_seconds,
                    audio_path = EXCLUDED.audio_path,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (episode_id, title, description, duration_seconds, audio_path, 
                 json.dumps(metadata) if metadata else None)
            )
            result = cursor.fetchone()
            return result['id']
    
    @staticmethod
    def update_status(episode_id: str, status: str) -> None:
        """Update episode processing status."""
        with db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE episodes SET processing_status = %s WHERE episode_id = %s",
                (status, episode_id)
            )
    
    @staticmethod
    def get_episode(episode_id: str) -> Optional[Dict[str, Any]]:
        """Get episode by ID."""
        with db.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM episodes WHERE episode_id = %s",
                (episode_id,)
            )
            return cursor.fetchone()
    
    @staticmethod
    def add_video_file(
        episode_id: str,
        camera_id: str,
        file_path: str,
        file_hash: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add video file record."""
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO video_files 
                (episode_id, camera_id, file_path, file_hash, duration_seconds, 
                 width, height, fps, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (episode_id, camera_id) DO UPDATE
                SET file_path = EXCLUDED.file_path,
                    file_hash = EXCLUDED.file_hash,
                    duration_seconds = EXCLUDED.duration_seconds,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    fps = EXCLUDED.fps,
                    metadata = EXCLUDED.metadata
                RETURNING id
                """,
                (episode_id, camera_id, file_path, file_hash, duration_seconds,
                 width, height, fps, json.dumps(metadata) if metadata else None)
            )
            result = cursor.fetchone()
            return result['id']
    
    @staticmethod
    def save_speaker_segments(
        episode_id: str,
        speaker_segments: List[Dict[str, Any]],
    ) -> None:
        """Save speaker diarization segments."""
        with db.get_cursor() as cursor:
            # Delete existing segments
            cursor.execute(
                "DELETE FROM speaker_segments WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert new segments
            for seg in speaker_segments:
                cursor.execute(
                    """
                    INSERT INTO speaker_segments 
                    (episode_id, speaker_id, start_time, end_time, text, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        seg['speaker_id'],
                        seg['start'],
                        seg['end'],
                        seg.get('text', ''),
                        seg.get('confidence', None),
                    )
                )
    
    @staticmethod
    def save_speakers(
        episode_id: str,
        role_mapping: Dict[str, str],
        speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Save speaker information."""
        with db.get_cursor() as cursor:
            # Delete existing speakers
            cursor.execute(
                "DELETE FROM speakers WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert speakers
            for speaker_id, role in role_mapping.items():
                stats = speaker_stats.get(speaker_id, {}) if speaker_stats else {}
                cursor.execute(
                    """
                    INSERT INTO speakers 
                    (episode_id, speaker_id, role, total_duration, segment_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        speaker_id,
                        role,
                        stats.get('total_duration', 0.0),
                        stats.get('segment_count', 0),
                        json.dumps(stats) if stats else None,
                    )
                )
    
    @staticmethod
    def save_edl_cuts(episode_id: str, cuts: List[Dict[str, Any]]) -> None:
        """
        Save EDL cuts to database.
        Handles both dict and object formats, with safe field access.
        
        Args:
            episode_id: Episode identifier
            cuts: List of cut dictionaries
        """
        if not cuts:
            logger.warning("No cuts to save")
            return
        
        logger.info(f"Saving {len(cuts)} EDL cuts for episode {episode_id}")
        
        with db.get_cursor() as cursor:
            # Delete existing cuts for this episode
            cursor.execute(
                "DELETE FROM edl_cuts WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert new cuts
            for i, cut in enumerate(cuts):
                try:
                    # Handle both dict and object formats
                    # Use .get() for safe access with defaults
                    start_time = cut.get('start_time') or getattr(cut, 'start_time', None)
                    end_time = cut.get('end_time') or getattr(cut, 'end_time', None)
                    camera_id = cut.get('camera_id') or getattr(cut, 'camera_id', None)
                    reason = cut.get('reason') or getattr(cut, 'reason', f'cut_{i}')
                    
                    # Validate required fields
                    if start_time is None or end_time is None or camera_id is None:
                        logger.error(f"Cut {i} missing required fields: {cut}")
                        logger.error(f"  start_time: {start_time}, end_time: {end_time}, camera_id: {camera_id}")
                        continue
                    
                    # Convert to float
                    start_time = float(start_time)
                    end_time = float(end_time)
                    
                    # Insert cut
                    cursor.execute(
                        """
                        INSERT INTO edl_cuts 
                        (episode_id, sequence_order, start_time, end_time, camera_id, reason)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            episode_id,
                            i,
                            start_time,
                            end_time,
                            str(camera_id),
                            str(reason),
                        )
                    )
                
                except Exception as e:
                    logger.error(f"Failed to insert cut {i}: {e}")
                    logger.error(f"Cut data: {cut}")
                    logger.error(f"Cut type: {type(cut)}")
                    raise
        
        logger.info(f"✅ Saved {len(cuts)} cuts to database")
    
    @staticmethod
    def get_edl_cuts(episode_id: str) -> List[Dict[str, Any]]:
        """Get EDL cuts for an episode."""
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT start_time, end_time, camera_id, reason
                FROM edl_cuts
                WHERE episode_id = %s
                ORDER BY sequence_order
                """,
                (episode_id,)
            )
            return cursor.fetchall()
