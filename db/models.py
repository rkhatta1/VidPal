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
    def save_edl_cuts(
        episode_id: str,
        cuts: List[Dict[str, Any]],
    ) -> None:
        """Save EDL cuts."""
        with db.get_cursor() as cursor:
            # Delete existing cuts
            cursor.execute(
                "DELETE FROM edl_cuts WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert new cuts
            for i, cut in enumerate(cuts):
                cursor.execute(
                    """
                    INSERT INTO edl_cuts 
                    (episode_id, start_time, end_time, camera_id, reason, sequence_order)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        cut['start_time'],
                        cut['end_time'],
                        cut['camera_id'],
                        cut.get('reason', 'speaker'),
                        i,
                    )
                )
    
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

    @staticmethod
    def save_reaction_events(
        episode_id: str,
        events: List[Dict[str, Any]],
    ) -> None:
        """Save detected reaction events to the database."""
        if not events:
            return

        with db.get_cursor() as cursor:
            # Clean up previous runs for this episode
            cursor.execute(
                "DELETE FROM reaction_events WHERE episode_id = %s",
                (episode_id,)
            )
            
            for event in events:
                cursor.execute(
                    """
                    INSERT INTO reaction_events 
                    (episode_id, camera_id, timestamp_seconds, event_type, score, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        event['camera'],
                        event['timestamp'],
                        event['event'],
                        event.get('score', 0.0),
                        json.dumps(event)
                    )
                )
    
    @staticmethod
    def get_reaction_events(
        episode_id: str, 
        camera_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve reaction events, optionally filtering by camera.
        Reconstructs the original event dictionary from the DB columns.
        """
        sql = "SELECT * FROM reaction_events WHERE episode_id = %s"
        params = [episode_id]
        
        if camera_id:
            sql += " AND camera_id = %s"
            params.append(camera_id)
            
        sql += " ORDER BY timestamp_seconds"
        
        with db.get_cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
        results = []
        for row in rows:
            # Reconstruct the event dictionary structure used by EmotionDetector
            results.append({
                "timestamp": row['timestamp_seconds'],
                "event": row['event_type'],
                "camera": row['camera_id'],
                "score": row['score']
            })
        return results

    @staticmethod
    def update_episode_metadata(episode_id: str, new_metadata: Dict[str, Any]) -> None:
        """Merges new metadata keys into the existing metadata JSON."""
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE episodes 
                SET metadata = coalesce(metadata, '{}'::jsonb) || %s
                WHERE episode_id = %s
                """,
                (json.dumps(new_metadata), episode_id)
            )


    @staticmethod
    def save_sync_result(episode_id: str, sync_result: 'SyncResult') -> None:
        """Save sync result and file offsets to database."""
        from processing.sync.models import SyncResult  # Avoid circular import
        
        with db.get_cursor() as cursor:
            # Upsert episode_sync
            cursor.execute(
                """
                INSERT INTO episode_sync 
                (episode_id, master_file_id, global_start, global_end, 
                 common_start, common_end, has_full_overlap, sync_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (episode_id) DO UPDATE SET
                    master_file_id = EXCLUDED.master_file_id,
                    global_start = EXCLUDED.global_start,
                    global_end = EXCLUDED.global_end,
                    common_start = EXCLUDED.common_start,
                    common_end = EXCLUDED.common_end,
                    has_full_overlap = EXCLUDED.has_full_overlap,
                    sync_method = EXCLUDED.sync_method
                """,
                (
                    episode_id,
                    sync_result.master_file_id,
                    sync_result.global_start,
                    sync_result.global_end,
                    sync_result.common_start,
                    sync_result.common_end,
                    sync_result.has_full_overlap,
                    'cross_correlation',
                )
            )
            
            # Delete old offsets
            cursor.execute(
                "DELETE FROM file_sync_offsets WHERE episode_id = %s",
                (episode_id,)
            )
            
            # Insert file offsets
            for file_id, offset in sync_result.file_offsets.items():
                cursor.execute(
                    """
                    INSERT INTO file_sync_offsets
                    (episode_id, file_id, offset_seconds, global_in_point, 
                     global_out_point, original_duration, sync_confidence, is_master)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id,
                        file_id,
                        offset.offset_seconds,
                        offset.global_in_point,
                        offset.global_out_point,
                        offset.original_duration,
                        offset.sync_confidence,
                        offset.is_master,
                    )
                )

    @staticmethod
    def get_sync_result(episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve sync result from database."""
        with db.get_cursor() as cursor:
            # Get main sync record
            cursor.execute(
                "SELECT * FROM episode_sync WHERE episode_id = %s",
                (episode_id,)
            )
            sync_row = cursor.fetchone()
            
            if not sync_row:
                return None
            
            # Get file offsets
            cursor.execute(
                "SELECT * FROM file_sync_offsets WHERE episode_id = %s",
                (episode_id,)
            )
            offset_rows = cursor.fetchall()
            
            file_offsets = {}
            for row in offset_rows:
                file_offsets[row['file_id']] = {
                    "file_id": row['file_id'],
                    "file_path": "",  # Not stored in DB, will be populated at runtime
                    "original_duration": row['original_duration'],
                    "offset_seconds": row['offset_seconds'],
                    "global_in_point": row['global_in_point'],
                    "global_out_point": row['global_out_point'],
                    "sync_confidence": row['sync_confidence'],
                }
            
            return {
                "master_file_id": sync_row['master_file_id'],
                "file_offsets": file_offsets,
                "global_start": sync_row['global_start'],
                "global_end": sync_row['global_end'],
                "common_start": sync_row['common_start'],
                "common_end": sync_row['common_end'],
                "has_full_overlap": sync_row['has_full_overlap'],
            }


    @staticmethod
    def get_video_files(episode_id: str) -> List[Dict[str, Any]]:
        """Return list of video files (camera_id + file_path) for an episode."""
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT camera_id, file_path, duration_seconds, width, height, fps, metadata
                FROM video_files
                WHERE episode_id = %s
                ORDER BY camera_id
                """,
                (episode_id,)
            )
            return cursor.fetchall()

    @staticmethod
    def get_speaker_segments(episode_id: str) -> List[Dict[str, Any]]:
        """
        Load diarization segments for an episode from the DB and
        normalize them to the shape expected by the EDL generator:
        {
            "speaker_id": str,
            "start": float,
            "end": float,
            "text": str,
            "confidence": float | None,
        }
        """
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT speaker_id, start_time, end_time, text, confidence
                FROM speaker_segments
                WHERE episode_id = %s
                ORDER BY start_time
                """,
                (episode_id,),
            )
            rows = cursor.fetchall()

        segments: List[Dict[str, Any]] = []
        for row in rows:
            segments.append(
                {
                    "speaker_id": row["speaker_id"],
                    "start": float(row["start_time"]),
                    "end": float(row["end_time"]),
                    "text": row.get("text") or "",
                    "confidence": (
                        float(row["confidence"])
                        if row.get("confidence") is not None
                        else None
                    ),
                }
            )
        return segments

    @staticmethod
    def get_role_mapping(episode_id: str) -> Dict[str, str]:
        """
        Reconstruct speaker_id -> role mapping from the speakers table.
        """
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT speaker_id, role
                FROM speakers
                WHERE episode_id = %s
                """,
                (episode_id,)
            )
            rows = cursor.fetchall()

        return {row["speaker_id"]: row["role"] for row in rows if row["role"]}
