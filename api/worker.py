# api/worker.py

from celery import Celery
from config import get_settings
from pipeline import VidPalAIPipeline
from processing.sync.exceptions import SyncError
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "vidpal_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

@celery_app.task(name='worker.run_pipeline', bind=True)
def run_pipeline(self, episode_id, audio_url, video_urls, title, sync_options=None):
    """
    Main pipeline task.
    
    Args:
        episode_id: Unique episode identifier
        audio_url: GCS URI for master audio (can be None)
        video_urls: Dict of camera_id -> GCS URI
        title: Episode title
        sync_options: Optional sync configuration dict
    """
    logger.info(f"Worker received task for episode {episode_id}")
    logger.info(f"  Audio URL: {audio_url or '(will extract from master video)'}")
    logger.info(f"  Video files: {list(video_urls.keys())}")
    logger.info(f"  Sync options: {sync_options}")
    
    try:
        pipeline = VidPalAIPipeline()
        result = pipeline.process_episode(
            episode_id=episode_id,
            audio_gcs_uri=audio_url,  # Can be None
            video_gcs_uris=video_urls,
            title=title,
            sync_options=sync_options,
        )
        return result
    except SyncError as e:
        logger.error(f"Sync error: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "error_type": "sync_error",
        }
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise e
